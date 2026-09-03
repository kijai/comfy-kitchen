/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// fp16-accumulate 3D convolution (implicit GEMM, NDHWC) with bias and an
// optional residual fused into the epilogue: D = (acc + bias[k]) + residual.
//
// Same rationale as cutlass_gemm_fp16.cu: on sm_120 the fp32-accumulate HMMA
// that cuDNN uses issues at half the rate of fp16-accumulate, and PyTorch has
// no way to ask cuDNN for fp16 accumulation. Zero padding only -- callers
// pre-pad (see group_norm_pad3d.cu, whose output layout this consumes
// directly). Bias is broadcast through the epilogue's vector input; a missing
// residual is a stride-0 view of a zero vector, so one kernel family serves
// both cases.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

struct Conv3dDims {
    int N, D, H, W, C;      // input NDHWC
    int K, T, R, S;         // filter KTRSC
    int Z, P, Q;            // output NZPQK
    int sd, sh, sw;         // strides
};

#ifdef COMFY_HAVE_CUTLASS

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/conv/kernel/default_conv3d_fprop_with_broadcast.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"

namespace {

using half_t = cutlass::half_t;
using Layout = cutlass::layout::TensorNDHWC;

// Z = identity(acc + bias) + residual, computed in fp32, rounded once to fp16.
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
    half_t, half_t, float, half_t, 8,
    cutlass::epilogue::thread::Identity, cutlass::plus, cutlass::epilogue::thread::Identity>;

template <int TBM, int TBN, int Stages>
struct Conv3dFp16 {
    using Kernel = typename cutlass::conv::kernel::DefaultConv3dFpropWithBroadcast<
        half_t, Layout, half_t, Layout, half_t, Layout, half_t,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TBM, TBN, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>, EpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;
    using Op = cutlass::conv::device::ImplicitGemmConvolution<Kernel>;

    static bool run(const half_t* x, const half_t* w, const half_t* bias,
                    const half_t* resid, bool resid_full, half_t* out,
                    const Conv3dDims& d, cudaStream_t stream) {
        const cutlass::Tensor5DCoord in_size(d.N, d.D, d.H, d.W, d.C);
        const cutlass::Tensor5DCoord filter_size(d.K, d.T, d.R, d.S, d.C);
        const cutlass::Tensor5DCoord out_size(d.N, d.Z, d.P, d.Q, d.K);
        cutlass::conv::Conv3dProblemSize problem(
            in_size, filter_size,
            cutlass::make_Coord(0, 0, 0), cutlass::make_Coord(d.sd, d.sh, d.sw), cutlass::make_Coord(1, 1, 1),
            out_size, cutlass::conv::Mode::kCrossCorrelation, 1, 1);
        const Layout lx = Layout::packed(in_size);
        const Layout lw = Layout::packed(filter_size);
        const Layout lo = Layout::packed(out_size);
        const Layout lr = resid_full ? lo : Layout(0, 0, 0, 0);  // every output row reads resid[0:K]

        // TensorRef holds non-const pointers; the kernel only reads x/w/resid.
        typename Op::Arguments args(
            problem,
            typename Kernel::TensorRefA(const_cast<half_t*>(x), lx),
            typename Kernel::TensorRefB(const_cast<half_t*>(w), lw),
            typename Kernel::TensorRefC(const_cast<half_t*>(resid), lr),
            typename Kernel::TensorRefC(out, lo),
            typename EpilogueOp::Params(1.0f, 1.0f), cutlass::conv::SplitKMode::kSerial,
            const_cast<half_t*>(bias), nullptr, /*ldr=*/0, /*ldt=*/d.K);

        Op op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return false;
        if (Op::get_workspace_size(args) != 0) return false;
        if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return false;
        return op.run(stream) == cutlass::Status::kSuccess;
    }
};

// Below this many threadblocks the launch cannot fill the GPU; those (deep-K,
// low-resolution) convs stay on cuDNN, which also keeps them fp32-accumulate.
constexpr int64_t kMinTiles = 128;

}  // namespace

extern "C" bool launch_cutlass_fp16_conv3d(
    const void* x, const void* w, const void* bias, const void* resid, bool resid_full, void* out,
    int N, int D, int H, int W, int C, int K, int T, int R, int S, int Z, int P, int Q,
    int sd, int sh, int sw, cudaStream_t stream) {
    const Conv3dDims d{N, D, H, W, C, K, T, R, S, Z, P, Q, sd, sh, sw};
    if (d.C % 8 != 0 || d.K % 8 != 0) return false;
    const int64_t m = static_cast<int64_t>(d.N) * d.Z * d.P * d.Q;
    const int64_t m_tiles = (m + 127) / 128;
    const auto xp = static_cast<const half_t*>(x);
    const auto wp = static_cast<const half_t*>(w);
    const auto bp = static_cast<const half_t*>(bias);
    const auto rp = static_cast<const half_t*>(resid);
    const auto op = static_cast<half_t*>(out);
    if (d.K >= 256) {
        if (m_tiles * ((d.K + 255) / 256) < kMinTiles) return false;
        return Conv3dFp16<128, 256, 3>::run(xp, wp, bp, rp, resid_full, op, d, stream);
    }
    if (m_tiles * ((d.K + 127) / 128) < kMinTiles) return false;
    return Conv3dFp16<128, 128, 4>::run(xp, wp, bp, rp, resid_full, op, d, stream);
}

#else  // !COMFY_HAVE_CUTLASS -- stub; caller falls back to torch's conv.

extern "C" bool launch_cutlass_fp16_conv3d(
    const void*, const void*, const void*, const void*, bool, void*,
    int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, cudaStream_t) {
    return false;
}

#endif
