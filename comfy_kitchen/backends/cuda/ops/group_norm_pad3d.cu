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
// group_norm_silu_pad3d: per-frame GroupNorm, SiLU and the causal-conv padding
// of a 3D CNN (reflect in space, zero frames in front) in one pass, in NDHWC.
//
// The eager sequence is permute+contiguous, GroupNorm statistics, normalize,
// SiLU, reflect pad and constant pad: seven memory passes plus the NCDHW<->NDHWC
// conversions cuDNN inserts around every conv. Here x is read twice (once for
// statistics, once to normalize) and the padded output is written once, in
// the NDHWC layout cuDNN runs natively.
//
//   stats_partial:  per (frame, row chunk, channel) sum / sum of squares
//   stats_finalize: reduce chunks and channels per group -> mean, rstd
//   apply:          y = silu((x - mean) * rstd * gamma + beta) written into
//                   the padded output, reflected spatial indices computed on
//                   the store side; the leading `front` frames are zero.
//
// x is [frames, rows, C] with rows = H*W; channels innermost. C must be a
// multiple of 8 (16-byte vectors) and C/8 must divide the block size.
// Statistics are sum / sum-of-squares in fp32 over 1024-row chunks, reduced in
// double; torch uses Welford, so the two agree to fp32 rounding, not bitwise.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

#include "dtype_dispatch.cuh"

namespace {

constexpr int kThreads = 256;
constexpr int kVec = 8;            // 16-byte vectors of half / bf16
constexpr int kChunkRows = 1024;   // rows per statistics block

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<half>(half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ half from_f<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ nv_bfloat16 from_f<nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ int reflect_index(int i, int n) {
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
stats_partial_kernel(const T* __restrict__ x, float2* __restrict__ partial,
                     int64_t rows, int C, int chunks) {
    const int cv = C / kVec;
    const int rows_per_iter = kThreads / cv;
    const int frame = blockIdx.y;
    const int chunk = blockIdx.x;
    const int cvec = threadIdx.x % cv;
    const int r0 = threadIdx.x / cv;
    const int64_t row_begin = static_cast<int64_t>(chunk) * kChunkRows;
    const int64_t row_end = min(row_begin + kChunkRows, rows);
    const T* base = x + frame * rows * C + cvec * kVec;

    float s[kVec], ss[kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) { s[i] = 0.f; ss[i] = 0.f; }
    for (int64_t r = row_begin + r0; r < row_end; r += rows_per_iter) {
        uint4 raw = *reinterpret_cast<const uint4*>(base + r * C);
        const T* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            float f = to_f(v[i]);
            s[i] += f;
            ss[i] += f * f;
        }
    }

    __shared__ float sh_s[kThreads][kVec];
    __shared__ float sh_ss[kThreads][kVec];
#pragma unroll
    for (int i = 0; i < kVec; ++i) { sh_s[threadIdx.x][i] = s[i]; sh_ss[threadIdx.x][i] = ss[i]; }
    __syncthreads();
    if (r0 == 0) {
        float2* dst = partial + (static_cast<int64_t>(frame) * chunks + chunk) * C + cvec * kVec;
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
            float ts = 0.f, tss = 0.f;
            for (int j = 0; j < rows_per_iter; ++j) {
                ts += sh_s[j * cv + cvec][i];
                tss += sh_ss[j * cv + cvec][i];
            }
            dst[i] = make_float2(ts, tss);
        }
    }
}

// One block per frame, one thread per group.
__global__ void stats_finalize_kernel(const float2* __restrict__ partial, float2* __restrict__ stats,
                                      int C, int G, int chunks, double n_per_group, float eps) {
    const int frame = blockIdx.x;
    const int g = threadIdx.x;
    const int cpg = C / G;
    double s = 0.0, ss = 0.0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const float2* p = partial + (static_cast<int64_t>(frame) * chunks + chunk) * C + g * cpg;
        for (int c = 0; c < cpg; ++c) { s += p[c].x; ss += p[c].y; }
    }
    const double mean = s / n_per_group;
    const double var = fmax(ss / n_per_group - mean * mean, 0.0);
    stats[frame * G + g] = make_float2(static_cast<float>(mean),
                                       rsqrtf(static_cast<float>(var) + eps));
}

template <typename T, bool kNorm, bool kSilu>
__global__ void __launch_bounds__(kThreads)
apply_kernel(const T* __restrict__ x, const float2* __restrict__ stats,
             const T* __restrict__ gamma, const T* __restrict__ beta, T* __restrict__ out,
             int T_in, int H, int W, int C, int G,
             int T_out, int H_out, int W_out, int left, int top, int front) {
    extern __shared__ float sh[];
    float* a = sh;
    float* b = sh + C;

    const int cv = C / kVec;
    const int oframe = blockIdx.y;
    const int bidx = oframe / T_out;
    const int t = oframe % T_out - front;
    const int64_t out_rows = static_cast<int64_t>(H_out) * W_out;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    const bool in_range = idx < out_rows * cv;
    T* dst = in_range ? out + (oframe * out_rows + idx / cv) * C + (idx % cv) * kVec : nullptr;

    if (t < 0) {
        if (in_range) *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        return;
    }
    const int iframe = bidx * T_in + t;
    if (kNorm) {
        const int cpg = C / G;
        for (int c = threadIdx.x; c < C; c += kThreads) {
            const float2 st = stats[iframe * G + c / cpg];
            a[c] = st.y * to_f(gamma[c]);
            b[c] = to_f(beta[c]) - st.x * a[c];
        }
        __syncthreads();
    }
    if (!in_range) return;

    const int orow = static_cast<int>(idx / cv);
    const int c0 = static_cast<int>(idx % cv) * kVec;
    const int y = reflect_index(orow / W_out - top, H);
    const int xx = reflect_index(orow % W_out - left, W);
    const T* src = x + ((static_cast<int64_t>(iframe) * H + y) * W + xx) * C + c0;

    uint4 raw = *reinterpret_cast<const uint4*>(src);
    T* v = reinterpret_cast<T*>(&raw);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
        float f = to_f(v[i]);
        if (kNorm) f = to_f(from_f<T>(a[c0 + i] * f + b[c0 + i]));  // round like the eager GroupNorm output
        if (kSilu) f = f / (1.f + expf(-f));
        v[i] = from_f<T>(f);
    }
    *reinterpret_cast<uint4*>(dst) = raw;
}

template <typename T>
void launch_typed(const T* x, const T* gamma, const T* beta, T* out, float2* workspace,
                  int B, int C, int T_in, int H, int W, int G, float eps,
                  int left, int right, int top, int bottom, int front, bool silu,
                  cudaStream_t stream) {
    const bool norm = gamma != nullptr;
    const int frames = B * T_in;
    const int64_t rows = static_cast<int64_t>(H) * W;
    const int chunks = static_cast<int>((rows + kChunkRows - 1) / kChunkRows);
    float2* partial = workspace;
    float2* stats = workspace + static_cast<int64_t>(frames) * chunks * C;

    if (norm) {
        stats_partial_kernel<T><<<dim3(chunks, frames), kThreads, 0, stream>>>(x, partial, rows, C, chunks);
        stats_finalize_kernel<<<frames, G, 0, stream>>>(
            partial, stats, C, G, chunks, static_cast<double>(rows) * (C / G), eps);
    }

    const int T_out = T_in + front;
    const int H_out = H + top + bottom;
    const int W_out = W + left + right;
    const int64_t out_vecs = static_cast<int64_t>(H_out) * W_out * (C / kVec);
    const dim3 grid(static_cast<unsigned>((out_vecs + kThreads - 1) / kThreads), B * T_out);
    const size_t smem = norm ? 2 * C * sizeof(float) : 0;
#define LAUNCH_APPLY(N, S)                                                            \
    apply_kernel<T, N, S><<<grid, kThreads, smem, stream>>>(                          \
        x, stats, gamma, beta, out, T_in, H, W, C, G, T_out, H_out, W_out, left, top, front)
    if (norm && silu) LAUNCH_APPLY(true, true);
    else if (norm) LAUNCH_APPLY(true, false);
    else if (silu) LAUNCH_APPLY(false, true);
    else LAUNCH_APPLY(false, false);
#undef LAUNCH_APPLY
}

}  // namespace

// x/out: NDHWC (channels_last_3d) of [B, C, T, H, W] / [B, C, T+front, H+top+bottom, W+left+right].
// gamma/beta: [C] in the same dtype, or both nullptr for pad-only. workspace:
// (B*T*ceil(H*W/1024)*C + B*T*G) float2, unused without gamma.
extern "C" void launch_group_norm_silu_pad3d(
    const void* x, const void* gamma, const void* beta, void* out, void* workspace,
    int B, int C, int T, int H, int W, int G, float eps,
    int left, int right, int top, int bottom, int front, bool silu,
    int dtype_code, cudaStream_t stream) {
    if (C % kVec != 0 || kThreads % (C / kVec) != 0 || (gamma != nullptr && (C % G != 0 || G > 1024))) {
        throw std::runtime_error("group_norm_silu_pad3d: unsupported channel/group count");
    }
    if (left >= W || right >= W || top >= H || bottom >= H) {
        throw std::runtime_error("group_norm_silu_pad3d: reflect padding must be smaller than the input");
    }
    // frames ride on grid.y; an oversize launch would fail silently and return `out` uninitialized
    if (static_cast<int64_t>(B) * (T + front) > 65535) {
        throw std::runtime_error("group_norm_silu_pad3d: batch * frames must be <= 65535");
    }
    switch (dtype_code) {
        case comfy::DTYPE_CODE_FLOAT16:
            launch_typed<half>(static_cast<const half*>(x), static_cast<const half*>(gamma),
                               static_cast<const half*>(beta), static_cast<half*>(out),
                               static_cast<float2*>(workspace), B, C, T, H, W, G, eps,
                               left, right, top, bottom, front, silu, stream);
            break;
        case comfy::DTYPE_CODE_BFLOAT16:
            launch_typed<nv_bfloat16>(static_cast<const nv_bfloat16*>(x), static_cast<const nv_bfloat16*>(gamma),
                                      static_cast<const nv_bfloat16*>(beta), static_cast<nv_bfloat16*>(out),
                                      static_cast<float2*>(workspace), B, C, T, H, W, G, eps,
                                      left, right, top, bottom, front, silu, stream);
            break;
        default:
            throw std::runtime_error("group_norm_silu_pad3d: only float16 and bfloat16 are supported");
    }
}
