/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Fused modulated RMSNorm:
//     out = (rmsnorm(x) * gamma) * (1 + scale) + shift      [plus_one_scale]
//     out = (rmsnorm(x) * gamma) *      scale  + shift      [otherwise]
// where rmsnorm(x) = x * rsqrt(mean(x^2) + eps), reduced over the last dim D.
//
// This is the AdaLN-single norm of single-stream DiTs (Krea2 etc.): the model
// does F.rms_norm(x.float(), weight=gamma).to(bf16) and THEN modulates per
// block. Folding it into one kernel reads x once (bf16) and accumulates the
// sum-of-squares in fp32 registers, avoiding the fp32 materialization + several
// bf16 HBM round-trips of the eager path. It is numerically equivalent to the
// reference within one bf16 ULP (the reduction order differs), not bit-exact.
//
// gamma is a single [D] vector shared by every row. scale/shift are broadcast
// per-sample exactly like ops/adaln.cu: the caller passes them in distinct-row
// form with `scale_group`/`shift_group` = the number of consecutive output rows
// that share one vector (group == 1 => a distinct vector per row). All global
// accesses are 128-bit vectorized when D * sizeof(T) % 16 == 0.

#include "utils.cuh"
#include "dtype_dispatch.cuh"

#include <limits>
#include <stdexcept>

namespace comfy {

namespace {

constexpr int kMRMSThreads = 256;
constexpr int kMRMSWarps = kMRMSThreads / kThreadsPerWarp;  // 8
constexpr int kMRMSRowsPerWarpBlock = kMRMSWarps;

template<typename T>
__device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float<float>(float val) { return val; }
template<> __device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ half from_float<half>(float val) { return __float2half_rn(val); }
template<> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float val) { return __float2bfloat16_rn(val); }

template<typename T> struct VecWidth { static constexpr int value = 16 / sizeof(T); };

template<typename T>
struct alignas(16) Vec {
    static constexpr int W = VecWidth<T>::value;
    T elts[W];
};

__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int offset = kThreadsPerWarp / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v, float* warp_smem) {
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int wid  = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_smem[wid] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < kMRMSWarps; ++w) total += warp_smem[w];
    return total;
}

__device__ __forceinline__ int modulation_row(int row, int group, int n_rows) {
    if (group == 1) return row;
    if (group == n_rows) return 0;
    if ((group & (group - 1)) == 0) return row >> (__ffs(group) - 1);
    return row / group;
}

// Match the reference's per-op bf16 rounding: t1 = (1+scale); t2 = t1 * n;
// out = t2 + shift, each rounded to T. `normed` is already T-rounded. The final
// (+ shift) rounding happens at the store (from_float<T> by the caller).
template<typename T>
__device__ __forceinline__ float modulate(float normed, float s, float sh, bool plus_one) {
    const float s1 = plus_one ? to_float(from_float<T>(1.0f + s)) : s;
    const float prod = to_float(from_float<T>(s1 * normed));
    return prod + sh;
}

// One block per row; 256 threads. For larger N (this kernel).
template<typename T>
__global__ void modulated_rmsnorm_kernel(
    const T* __restrict__ x,
    const float* __restrict__ gamma,   // RMSNorm weight kept in fp32 (matches model)
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int D,
    int scale_group,
    int shift_group,
    float eps,
    int plus_one)
{
    constexpr int VEC = VecWidth<T>::value;
    const int row = static_cast<int>(blockIdx.x);
    const int n_rows = static_cast<int>(gridDim.x);
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    __shared__ float warp_smem[kMRMSWarps];

    const int scale_row = modulation_row(row, scale_group, n_rows);
    const int shift_row = (shift_group == scale_group)
        ? scale_row
        : modulation_row(row, shift_group, n_rows);

    const T* x_row  = x     + row * D;
    const T* s_row  = scale + scale_row * D;
    const T* sh_row = shift + shift_row * D;
    T*       o_row  = out   + row * D;

    const int n_vec   = (D % VEC == 0) ? (D / VEC) : 0;
    const int vec_end = n_vec * VEC;
    const Vec<T>* x_vec = reinterpret_cast<const Vec<T>*>(x_row);

    // ---- Pass 1: sum of squares ----
    float ss = 0.0f;
    for (int v = tid; v < n_vec; v += nthreads) {
        Vec<T> xv = x_vec[v];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float f = to_float(xv.elts[j]);
            ss += f * f;
        }
    }
    for (int i = vec_end + tid; i < D; i += nthreads) {
        float f = to_float(x_row[i]);
        ss += f * f;
    }

    ss = block_reduce_sum(ss, warp_smem);
    const float rstd = rsqrtf(ss / static_cast<float>(D) + eps);
    const bool po = plus_one != 0;

    // ---- Pass 2: rmsnorm * gamma, then modulate ----
    const Vec<T>* s_vec  = reinterpret_cast<const Vec<T>*>(s_row);
    const Vec<T>* sh_vec = reinterpret_cast<const Vec<T>*>(sh_row);
    Vec<T>*       o_vec  = reinterpret_cast<Vec<T>*>(o_row);
    for (int v = tid; v < n_vec; v += nthreads) {
        Vec<T> xv  = x_vec[v];
        Vec<T> sv  = s_vec[v];
        Vec<T> shv = sh_vec[v];
        Vec<T> ov;
        const int base = v * VEC;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            // Round the rmsnorm output to T before modulating, matching the
            // reference (F.rms_norm(...).to(dtype) then modulate). gamma is fp32.
            float normed = to_float(from_float<T>(to_float(xv.elts[j]) * rstd * gamma[base + j]));
            ov.elts[j] = from_float<T>(modulate<T>(normed, to_float(sv.elts[j]), to_float(shv.elts[j]), po));
        }
        o_vec[v] = ov;
    }
    for (int i = vec_end + tid; i < D; i += nthreads) {
        float normed = to_float(from_float<T>(to_float(x_row[i]) * rstd * gamma[i]));
        o_row[i] = from_float<T>(modulate<T>(normed, to_float(s_row[i]), to_float(sh_row[i]), po));
    }
}

// One warp per row; for small N (<= 1024 rows).
template<typename T>
__global__ void modulated_rmsnorm_warp_kernel(
    const T* __restrict__ x,
    const float* __restrict__ gamma,
    const T* __restrict__ scale,
    const T* __restrict__ shift,
    T* __restrict__ out,
    int N,
    int D,
    int scale_group,
    int shift_group,
    float eps,
    int plus_one)
{
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & (kThreadsPerWarp - 1);
    const int row = blockIdx.x * kMRMSRowsPerWarpBlock + warp_in_block;
    if (row >= N) return;

    const int scale_row = modulation_row(row, scale_group, N);
    const int shift_row = (shift_group == scale_group)
        ? scale_row
        : modulation_row(row, shift_group, N);

    const T* x_row  = x     + row * D;
    const T* s_row  = scale + scale_row * D;
    const T* sh_row = shift + shift_row * D;
    T*       o_row  = out   + row * D;

    float ss = 0.0f;
    for (int i = lane; i < D; i += kThreadsPerWarp) {
        float f = to_float(x_row[i]);
        ss += f * f;
    }
    ss = warp_reduce_sum(ss);
    ss = __shfl_sync(0xffffffff, ss, 0);
    const float rstd = rsqrtf(ss / static_cast<float>(D) + eps);
    const bool po = plus_one != 0;

    for (int i = lane; i < D; i += kThreadsPerWarp) {
        float normed = to_float(from_float<T>(to_float(x_row[i]) * rstd * gamma[i]));
        o_row[i] = from_float<T>(modulate<T>(normed, to_float(s_row[i]), to_float(sh_row[i]), po));
    }
}

}  // namespace

}  // namespace comfy


extern "C" {

void launch_modulated_rmsnorm_kernel(
    const void* x,
    const void* gamma,
    const void* scale,
    const void* shift,
    void*       out,
    int64_t     N,
    int64_t     D,
    int64_t     scale_group,
    int64_t     shift_group,
    float       eps,
    int         plus_one,
    int         dtype_code,
    cudaStream_t stream)
{
    if (N > std::numeric_limits<int>::max() ||
        D > std::numeric_limits<int>::max() ||
        scale_group > std::numeric_limits<int>::max() ||
        shift_group > std::numeric_limits<int>::max()) {
        throw std::runtime_error("modulated_rmsnorm dimensions exceed CUDA kernel int32 indexing limits");
    }

    dim3 block(comfy::kMRMSThreads);

    DISPATCH_FP_DTYPE(dtype_code, T, [&]() {
        if (N <= 1024) {
            dim3 warp_grid(
                static_cast<unsigned int>((N + comfy::kMRMSRowsPerWarpBlock - 1) /
                                          comfy::kMRMSRowsPerWarpBlock));
            comfy::modulated_rmsnorm_warp_kernel<T><<<warp_grid, block, 0, stream>>>(
                static_cast<const T*>(x),
                static_cast<const float*>(gamma),
                static_cast<const T*>(scale),
                static_cast<const T*>(shift),
                static_cast<T*>(out),
                static_cast<int>(N),
                static_cast<int>(D),
                static_cast<int>(scale_group),
                static_cast<int>(shift_group),
                eps,
                plus_one);
        } else {
            dim3 grid(static_cast<unsigned int>(N));
            comfy::modulated_rmsnorm_kernel<T><<<grid, block, 0, stream>>>(
                static_cast<const T*>(x),
                static_cast<const float*>(gamma),
                static_cast<const T*>(scale),
                static_cast<const T*>(shift),
                static_cast<T*>(out),
                static_cast<int>(D),
                static_cast<int>(scale_group),
                static_cast<int>(shift_group),
                eps,
                plus_one);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("modulated_rmsnorm kernel launch failed: ") + cudaGetErrorString(err));
    });
}

}  // extern "C"
