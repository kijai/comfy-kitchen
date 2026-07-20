/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused RMSNorm + RoPE for Q/K or a single tensor. Logical B/H/S/D sizes
 * and strides support both BHND and BNHD tensor layouts.
 *
 * The head_dim=64 and 128 hot paths use one warp per head and stage frequency
 * and scale values once per CTA. A generic fused kernel handles other positive
 * multiples of 32. Compile-time specializations remove unused K work and select
 * interleaved or split-half rotation without runtime branches.
 */

#include "dtype_dispatch.cuh"
#include "float_utils.cuh"
#include "utils.cuh"

#include <stdexcept>

namespace comfy {
namespace {

template <typename T> struct alignas(4) Vec2 {
  T value[2];
};

template <typename T> struct alignas(8) Vec4 {
  T value[4];
};

template <int HeadDim, typename InputType, typename FreqsType,
          typename ScaleType, bool HasK, bool SplitHalf>
__global__ __launch_bounds__(
    (HeadDim == 64 ? 16 : 8) *
    kThreadsPerWarp) void rms_rope_kernel(const InputType *__restrict__ q,
                                          const InputType *__restrict__ k,
                                          const FreqsType *__restrict__ freqs,
                                          const ScaleType *__restrict__ q_scale,
                                          const ScaleType *__restrict__ k_scale,
                                          InputType *__restrict__ q_out,
                                          InputType *__restrict__ k_out,
                                          int heads, int seq_len,
                                          int64_t stride_q_batch,
                                          int64_t stride_q_head,
                                          int64_t stride_q_seq,
                                          int64_t stride_k_batch,
                                          int64_t stride_k_head,
                                          int64_t stride_k_seq,
                                          int64_t stride_out_batch,
                                          int64_t stride_out_head,
                                          int64_t stride_out_seq,
                                          int64_t stride_freqs_batch,
                                          int64_t stride_freqs_seq,
                                          int freqs_batch, int freqs_seq_len,
                                          float epsilon) {

  static_assert(HeadDim == 64 || HeadDim == 128);
  constexpr int kHeadsPerCta = HeadDim == 64 ? 16 : 8;
  constexpr int kPairs = HeadDim / 2;
  constexpr int kValuesPerLane = HeadDim / kThreadsPerWarp;
  constexpr int kPairsPerLane = kValuesPerLane / 2;

  __shared__ float shared_freqs[kPairs][4];
  __shared__ float shared_q_scale[HeadDim];
  __shared__ float shared_k_scale[HasK ? HeadDim : 1];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int token = blockIdx.x;
  const int batch_idx = token / seq_len;
  const int seq_idx = token - batch_idx * seq_len;
  const int head_idx = blockIdx.y * kHeadsPerCta + warp;

  if (tid < HeadDim) {
    shared_q_scale[tid] = static_cast<float>(q_scale[tid]);
    if constexpr (HasK) {
      shared_k_scale[tid] = static_cast<float>(k_scale[tid]);
    }
  }

  if (tid < kPairs * 4) {
    const int freq_batch_idx = freqs_batch == 1 ? 0 : batch_idx;
    const int freq_seq_idx = freqs_seq_len == 1 ? 0 : seq_idx;
    const int64_t freq_base =
        static_cast<int64_t>(freq_batch_idx) * stride_freqs_batch +
        static_cast<int64_t>(freq_seq_idx) * stride_freqs_seq;
    shared_freqs[tid >> 2][tid & 3] =
        static_cast<float>(freqs[freq_base + tid]);
  }
  __syncthreads();

  if (head_idx >= heads) {
    return;
  }

  const int64_t q_row_offset =
      static_cast<int64_t>(batch_idx) * stride_q_batch +
      static_cast<int64_t>(head_idx) * stride_q_head +
      static_cast<int64_t>(seq_idx) * stride_q_seq;
  int64_t k_row_offset = 0;
  if constexpr (HasK) {
    k_row_offset = static_cast<int64_t>(batch_idx) * stride_k_batch +
                   static_cast<int64_t>(head_idx) * stride_k_head +
                   static_cast<int64_t>(seq_idx) * stride_k_seq;
  }
  const int64_t out_row_offset =
      static_cast<int64_t>(batch_idx) * stride_out_batch +
      static_cast<int64_t>(head_idx) * stride_out_head +
      static_cast<int64_t>(seq_idx) * stride_out_seq;

  const int elem = lane * kValuesPerLane;
  float qv[kValuesPerLane];
  float kv[kValuesPerLane];
  const InputType *q_values;
  const InputType *k_values = nullptr;
  if constexpr (HeadDim == 128) {
    q_values = load_f16x4(q + q_row_offset + elem);
    if constexpr (HasK) {
      k_values = load_f16x4(k + k_row_offset + elem);
    }
  } else {
    q_values = load_f16x2(q + q_row_offset + elem);
    if constexpr (HasK) {
      k_values = load_f16x2(k + k_row_offset + elem);
    }
  }
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    qv[i] = static_cast<float>(q_values[i]);
    if constexpr (HasK) {
      kv[i] = static_cast<float>(k_values[i]);
    }
  }

  float q_sum = 0.0f;
  float k_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    q_sum = fmaf(qv[i], qv[i], q_sum);
    if constexpr (HasK) {
      k_sum = fmaf(kv[i], kv[i], k_sum);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    q_sum += __shfl_down_sync(0xffffffffu, q_sum, offset);
    if constexpr (HasK) {
      k_sum += __shfl_down_sync(0xffffffffu, k_sum, offset);
    }
  }
  q_sum = __shfl_sync(0xffffffffu, q_sum, 0);
  if constexpr (HasK) {
    k_sum = __shfl_sync(0xffffffffu, k_sum, 0);
  }

  const float q_rrms = rsqrtf(q_sum * (1.0f / HeadDim) + epsilon);
  float k_rrms = 0.0f;
  if constexpr (HasK) {
    k_rrms = rsqrtf(k_sum * (1.0f / HeadDim) + epsilon);
  }
  if constexpr (SplitHalf) {
    const int source_lane = lane >> 1;
    const int second_half_source_lane = source_lane + 16;
    const bool use_upper_pair = (lane & 1) != 0;

    const float q_lo0 = __shfl_sync(0xffffffffu, qv[0], source_lane);
    const float q_lo1 = __shfl_sync(0xffffffffu, qv[1], source_lane);
    const float q_hi0 =
        __shfl_sync(0xffffffffu, qv[0], second_half_source_lane);
    const float q_hi1 =
        __shfl_sync(0xffffffffu, qv[1], second_half_source_lane);
    if constexpr (HeadDim == 64) {
      qv[0] = use_upper_pair ? q_lo1 : q_lo0;
      qv[1] = use_upper_pair ? q_hi1 : q_hi0;
    } else {
      const float q_lo2 = __shfl_sync(0xffffffffu, qv[2], source_lane);
      const float q_lo3 = __shfl_sync(0xffffffffu, qv[3], source_lane);
      const float q_hi2 =
          __shfl_sync(0xffffffffu, qv[2], second_half_source_lane);
      const float q_hi3 =
          __shfl_sync(0xffffffffu, qv[3], second_half_source_lane);
      qv[0] = use_upper_pair ? q_lo2 : q_lo0;
      qv[1] = use_upper_pair ? q_hi2 : q_hi0;
      qv[2] = use_upper_pair ? q_lo3 : q_lo1;
      qv[3] = use_upper_pair ? q_hi3 : q_hi1;
    }

    if constexpr (HasK) {
      const float k_lo0 = __shfl_sync(0xffffffffu, kv[0], source_lane);
      const float k_lo1 = __shfl_sync(0xffffffffu, kv[1], source_lane);
      const float k_hi0 =
          __shfl_sync(0xffffffffu, kv[0], second_half_source_lane);
      const float k_hi1 =
          __shfl_sync(0xffffffffu, kv[1], second_half_source_lane);
      if constexpr (HeadDim == 64) {
        kv[0] = use_upper_pair ? k_lo1 : k_lo0;
        kv[1] = use_upper_pair ? k_hi1 : k_hi0;
      } else {
        const float k_lo2 = __shfl_sync(0xffffffffu, kv[2], source_lane);
        const float k_lo3 = __shfl_sync(0xffffffffu, kv[3], source_lane);
        const float k_hi2 =
            __shfl_sync(0xffffffffu, kv[2], second_half_source_lane);
        const float k_hi3 =
            __shfl_sync(0xffffffffu, kv[3], second_half_source_lane);
        kv[0] = use_upper_pair ? k_lo2 : k_lo0;
        kv[1] = use_upper_pair ? k_hi2 : k_hi0;
        kv[2] = use_upper_pair ? k_lo3 : k_lo1;
        kv[3] = use_upper_pair ? k_hi3 : k_hi1;
      }
    }
  }

  // Match the unfused contract: RMSNorm rounds to the input dtype before
  // RoPE converts back to fp32 for the 2x2 rotation.
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    const int local_pair = i >> 1;
    const int component = i & 1;
    const int pair = lane * kPairsPerLane + local_pair;
    const int scale_idx = SplitHalf ? pair + component * kPairs : elem + i;
    qv[i] = static_cast<float>(
        static_cast<InputType>(qv[i] * q_rrms * shared_q_scale[scale_idx]));
    if constexpr (HasK) {
      kv[i] = static_cast<float>(
          static_cast<InputType>(kv[i] * k_rrms * shared_k_scale[scale_idx]));
    }
  }

  InputType q_result[kValuesPerLane];
  InputType k_result[kValuesPerLane];
#pragma unroll
  for (int local_pair = 0; local_pair < kPairsPerLane; ++local_pair) {
    const int pair = lane * kPairsPerLane + local_pair;
    const int i = local_pair * 2;
    const float *f = shared_freqs[pair];
    q_result[i] = static_cast<InputType>(fmaf(f[1], qv[i + 1], f[0] * qv[i]));
    q_result[i + 1] =
        static_cast<InputType>(fmaf(f[3], qv[i + 1], f[2] * qv[i]));
    if constexpr (HasK) {
      k_result[i] = static_cast<InputType>(fmaf(f[1], kv[i + 1], f[0] * kv[i]));
      k_result[i + 1] =
          static_cast<InputType>(fmaf(f[3], kv[i + 1], f[2] * kv[i]));
    }
  }

  if constexpr (SplitHalf) {
    const int pair = lane * kPairsPerLane;
    if constexpr (HeadDim == 128) {
      Vec2<InputType> q_lo;
      Vec2<InputType> q_hi;
      q_lo.value[0] = q_result[0];
      q_lo.value[1] = q_result[2];
      q_hi.value[0] = q_result[1];
      q_hi.value[1] = q_result[3];
      *reinterpret_cast<Vec2<InputType> *>(q_out + out_row_offset + pair) =
          q_lo;
      *reinterpret_cast<Vec2<InputType> *>(q_out + out_row_offset + pair +
                                           kPairs) = q_hi;
      if constexpr (HasK) {
        Vec2<InputType> k_lo;
        Vec2<InputType> k_hi;
        k_lo.value[0] = k_result[0];
        k_lo.value[1] = k_result[2];
        k_hi.value[0] = k_result[1];
        k_hi.value[1] = k_result[3];
        *reinterpret_cast<Vec2<InputType> *>(k_out + out_row_offset + pair) =
            k_lo;
        *reinterpret_cast<Vec2<InputType> *>(k_out + out_row_offset + pair +
                                             kPairs) = k_hi;
      }
    } else {
      q_out[out_row_offset + pair] = q_result[0];
      q_out[out_row_offset + pair + kPairs] = q_result[1];
      if constexpr (HasK) {
        k_out[out_row_offset + pair] = k_result[0];
        k_out[out_row_offset + pair + kPairs] = k_result[1];
      }
    }
  } else if constexpr (HeadDim == 128) {
    Vec4<InputType> q_vec;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      q_vec.value[i] = q_result[i];
    }
    *reinterpret_cast<Vec4<InputType> *>(q_out + out_row_offset + elem) = q_vec;
    if constexpr (HasK) {
      Vec4<InputType> k_vec;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        k_vec.value[i] = k_result[i];
      }
      *reinterpret_cast<Vec4<InputType> *>(k_out + out_row_offset + elem) =
          k_vec;
    }
  } else {
    Vec2<InputType> q_vec;
#pragma unroll
    for (int i = 0; i < kValuesPerLane; ++i) {
      q_vec.value[i] = q_result[i];
    }
    *reinterpret_cast<Vec2<InputType> *>(q_out + out_row_offset + elem) = q_vec;
    if constexpr (HasK) {
      Vec2<InputType> k_vec;
#pragma unroll
      for (int i = 0; i < kValuesPerLane; ++i) {
        k_vec.value[i] = k_result[i];
      }
      *reinterpret_cast<Vec2<InputType> *>(k_out + out_row_offset + elem) =
          k_vec;
    }
  }
}

template <typename InputType, typename FreqsType, typename ScaleType, bool HasK,
          bool SplitHalf>
__global__ __launch_bounds__(8 * kThreadsPerWarp) void rms_rope_generic_kernel(
    const InputType *__restrict__ q, const InputType *__restrict__ k,
    const FreqsType *__restrict__ freqs, const ScaleType *__restrict__ q_scale,
    const ScaleType *__restrict__ k_scale, InputType *__restrict__ q_out,
    InputType *__restrict__ k_out, int heads, int seq_len, int head_dim,
    int64_t stride_q_batch, int64_t stride_q_head, int64_t stride_q_seq,
    int64_t stride_k_batch, int64_t stride_k_head, int64_t stride_k_seq,
    int64_t stride_out_batch, int64_t stride_out_head, int64_t stride_out_seq,
    int64_t stride_freqs_batch, int64_t stride_freqs_seq, int freqs_batch,
    int freqs_seq_len, float epsilon) {
  constexpr int kHeadsPerCta = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x;
  const int batch_idx = token / seq_len;
  const int seq_idx = token - batch_idx * seq_len;
  const int head_idx = blockIdx.y * kHeadsPerCta + warp;
  if (head_idx >= heads) {
    return;
  }

  const int64_t q_row_offset =
      static_cast<int64_t>(batch_idx) * stride_q_batch +
      static_cast<int64_t>(head_idx) * stride_q_head +
      static_cast<int64_t>(seq_idx) * stride_q_seq;
  int64_t k_row_offset = 0;
  if constexpr (HasK) {
    k_row_offset = static_cast<int64_t>(batch_idx) * stride_k_batch +
                   static_cast<int64_t>(head_idx) * stride_k_head +
                   static_cast<int64_t>(seq_idx) * stride_k_seq;
  }
  const int64_t out_row_offset =
      static_cast<int64_t>(batch_idx) * stride_out_batch +
      static_cast<int64_t>(head_idx) * stride_out_head +
      static_cast<int64_t>(seq_idx) * stride_out_seq;

  const int values_per_lane = head_dim / kThreadsPerWarp;
  const int elem = lane * values_per_lane;
  float q_sum = 0.0f;
  float k_sum = 0.0f;
#pragma unroll 1
  for (int i = 0; i < values_per_lane; ++i) {
    const float qv = static_cast<float>(q[q_row_offset + elem + i]);
    q_sum = fmaf(qv, qv, q_sum);
    if constexpr (HasK) {
      const float kv = static_cast<float>(k[k_row_offset + elem + i]);
      k_sum = fmaf(kv, kv, k_sum);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    q_sum += __shfl_down_sync(0xffffffffu, q_sum, offset);
    if constexpr (HasK) {
      k_sum += __shfl_down_sync(0xffffffffu, k_sum, offset);
    }
  }
  q_sum = __shfl_sync(0xffffffffu, q_sum, 0);
  if constexpr (HasK) {
    k_sum = __shfl_sync(0xffffffffu, k_sum, 0);
  }

  const float inv_head_dim = 1.0f / static_cast<float>(head_dim);
  const float q_rrms = rsqrtf(q_sum * inv_head_dim + epsilon);
  float k_rrms = 0.0f;
  if constexpr (HasK) {
    k_rrms = rsqrtf(k_sum * inv_head_dim + epsilon);
  }

  const int freq_batch_idx = freqs_batch == 1 ? 0 : batch_idx;
  const int freq_seq_idx = freqs_seq_len == 1 ? 0 : seq_idx;
  const int64_t freq_base =
      static_cast<int64_t>(freq_batch_idx) * stride_freqs_batch +
      static_cast<int64_t>(freq_seq_idx) * stride_freqs_seq;
  const int pairs = head_dim / 2;
  for (int pair = lane; pair < pairs; pair += kThreadsPerWarp) {
    const int first = SplitHalf ? pair : pair * 2;
    const int second = SplitHalf ? pair + pairs : first + 1;
    float q0 = static_cast<float>(q[q_row_offset + first]);
    float q1 = static_cast<float>(q[q_row_offset + second]);
    q0 = static_cast<float>(static_cast<InputType>(
        q0 * q_rrms * static_cast<float>(q_scale[first])));
    q1 = static_cast<float>(static_cast<InputType>(
        q1 * q_rrms * static_cast<float>(q_scale[second])));

    const FreqsType *f = freqs + freq_base + pair * 4;
    const float f0 = static_cast<float>(f[0]);
    const float f1 = static_cast<float>(f[1]);
    const float f2 = static_cast<float>(f[2]);
    const float f3 = static_cast<float>(f[3]);
    q_out[out_row_offset + first] =
        static_cast<InputType>(fmaf(f1, q1, f0 * q0));
    q_out[out_row_offset + second] =
        static_cast<InputType>(fmaf(f3, q1, f2 * q0));

    if constexpr (HasK) {
      float k0 = static_cast<float>(k[k_row_offset + first]);
      float k1 = static_cast<float>(k[k_row_offset + second]);
      k0 = static_cast<float>(static_cast<InputType>(
          k0 * k_rrms * static_cast<float>(k_scale[first])));
      k1 = static_cast<float>(static_cast<InputType>(
          k1 * k_rrms * static_cast<float>(k_scale[second])));
      k_out[out_row_offset + first] =
          static_cast<InputType>(fmaf(f1, k1, f0 * k0));
      k_out[out_row_offset + second] =
          static_cast<InputType>(fmaf(f3, k1, f2 * k0));
    }
  }
}

template <typename InputType, typename FreqsType, typename ScaleType, bool HasK,
          bool SplitHalf>
void rms_rope_generic_launch_config(
    const InputType *q, const InputType *k, const FreqsType *freqs,
    const ScaleType *q_scale, const ScaleType *k_scale, InputType *q_out,
    InputType *k_out, int batch, int heads, int seq_len, int head_dim,
    int64_t stride_q_batch, int64_t stride_q_head, int64_t stride_q_seq,
    int64_t stride_k_batch, int64_t stride_k_head, int64_t stride_k_seq,
    int64_t stride_out_batch, int64_t stride_out_head, int64_t stride_out_seq,
    int64_t stride_freqs_batch, int64_t stride_freqs_seq, int freqs_batch,
    int freqs_seq_len, float epsilon, cudaStream_t stream) {
  constexpr int kHeadsPerCta = 8;
  constexpr int kThreads = kHeadsPerCta * kThreadsPerWarp;
  const dim3 grid(batch * seq_len, (heads + kHeadsPerCta - 1) / kHeadsPerCta);
  rms_rope_generic_kernel<InputType, FreqsType, ScaleType, HasK, SplitHalf>
      <<<grid, kThreads, 0, stream>>>(
          q, k, freqs, q_scale, k_scale, q_out, k_out, heads, seq_len, head_dim,
          stride_q_batch, stride_q_head, stride_q_seq, stride_k_batch,
          stride_k_head, stride_k_seq, stride_out_batch, stride_out_head,
          stride_out_seq, stride_freqs_batch, stride_freqs_seq, freqs_batch,
          freqs_seq_len, epsilon);
}

template <int HeadDim, typename InputType, typename FreqsType,
          typename ScaleType, bool HasK, bool SplitHalf>
void rms_rope_launch_config(
    const InputType *q, const InputType *k, const FreqsType *freqs,
    const ScaleType *q_scale, const ScaleType *k_scale, InputType *q_out,
    InputType *k_out, int batch, int heads, int seq_len, int64_t stride_q_batch,
    int64_t stride_q_head, int64_t stride_q_seq, int64_t stride_k_batch,
    int64_t stride_k_head, int64_t stride_k_seq, int64_t stride_out_batch,
    int64_t stride_out_head, int64_t stride_out_seq, int64_t stride_freqs_batch,
    int64_t stride_freqs_seq, int freqs_batch, int freqs_seq_len, float epsilon,
    cudaStream_t stream) {

  constexpr int kHeadsPerCta = HeadDim == 64 ? 16 : 8;
  constexpr int kThreads = kHeadsPerCta * kThreadsPerWarp;
  const dim3 grid(batch * seq_len, (heads + kHeadsPerCta - 1) / kHeadsPerCta);
  rms_rope_kernel<HeadDim, InputType, FreqsType, ScaleType, HasK, SplitHalf>
      <<<grid, kThreads, 0, stream>>>(
          q, k, freqs, q_scale, k_scale, q_out, k_out, heads, seq_len,
          stride_q_batch, stride_q_head, stride_q_seq, stride_k_batch,
          stride_k_head, stride_k_seq, stride_out_batch, stride_out_head,
          stride_out_seq, stride_freqs_batch, stride_freqs_seq, freqs_batch,
          freqs_seq_len, epsilon);
}

template <typename InputType, typename FreqsType, typename ScaleType>
void rms_rope_launcher(const InputType *q, const InputType *k,
                       const FreqsType *freqs, const ScaleType *q_scale,
                       const ScaleType *k_scale, InputType *q_out,
                       InputType *k_out, int batch, int heads, int seq_len,
                       int head_dim, int64_t stride_q_batch,
                       int64_t stride_q_head, int64_t stride_q_seq,
                       int64_t stride_k_batch, int64_t stride_k_head,
                       int64_t stride_k_seq, int64_t stride_out_batch,
                       int64_t stride_out_head, int64_t stride_out_seq,
                       int64_t stride_freqs_batch, int64_t stride_freqs_seq,
                       int freqs_batch, int freqs_seq_len, float epsilon,
                       bool has_k, bool split_half, cudaStream_t stream) {

  if (batch == 0 || heads == 0 || seq_len == 0) {
    return;
  }

#define LAUNCH_CONFIG(HEAD_DIM, HAS_K, SPLIT_HALF)                             \
  rms_rope_launch_config<HEAD_DIM, InputType, FreqsType, ScaleType, HAS_K,     \
                         SPLIT_HALF>(                                          \
      q, k, freqs, q_scale, k_scale, q_out, k_out, batch, heads, seq_len,      \
      stride_q_batch, stride_q_head, stride_q_seq, stride_k_batch,             \
      stride_k_head, stride_k_seq, stride_out_batch, stride_out_head,          \
      stride_out_seq, stride_freqs_batch, stride_freqs_seq, freqs_batch,       \
      freqs_seq_len, epsilon, stream)

#define LAUNCH_GENERIC(HAS_K, SPLIT_HALF)                                      \
  rms_rope_generic_launch_config<InputType, FreqsType, ScaleType, HAS_K,       \
                                 SPLIT_HALF>(                                  \
      q, k, freqs, q_scale, k_scale, q_out, k_out, batch, heads, seq_len,      \
      head_dim, stride_q_batch, stride_q_head, stride_q_seq, stride_k_batch,   \
      stride_k_head, stride_k_seq, stride_out_batch, stride_out_head,          \
      stride_out_seq, stride_freqs_batch, stride_freqs_seq, freqs_batch,       \
      freqs_seq_len, epsilon, stream)

#define LAUNCH_HEAD_DIM(HAS_K, SPLIT_HALF)                                     \
  if (head_dim == 64) {                                                        \
    LAUNCH_CONFIG(64, HAS_K, SPLIT_HALF);                                      \
  } else if (head_dim == 128) {                                                \
    LAUNCH_CONFIG(128, HAS_K, SPLIT_HALF);                                     \
  } else {                                                                     \
    LAUNCH_GENERIC(HAS_K, SPLIT_HALF);                                         \
  }

  if (has_k) {
    if (split_half) {
      LAUNCH_HEAD_DIM(true, true);
    } else {
      LAUNCH_HEAD_DIM(true, false);
    }
  } else if (split_half) {
    LAUNCH_HEAD_DIM(false, true);
  } else {
    LAUNCH_HEAD_DIM(false, false);
  }
#undef LAUNCH_HEAD_DIM
#undef LAUNCH_GENERIC
#undef LAUNCH_CONFIG

  CUDA_CHECK(cudaGetLastError());
}

} // namespace
} // namespace comfy

extern "C" void launch_rms_rope_kernel(
    const void *q, const void *k, const void *freqs, const void *q_scale,
    const void *k_scale, void *q_out, void *k_out, int batch, int heads,
    int seq_len, int head_dim, int64_t stride_q_batch, int64_t stride_q_head,
    int64_t stride_q_seq, int64_t stride_k_batch, int64_t stride_k_head,
    int64_t stride_k_seq, int64_t stride_out_batch, int64_t stride_out_head,
    int64_t stride_out_seq, int64_t stride_freqs_batch,
    int64_t stride_freqs_seq, int freqs_batch, int freqs_seq_len, float epsilon,
    int input_dtype_code, int freqs_dtype_code, int scale_dtype_code,
    bool has_k, bool split_half, cudaStream_t stream) {

  DISPATCH_HALF_DTYPE(input_dtype_code, InputType, [&] {
    DISPATCH_FP_DTYPE(freqs_dtype_code, FreqsType, [&] {
      DISPATCH_FP_DTYPE(scale_dtype_code, ScaleType, [&] {
        comfy::rms_rope_launcher<InputType, FreqsType, ScaleType>(
            static_cast<const InputType *>(q),
            static_cast<const InputType *>(k),
            static_cast<const FreqsType *>(freqs),
            static_cast<const ScaleType *>(q_scale),
            static_cast<const ScaleType *>(k_scale),
            static_cast<InputType *>(q_out), static_cast<InputType *>(k_out),
            batch, heads, seq_len, head_dim, stride_q_batch, stride_q_head,
            stride_q_seq, stride_k_batch, stride_k_head, stride_k_seq,
            stride_out_batch, stride_out_head, stride_out_seq,
            stride_freqs_batch, stride_freqs_seq, freqs_batch, freqs_seq_len,
            epsilon, has_k, split_half, stream);
      });
    });
  });
}
