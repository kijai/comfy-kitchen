// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. Derived from SageAttention
// (https://github.com/thu-ml/SageAttention) commit
// d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5. DLPack-compatible launcher for the
// SM89 INT8-QK / FP8-V attention kernel. Fixed configuration: per-thread QK
// quantization, FP32 accumulation, unfused V (fv=0), no LSE return, no rotary
// fusion.

#include "qk_int_sv_f8_cuda_sm89.cuh"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

template <int HEAD_DIM, MaskMode mask_mode, typename DTypeOut>
void launch_impl(int8_t *q, int8_t *k, int8_t *v, DTypeOut *o, float *q_scale,
                 float *k_scale, float *v_scale, int qo_len, int kv_len,
                 int num_qo_heads, int num_kv_groups, int stride_bz_q,
                 int stride_seq_q, int stride_h_q, int stride_bz_k,
                 int stride_seq_k, int stride_h_k, int stride_bz_v,
                 int stride_h_v, int stride_d_v, int stride_bz_o,
                 int stride_seq_o, int stride_h_o, float sm_scale,
                 int batch_size, bool pv_fp16, bool qk_per_warp,
                 cudaStream_t stream) {
  // Tiling constants — must match sage_attention.py and dlpack_bindings.cpp.
  constexpr int CTA_Q = 128;
  constexpr int CTA_K = 64;
  constexpr int WARP_Q = (HEAD_DIM >= 256) ? 16 : 32;
  constexpr int WARP_K = 64;

  size_t smem_max =
      std::max(static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(int8_t) +
                                   CTA_K * HEAD_DIM * sizeof(int8_t) +
                                   CTA_K * HEAD_DIM * sizeof(int8_t)),
               static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(half)));

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  auto launch = [&](auto kernel) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_max);
    kernel<<<grid, block, smem_max, stream>>>(
        q, k, v, o, nullptr, q_scale, k_scale, v_scale, nullptr, qo_len, kv_len,
        num_kv_groups, stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,
        stride_seq_k, stride_h_k, stride_bz_v, stride_h_v, stride_d_v,
        stride_bz_o, stride_seq_o, stride_h_o, sm_scale);
  };

  // pv_fp16: accumulate the P@V matmul with fp16 (mma f8f8f16, ~2x rate on
  // sm89; SageAttention2++). Requires V quantized to the small +-2.25 e4m3 range
  // (handled in the V quant launch) so the running PV sum stays inside fp16.
  // Otherwise use plain fp32 accumulation with the full +-448 V range.
  // qk_per_warp: per-warp INT8 Q/K quant granularity (upstream's sm120 choice);
  // otherwise per-thread (sm89 default). Both Q and K use the same granularity.
#define SAGE_LAUNCH(QGRAN, INST_BUF, PV16)                                     \
  launch(qk_int_sv_f8_attn_kernel<                                            \
         CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,             \
         QuantGranularity::QGRAN, QuantGranularity::QGRAN, float, INST_BUF,   \
         DTypeOut, ComputeUnit::kCudaCore, mask_mode, false, true, false, PV16>)

  if (qk_per_warp) {
    if (pv_fp16)
      SAGE_LAUNCH(kPerWarp, true, true);
    else
      SAGE_LAUNCH(kPerWarp, false, false);
  } else {
    if (pv_fp16)
      SAGE_LAUNCH(kPerThread, true, true);
    else
      SAGE_LAUNCH(kPerThread, false, false);
  }
#undef SAGE_LAUNCH
}

} // anonymous namespace

extern "C" void launch_sage_attn_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, int batch_size, int qo_len,
    int kv_len, int num_qo_heads, int num_kv_heads, int head_dim,
    int stride_bz_q, int stride_seq_q, int stride_h_q, int stride_bz_k,
    int stride_seq_k, int stride_h_k, int stride_bz_v, int stride_h_v,
    int stride_d_v, int stride_bz_o, int stride_seq_o, int stride_h_o,
    int is_causal, float sm_scale, int output_dtype_code, int pv_fp16_accum,
    int qk_per_warp_gran, cudaStream_t stream) {
  int num_kv_groups = num_qo_heads / num_kv_heads;
  const bool pv_fp16 = pv_fp16_accum != 0;
  const bool qk_per_warp = qk_per_warp_gran != 0;

  // Upstream kernel uses non-const pointers; cast away const from the
  // extern "C" boundary (kernel does not modify inputs).
  auto q_ = const_cast<int8_t *>(static_cast<const int8_t *>(q));
  auto k_ = const_cast<int8_t *>(static_cast<const int8_t *>(k));
  auto v_ = const_cast<int8_t *>(static_cast<const int8_t *>(v));
  auto qs_ = const_cast<float *>(static_cast<const float *>(q_scale));
  auto ks_ = const_cast<float *>(static_cast<const float *>(k_scale));
  auto vs_ = const_cast<float *>(static_cast<const float *>(v_scale));

#define LAUNCH(HD, MM, DT)                                                     \
  launch_impl<HD, MM, DT>(q_, k_, v_, static_cast<DT *>(o), qs_, ks_, vs_,     \
                          qo_len, kv_len, num_qo_heads, num_kv_groups,         \
                          stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,  \
                          stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,   \
                          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o,   \
                          sm_scale, batch_size, pv_fp16, qk_per_warp, stream)

#define DISPATCH_DTYPE(HD, MM)                                                 \
  if (output_dtype_code == 1) {                                                \
    LAUNCH(HD, MM, half);                                                      \
  } else {                                                                     \
    LAUNCH(HD, MM, nv_bfloat16);                                               \
  }

#define DISPATCH_CAUSAL(HD)                                                    \
  if (is_causal) {                                                             \
    DISPATCH_DTYPE(HD, MaskMode::kCausal);                                     \
  } else {                                                                     \
    DISPATCH_DTYPE(HD, MaskMode::kNone);                                       \
  }

  if (head_dim == 64) {
    DISPATCH_CAUSAL(64);
  } else if (head_dim == 128) {
    DISPATCH_CAUSAL(128);
  } else if (head_dim == 256) {
    DISPATCH_CAUSAL(256);
  } else {
    throw std::runtime_error("sage_attn: unsupported head_dim " +
                             std::to_string(head_dim));
  }

#undef LAUNCH
#undef DISPATCH_DTYPE
#undef DISPATCH_CAUSAL
}
