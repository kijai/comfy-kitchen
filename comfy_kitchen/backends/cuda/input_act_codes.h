// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Codes for the optional activation a quantizer absorbs on the way in. Shared
// by the CUDA kernels and the nanobind layer; must match INPUT_ACT_TO_CODE in
// comfy_kitchen/backends/_activations.py.
#pragma once

namespace comfy {

// SwiGLU is the gated pair: the raw row is [gate | up] (2*K wide) and the
// activated row silu(gate) * up is K wide. RmsNorm is the row-wise
// normalization x * rsqrt(mean(x^2) + eps) * weight and needs the weight
// pointer and eps carried alongside the code. NanToNum matches
// torch.nan_to_num defaults (nan -> 0, +/-inf -> dtype finite max/min);
// the others are elementwise.
enum : int { kActNone = 0, kActGeluTanh = 1, kActSwiGLU = 2, kActRmsNorm = 3, kActNanToNum = 4 };

}  // namespace comfy
