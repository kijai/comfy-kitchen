# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry


def modulated_rmsnorm(
    x: Tensor,
    scale: Tensor,
    shift: Tensor,
    gamma: Tensor,
    eps: float = 1e-6,
    plus_one_scale: bool = True,
) -> Tensor:
    """Fused modulated RMSNorm: (rms_norm(x) * gamma) * (1 + scale) + shift.

    Mirrors the single-stream DiT (Krea2) norm: RMSNorm reduced over the last dim
    with a per-channel weight ``gamma``, then per-sample modulation. If
    ``plus_one_scale`` is False the scale is applied directly (``* scale``).
    """
    n = functional.rms_norm(x.float(), x.shape[-1:], weight=gamma.float(), eps=eps).to(x.dtype)
    s = (1 + scale) if plus_one_scale else scale
    return n * s + shift


# =============================================================================
# torch.library Custom Op Definition
# =============================================================================


@torch.library.custom_op("comfy_kitchen::modulated_rmsnorm", mutates_args=())
def _op_modulated_rmsnorm(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    plus_one_scale: bool,
) -> torch.Tensor:
    kwargs = {
        "x": x,
        "scale": scale,
        "shift": shift,
        "gamma": gamma,
        "eps": eps,
        "plus_one_scale": plus_one_scale,
    }
    impl = registry.get_implementation("modulated_rmsnorm", kwargs=kwargs)
    return impl(**kwargs)


@_op_modulated_rmsnorm.register_fake
def _op_modulated_rmsnorm_fake(x, scale, shift, gamma, eps, plus_one_scale):
    return torch.empty_like(x)
