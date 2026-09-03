# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nn import functional

from comfy_kitchen.registry import registry


def group_norm_silu_pad3d(
    x: Tensor,
    weight: Tensor | None,
    bias: Tensor | None,
    num_groups: int,
    eps: float,
    pad: list[int],
    silu: bool,
) -> Tensor:
    """Per-frame GroupNorm (statistics over one frame's C, H, W), optional SiLU,
    then the causal 3D conv padding: reflect (left, right, top, bottom) in
    space and `front` zero frames in time. weight=None skips the norm."""
    orig = x
    b, c, t, h, w = x.shape
    if weight is not None:
        # group_norm on CUDA rejects mixed dtypes; the affine params may be fp32
        weight = weight.to(x.dtype)
        bias = None if bias is None else bias.to(x.dtype)
        y = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        y = functional.group_norm(y, num_groups, weight, bias, eps)
        x = y.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    if silu:
        x = functional.silu(x)
    left, right, top, bottom, front = pad
    if left or right or top or bottom:
        x = functional.pad(x, (left, right, top, bottom, 0, 0), mode="reflect")
    if front:
        x = functional.pad(x, (0, 0, 0, 0, front, 0))
    # a custom op's output must not alias its input (nothing-to-do call)
    return x if x is not orig else x.clone()


@torch.library.custom_op("comfy_kitchen::group_norm_silu_pad3d", mutates_args=())
def _op_group_norm_silu_pad3d(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    num_groups: int,
    eps: float,
    pad: list[int],
    silu: bool,
) -> torch.Tensor:
    kwargs = {"x": x, "weight": weight, "bias": bias, "num_groups": num_groups,
              "eps": eps, "pad": pad, "silu": silu}
    impl = registry.get_implementation("group_norm_silu_pad3d", kwargs=kwargs)
    return impl(**kwargs)


@_op_group_norm_silu_pad3d.register_fake
def _op_group_norm_silu_pad3d_fake(x, weight, bias, num_groups, eps, pad, silu):
    b, c, t, h, w = x.shape
    left, right, top, bottom, front = pad
    return torch.empty((b, c, t + front, h + top + bottom, w + left + right),
                       dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d)
