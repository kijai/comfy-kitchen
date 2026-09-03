# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp16_conv3d — CUTLASS fp16-accumulate NDHWC conv3d with fused bias/residual.

Tolerances follow test_fp16_linear.py: fp16 accumulation over K = C*T*R*S
products compounds roughly with sqrt(K) * 2^-11 against torch's fp32-accumulate
reference.
"""

import math

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import fp16_accum_tol, get_capable_backends, rel_err

CL3D = torch.channels_last_3d


def _inputs(c, k, d, h, w, ksize, with_bias=True, with_residual=False, stride=(1, 1, 1)):
    x = torch.randn(1, c, d, h, w, dtype=torch.float16, device="cuda").contiguous(memory_format=CL3D)
    weight = (torch.randn(k, c, *ksize, dtype=torch.float16, device="cuda") * 0.02).contiguous(memory_format=CL3D)
    bias = torch.randn(k, dtype=torch.float16, device="cuda") if with_bias else None
    residual = None
    if with_residual:
        oshape = (1, k, (d - ksize[0]) // stride[0] + 1, (h - ksize[1]) // stride[1] + 1, (w - ksize[2]) // stride[2] + 1)
        residual = torch.randn(oshape, dtype=torch.float16, device="cuda").contiguous(memory_format=CL3D)
    return x, weight, bias, residual


def _ref(x, weight, bias, residual, stride):
    out = functional.conv3d(x.float(), weight.float(), None if bias is None else bias.float(), stride=stride)
    return out if residual is None else out + residual.float()


class TestFp16Conv3d:
    # encoder stage shapes (input already padded, large enough to clear the
    # minimum-launch threshold): 3x3x3 at 128 and 256 channels, the strided
    # downsample, and the 1x1x1 shortcut projection
    @pytest.mark.parametrize(
        "c,k,d,h,w,ksize,stride",
        [
            (128, 128, 5, 130, 130, (3, 3, 3), (1, 1, 1)),
            (256, 256, 5, 130, 130, (3, 3, 3), (1, 1, 1)),
            (128, 128, 9, 129, 129, (3, 3, 3), (1, 2, 2)),
            (128, 256, 5, 128, 128, (1, 1, 1), (1, 1, 1)),
        ],
    )
    @pytest.mark.parametrize("with_bias", [True, False])
    def test_matches_fp32_accum_reference(self, c, k, d, h, w, ksize, stride, with_bias, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x, weight, bias, _ = _inputs(c, k, d, h, w, ksize, with_bias=with_bias, stride=stride)
        got = cuda_backend._cutlass_fp16_conv3d(x, weight, bias, None, list(stride))
        assert got is not None, "fused kernel declined a shape it should serve"
        assert got.is_contiguous(memory_format=CL3D)
        ref = _ref(x, weight, bias, None, stride)
        kdim = c * math.prod(ksize)
        rel = rel_err(got.float(), ref)
        assert rel < fp16_accum_tol(kdim), f"rel={rel:.4f} tol={fp16_accum_tol(kdim):.4f}"

    def test_residual_fused(self, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x, weight, bias, residual = _inputs(128, 128, 5, 130, 130, (3, 3, 3), with_residual=True)
        got = cuda_backend._cutlass_fp16_conv3d(x, weight, bias, residual, [1, 1, 1])
        assert got is not None
        ref = _ref(x, weight, bias, residual, (1, 1, 1))
        rel = rel_err(got.float(), ref)
        assert rel < fp16_accum_tol(128 * 27), f"rel={rel:.4f}"

    def test_small_launch_declined(self, seed, cuda_available):
        """Deep-K low-resolution convs cannot fill the GPU and stay on cuDNN;
        the public op must still return the right answer."""
        if not cuda_available:
            pytest.skip("CUDA required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x, weight, bias, residual = _inputs(512, 512, 7, 18, 18, (3, 3, 3), with_residual=True)
        assert cuda_backend._cutlass_fp16_conv3d(x, weight, bias, residual, [1, 1, 1]) is None
        got = ck.fp16_conv3d(x, weight, bias, residual)
        ref = _ref(x, weight, bias, residual, (1, 1, 1))
        assert rel_err(got.float(), ref) < 2e-3  # cuDNN fp32-accumulate path

    def test_unaligned_channels_fall_back(self, seed, cuda_available):
        """C=3 (the pixel input) cannot use the kernel but must be correct."""
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(1, 3, 5, 34, 34, dtype=torch.float16, device="cuda")
        weight = torch.randn(128, 3, 3, 3, 3, dtype=torch.float16, device="cuda") * 0.1
        got = ck.fp16_conv3d(x, weight, None)
        ref = _ref(x, weight, None, None, (1, 1, 1))
        assert rel_err(got.float(), ref) < 2e-3

    def test_residual_shape_mismatch_falls_back(self, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x, weight, bias, residual = _inputs(128, 128, 5, 130, 130, (3, 3, 3), with_residual=True)
        assert cuda_backend._cutlass_fp16_conv3d(x, weight, bias, residual[:, :64], [1, 1, 1]) is None

    def test_eager_backend_agrees(self, seed, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        if "eager" not in get_capable_backends("fp16_conv3d", device):
            pytest.skip("eager backend not capable")
        x = torch.randn(1, 16, 4, 10, 10, dtype=torch.float32, device=device)
        weight = torch.randn(16, 16, 3, 3, 3, dtype=torch.float32, device=device) * 0.05
        bias = torch.randn(16, dtype=torch.float32, device=device)
        residual = torch.randn(1, 16, 2, 8, 8, dtype=torch.float32, device=device)
        with ck.use_backend("eager"):
            got = ck.fp16_conv3d(x, weight, bias, residual)
        assert torch.allclose(got, _ref(x, weight, bias, residual, (1, 1, 1)), atol=1e-4)
