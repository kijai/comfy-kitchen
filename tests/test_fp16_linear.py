# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fp16_linear — CUTLASS fp16-accumulate GEMM with an optional fused residual.

The CUDA path accumulates in fp16 (the allow_fp16_accumulation numerics), so
comparisons against the fp32-accumulate torch reference use a tolerance that
grows with K: rounding error compounds roughly with sqrt(K) * 2^-11.
"""


import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import fp16_accum_tol, get_capable_backends, rel_err


class TestFp16Linear:
    # covers the identity-swizzle configs and the deep-K stream-K path
    @pytest.mark.parametrize(
        "m,n,k",
        [
            (1797, 6144, 2048),
            (1797, 2048, 2048),
            (1797, 16384, 2048),
            (1797, 2048, 8192),
            (512, 512, 512),
            (37, 264, 128),
        ],
    )
    @pytest.mark.parametrize("with_bias", [True, False])
    def test_matches_fp32_accum_reference(self, m, n, k, with_bias, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        b = torch.randn(n, dtype=torch.float16, device="cuda") if with_bias else None

        ref = torch.nn.functional.linear(x, w, b).float()
        got = ck.fp16_linear(x, w, b).float()
        rel = rel_err(got, ref)
        assert rel < fp16_accum_tol(k), f"rel={rel:.4f} tol={fp16_accum_tol(k):.4f}"

    @pytest.mark.parametrize("m,n,k", [(1797, 2048, 2048), (1797, 2048, 8192), (512, 512, 512)])
    def test_residual_matches_eager_addcmul(self, m, n, k, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        b = torch.randn(n, dtype=torch.float16, device="cuda")
        resid = torch.randn(m, n, dtype=torch.float16, device="cuda")
        rscale = torch.randn(n, dtype=torch.float16, device="cuda")

        # same fp16-accum GEMM either way; only the epilogue rounding differs
        plain = ck.fp16_linear(x, w, b)
        ref = torch.addcmul(resid, plain, rscale).float()
        got = ck.fp16_linear(x, w, b, residual=resid, residual_scale=rscale).float()
        rel = rel_err(got, ref)
        assert rel < 1e-2, f"rel={rel:.3e}"

    def test_residual_without_bias(self, seed, cuda_available):
        """Bias-less residual projections fuse too (nullptr bias broadcasts 0)."""
        if not cuda_available:
            pytest.skip("CUDA required")

        m, n, k = 1797, 2048, 2048
        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        resid = torch.randn(m, n, dtype=torch.float16, device="cuda")
        rscale = torch.randn(n, dtype=torch.float16, device="cuda")

        plain = ck.fp16_linear(x, w, None)
        ref = torch.addcmul(resid, plain, rscale).float()
        got = ck.fp16_linear(x, w, None, residual=resid, residual_scale=rscale).float()
        rel = rel_err(got, ref)
        assert rel < 1e-2, f"rel={rel:.3e}"

    def test_3d_input(self, seed, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(2, 512, 2048, dtype=torch.float16, device="cuda")
        w = torch.randn(1024, 2048, dtype=torch.float16, device="cuda") * 0.02
        b = torch.randn(1024, dtype=torch.float16, device="cuda")
        resid = torch.randn(2, 512, 1024, dtype=torch.float16, device="cuda")
        rscale = torch.randn(1024, dtype=torch.float16, device="cuda")

        got = ck.fp16_linear(x, w, b, residual=resid, residual_scale=rscale)
        assert got.shape == (2, 512, 1024)
        ref = torch.addcmul(resid, torch.nn.functional.linear(x, w, b), rscale).float()
        rel = rel_err(got, ref)
        assert rel < fp16_accum_tol(2048), f"rel={rel:.4f}"

    def test_unaligned_k_falls_back(self, seed, cuda_available):
        """K % 8 != 0 cannot run the fused kernel but must still be correct."""
        if not cuda_available:
            pytest.skip("CUDA required")

        x = torch.randn(64, 132, dtype=torch.float16, device="cuda")
        w = torch.randn(96, 132, dtype=torch.float16, device="cuda") * 0.02
        got = ck.fp16_linear(x, w, None).float()
        ref = torch.nn.functional.linear(x, w, None).float()
        # the fallback is torch's own linear, so this is near-exact
        assert ((got - ref).abs().max()).item() < 1e-3

    def test_misaligned_input_falls_back(self, seed, cuda_available):
        """A 16-byte-misaligned contiguous view must fall back to torch, not
        hand the pointer to the CUTLASS kernel (async misaligned-address)."""
        if not cuda_available:
            pytest.skip("CUDA required")

        base = torch.randn(64 * 2048 + 4, dtype=torch.float16, device="cuda")
        x = base[4:4 + 64 * 2048].reshape(64, 2048)
        assert x.data_ptr() % 16 == 8
        w = torch.randn(512, 2048, dtype=torch.float16, device="cuda") * 0.02
        got = ck.fp16_linear(x, w, None).float()
        torch.cuda.synchronize()
        ref = torch.nn.functional.linear(x, w, None).float()
        assert ((got - ref).abs().max()).item() < 1e-3

    def test_residual_requires_scale(self, cuda_available):
        if not cuda_available:
            pytest.skip("CUDA required")
        x = torch.randn(4, 128, dtype=torch.float16, device="cuda")
        w = torch.randn(64, 128, dtype=torch.float16, device="cuda")
        resid = torch.randn(4, 64, dtype=torch.float16, device="cuda")
        with pytest.raises(ValueError, match="residual"):
            ck.fp16_linear(x, w, None, residual=resid)

    def test_eager_backend_agrees(self, seed, cuda_available):
        device = "cuda" if cuda_available else "cpu"
        if "eager" not in get_capable_backends("fp16_linear", device):
            pytest.skip("eager backend not capable")

        x = torch.randn(256, 512, dtype=torch.float16, device=device)
        w = torch.randn(128, 512, dtype=torch.float16, device=device) * 0.02
        b = torch.randn(128, dtype=torch.float16, device=device)
        resid = torch.randn(256, 128, dtype=torch.float16, device=device)
        rscale = torch.randn(128, dtype=torch.float16, device=device)

        with ck.use_backend("eager"):
            got = ck.fp16_linear(x, w, b, residual=resid, residual_scale=rscale)
        ref = torch.addcmul(resid, torch.nn.functional.linear(x, w, b), rscale)
        assert ((got.float() - ref.float()).abs().max()).item() < 1e-2

    @pytest.mark.parametrize(
        "m,n,k,served",
        [
            (1797, 6144, 2048, True),   # decoder qkv: 360 plain tiles
            (1797, 2048, 8192, True),   # decoder w2: stream-K, 120 tiles
            (1024, 2048, 2048, False),  # 64 plain tiles: cuBLAS split-K is ~2x faster
            (64, 1024, 4096, False),    # 4 tiles: ~10x slower than cuBLAS
            (64, 2048, 8192, False),    # stream-K but only 16 tiles
        ],
    )
    def test_small_launches_are_declined(self, m, n, k, served, seed, cuda_available):
        """The kernel declines launches too small to fill the GPU so the caller
        runs cuBLAS; the public op stays correct either way."""
        if not cuda_available:
            pytest.skip("CUDA required")
        from comfy_kitchen.backends import cuda as cuda_backend

        x = torch.randn(m, k, dtype=torch.float16, device="cuda")
        w = torch.randn(n, k, dtype=torch.float16, device="cuda") * 0.02
        out = torch.empty(m, n, dtype=torch.float16, device="cuda")
        wrap = cuda_backend._wrap_for_dlpack
        ok = cuda_backend._C.cutlass_fp16_linear(
            wrap(x), wrap(w), wrap(cuda_backend._empty_cuda_tensor(x.device, torch.float16)), wrap(out),
            torch.cuda.current_stream().cuda_stream)
        assert ok == served
        got = ck.fp16_linear(x, w, None).float()
        ref = torch.nn.functional.linear(x, w, None).float()
        assert rel_err(got, ref) < fp16_accum_tol(k)
