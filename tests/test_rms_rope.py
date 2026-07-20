import pytest
import torch

import comfy_kitchen as ck

from .conftest import assert_values_close, get_capable_backends


def _reference_rms_rope(x, freqs_cis, scale, epsilon, *, split_half=False):
    x_float = x.float()
    rrms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + epsilon)
    x_norm = (x_float * rrms * scale.float()).to(x.dtype).float()
    freqs = freqs_cis.float()
    if split_half:
        pairs = x_norm.reshape(*x.shape[:-1], 2, -1).movedim(-2, -1).unsqueeze(-2)
        out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
        return out.movedim(-1, -2).reshape_as(x).to(x.dtype)

    pairs = x_norm.reshape(*x.shape[:-1], -1, 1, 2)
    out = freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]
    return out.reshape_as(x).to(x.dtype)


_INTERLEAVED_CONFIGS = [
    ("FLUX", "BHND", (1, 24, 4352, 128)),
    ("LTX", "BHND", (2, 32, 4996, 64)),
    ("ZIMAGE", "BNHD", (1, 4096, 30, 128)),
]

_SPLIT_HALF_CONFIGS = [
    ("WAN", "BNHD", (2, 12288, 16, 128)),
    ("FLUX", "BHND", (1, 24, 4352, 128)),
    ("LTX", "BHND", (2, 32, 4996, 64)),
]


def _shapes(layout, config):
    if layout == "BHND":
        batch, heads, seq_len, head_dim = config
        return (
            (batch, heads, seq_len, head_dim),
            (batch, 1, seq_len, head_dim // 2, 2, 2),
            head_dim,
        )

    batch, seq_len, heads, head_dim = config
    return (
        (batch, seq_len, heads, head_dim),
        (1, seq_len, 1, head_dim // 2, 2, 2),
        head_dim,
    )


def _run_backend(op_name, backend, args, layout, monkeypatch):
    if backend == "cuda":
        from comfy_kitchen.backends.eager import rope as eager_rope

        def fail_fallback(*unused_args, **unused_kwargs):
            raise AssertionError(f"{op_name} unexpectedly used the eager fallback")

        with monkeypatch.context() as patch:
            patch.setattr(eager_rope, op_name, fail_fallback)
            with ck.use_backend(backend):
                return getattr(ck, op_name)(*args)

    with ck.use_backend(backend):
        return getattr(ck, op_name)(*args)


def _max_mismatch(freqs_dtype, dtype):
    if freqs_dtype == torch.bfloat16:
        return 0.35
    if freqs_dtype == torch.float16 or dtype == torch.bfloat16:
        return 0.055
    return 1e-5


class TestRMSRope:
    """Interleaved fused RMSNorm + RoPE functionality and correctness tests."""

    @pytest.mark.parametrize("op_name", ["rms_rope", "rms_rope1"])
    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize(
        "freqs_dtype",
        [torch.float32, torch.float16, torch.bfloat16],
        ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"],
    )
    @pytest.mark.parametrize(
        "config_name,layout,config",
        _INTERLEAVED_CONFIGS,
        ids=["FLUX", "LTX", "ZIMAGE"],
    )
    def test_rms_rope_ops(
        self,
        op_name,
        backend,
        device,
        seed,
        dtype,
        freqs_dtype,
        config_name,
        layout,
        config,
        monkeypatch,
    ):
        backends = get_capable_backends(op_name, device)
        if backend not in backends:
            pytest.skip(f"{backend} does not support {op_name} on {device}")

        x_shape, freqs_shape, head_dim = _shapes(layout, config)
        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)
        q_scale = torch.randn(head_dim, dtype=torch.float32, device=device)

        if op_name == "rms_rope":
            q = torch.randn(x_shape, dtype=dtype, device=device)
            k = torch.randn(x_shape, dtype=dtype, device=device)
            k_scale = torch.randn(head_dim, dtype=torch.float32, device=device)
            q_out, k_out = _run_backend(
                op_name,
                backend,
                (q, k, freqs_cis, q_scale, k_scale),
                layout,
                monkeypatch,
            )

            q_ref = None
            k_ref = None
            if backend != "eager":
                with ck.use_backend("eager"):
                    q_ref, k_ref = ck.rms_rope(q, k, freqs_cis, q_scale, k_scale)
            self._validate(q, q_out, layout, dtype, freqs_dtype, config_name, backend, q_ref)
            self._validate(k, k_out, layout, dtype, freqs_dtype, config_name, backend, k_ref)
        else:
            x = torch.randn(x_shape, dtype=dtype, device=device)
            x_out = _run_backend(
                op_name,
                backend,
                (x, freqs_cis, q_scale),
                layout,
                monkeypatch,
            )

            x_ref = None
            if backend != "eager":
                with ck.use_backend("eager"):
                    x_ref = ck.rms_rope1(x, freqs_cis, q_scale)
            self._validate(x, x_out, layout, dtype, freqs_dtype, config_name, backend, x_ref)

    def _validate(self, x, x_out, layout, dtype, freqs_dtype, config_name, backend, ref=None):
        assert x_out.shape == x.shape, f"{layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{layout} dtype mismatch"
        assert x_out.device == x.device

        if ref is not None:
            assert_values_close(
                x_out,
                ref,
                rtol=1e-3,
                atol=1e-3,
                max_mismatch_ratio=_max_mismatch(freqs_dtype, dtype),
                name=(f"{config_name} {layout} x ({backend} vs eager, freqs={freqs_dtype})"),
            )


class TestRMSRopeSplitHalf:
    """Split-half fused RMSNorm + RoPE functionality and correctness tests."""

    @pytest.mark.parametrize("op_name", ["rms_rope_split_half", "rms_rope_split_half1"])
    @pytest.mark.parametrize("backend", ["cuda", "triton", "eager"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize(
        "freqs_dtype",
        [torch.float32, torch.float16, torch.bfloat16],
        ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"],
    )
    @pytest.mark.parametrize(
        "config_name,layout,config",
        _SPLIT_HALF_CONFIGS,
        ids=["WAN", "FLUX", "LTX"],
    )
    def test_split_half_vs_reference(
        self,
        op_name,
        backend,
        device,
        seed,
        dtype,
        freqs_dtype,
        config_name,
        layout,
        config,
        monkeypatch,
    ):
        backends = get_capable_backends(op_name, device)
        if backend not in backends:
            pytest.skip(f"{backend} does not support {op_name} on {device}")

        x_shape, freqs_shape, head_dim = _shapes(layout, config)
        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)
        q_scale = torch.randn(head_dim, dtype=torch.float32, device=device)

        if op_name == "rms_rope_split_half":
            q = torch.randn(x_shape, dtype=dtype, device=device)
            k = torch.randn(x_shape, dtype=dtype, device=device)
            k_scale = torch.randn(head_dim, dtype=torch.float32, device=device)
            q_out, k_out = _run_backend(
                op_name,
                backend,
                (q, k, freqs_cis, q_scale, k_scale),
                layout,
                monkeypatch,
            )
            q_ref = _reference_rms_rope(q, freqs_cis, q_scale, 1e-6, split_half=True)
            k_ref = _reference_rms_rope(k, freqs_cis, k_scale, 1e-6, split_half=True)
            self._validate(q, q_out, q_ref, layout, dtype, freqs_dtype, config_name, backend)
            self._validate(k, k_out, k_ref, layout, dtype, freqs_dtype, config_name, backend)
        else:
            x = torch.randn(x_shape, dtype=dtype, device=device)
            x_out = _run_backend(
                op_name,
                backend,
                (x, freqs_cis, q_scale),
                layout,
                monkeypatch,
            )
            ref = _reference_rms_rope(x, freqs_cis, q_scale, 1e-6, split_half=True)
            self._validate(x, x_out, ref, layout, dtype, freqs_dtype, config_name, backend)

    @pytest.mark.parametrize("op_name", ["rms_rope_split_half", "rms_rope_split_half1"])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
    @pytest.mark.parametrize(
        "freqs_dtype",
        [torch.float32, torch.float16, torch.bfloat16],
        ids=["freqs_fp32", "freqs_fp16", "freqs_bf16"],
    )
    def test_split_half_cross_backend(self, op_name, device, seed, dtype, freqs_dtype, monkeypatch):
        if device != "cuda":
            pytest.skip("cross-backend test requires CUDA")

        backends = get_capable_backends(op_name, device)
        if len(backends) < 2:
            pytest.skip(f"need at least 2 backends for cross-backend test, got {backends}")
        if "eager" not in backends:
            pytest.skip("cross-backend test requires eager as the reference backend")

        batch, seq_len, heads, head_dim = 2, 256, 16, 128
        x_shape = (batch, seq_len, heads, head_dim)
        freqs_shape = (1, seq_len, 1, head_dim // 2, 2, 2)
        freqs_cis = torch.randn(freqs_shape, dtype=freqs_dtype, device=device)
        q_scale = torch.randn(head_dim, dtype=torch.float32, device=device)

        results = {}
        if op_name == "rms_rope_split_half":
            q = torch.randn(x_shape, dtype=dtype, device=device)
            k = torch.randn(x_shape, dtype=dtype, device=device)
            k_scale = torch.randn(head_dim, dtype=torch.float32, device=device)
            for backend in backends:
                results[backend] = _run_backend(
                    op_name,
                    backend,
                    (q, k, freqs_cis, q_scale, k_scale),
                    "BNHD",
                    monkeypatch,
                )
            ref_q, ref_k = results["eager"]
            for backend, (q_out, k_out) in results.items():
                if backend == "eager":
                    continue
                mismatch = _max_mismatch(freqs_dtype, dtype)
                assert_values_close(
                    q_out,
                    ref_q,
                    rtol=1e-3,
                    atol=1e-3,
                    max_mismatch_ratio=mismatch,
                    name=f"rms_rope_split_half q ({backend} vs eager)",
                )
                assert_values_close(
                    k_out,
                    ref_k,
                    rtol=1e-3,
                    atol=1e-3,
                    max_mismatch_ratio=mismatch,
                    name=f"rms_rope_split_half k ({backend} vs eager)",
                )
        else:
            x = torch.randn(x_shape, dtype=dtype, device=device)
            for backend in backends:
                results[backend] = _run_backend(
                    op_name,
                    backend,
                    (x, freqs_cis, q_scale),
                    "BNHD",
                    monkeypatch,
                )
            ref = results["eager"]
            for backend, out in results.items():
                if backend == "eager":
                    continue
                assert_values_close(
                    out,
                    ref,
                    rtol=1e-3,
                    atol=1e-3,
                    max_mismatch_ratio=_max_mismatch(freqs_dtype, dtype),
                    name=f"rms_rope_split_half1 ({backend} vs eager)",
                )

    def _validate(self, x, x_out, ref, layout, dtype, freqs_dtype, config_name, backend):
        assert x_out.shape == x.shape, f"{config_name} {layout} shape mismatch"
        assert x_out.dtype == x.dtype, f"{config_name} {layout} dtype mismatch"
        assert x_out.device == x.device
        assert_values_close(
            x_out,
            ref,
            rtol=1e-3,
            atol=1e-3,
            max_mismatch_ratio=_max_mismatch(freqs_dtype, dtype),
            name=(f"{config_name} {layout} ({backend} vs reference, freqs={freqs_dtype})"),
        )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "op_name",
    ["rms_rope", "rms_rope1", "rms_rope_split_half", "rms_rope_split_half1"],
)
@pytest.mark.parametrize("head_dim", [32, 96, 160])
def test_rms_rope_cuda_multiple_of_32(op_name, head_dim, monkeypatch):
    torch.manual_seed(7)
    q = torch.randn(1, 3, 11, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    freqs = torch.randn(1, 1, 11, head_dim // 2, 2, 2, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(head_dim, device="cuda", dtype=torch.float32)
    split_half = "split_half" in op_name

    if op_name.endswith("1"):
        q_out = _run_backend(op_name, "cuda", (q, freqs, scale), "BHND", monkeypatch)
    else:
        q_out, k_out = _run_backend(op_name, "cuda", (q, k, freqs, scale), "BHND", monkeypatch)

    q_ref = _reference_rms_rope(q, freqs, scale, 1e-6, split_half=split_half)
    assert_values_close(
        q_out,
        q_ref,
        rtol=1e-3,
        atol=1e-3,
        max_mismatch_ratio=0.25,
        name=f"{op_name} D={head_dim} q",
    )
    if not op_name.endswith("1"):
        k_ref = _reference_rms_rope(k, freqs, scale, 1e-6, split_half=split_half)
        assert_values_close(
            k_out,
            k_ref,
            rtol=1e-3,
            atol=1e-3,
            max_mismatch_ratio=0.25,
            name=f"{op_name} D={head_dim} k",
        )
