"""INT8 + Hadamard (ConvRot) acceleration for the Krea2 (K2) single-stream MMDiT.

Strategy (#1): collapse the per-block projections that share an input into a single
rotated/quantized activation feeding one wide INT8 GEMM:

  * Attention   : wq | wk | wv | gate   -> one [K -> Nq+Nk+Nv+Ng] INT8 GEMM, then split
  * SwiGLU MLP  : gate | up             -> one [K -> 2*mlpdim]      INT8 GEMM, then split

Because the online Hadamard rotation acts on the *shared* input dim K, all the fused
projections use the same rotation, so the rotate+row-quant of the activation runs ONCE
per group instead of 4x (attn) / 2x (mlp). Weights are rotated offline (W @ H^T, with the
symmetric normalized H so (xH)(WH)^T == xW^T) and quantized per-output-channel to INT8.

The remaining single projections (wo, down) just use convrot INT8 directly.

This module is import-safe without ComfyUI; the ComfyUI node + attention forward only
import comfy lazily.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy_kitchen as ck
from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight

GROUP = 256


# ---------------------------------------------------------------------------
# Offline weight preparation: concat -> rotate -> per-output-channel INT8
# ---------------------------------------------------------------------------
def _quantize_weight_int8_perchannel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel (per-row) INT8 quant of an already-rotated [N, K] weight."""
    absmax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = absmax / 127.0
    q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    return q.contiguous(), scale.reshape(-1).to(torch.float32).contiguous()


def build_fused_int8_weight(
    weights: list[torch.Tensor], device: torch.device | str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Concatenate [N_i, K] weights along output dim, Hadamard-rotate, per-channel INT8 quant.

    Returns (q_int8 [sumN, K], scale [sumN], split_sizes).
    """
    k = weights[0].shape[1]
    assert all(w.shape[1] == k for w in weights), "fused projections must share input dim K"
    if k % GROUP != 0:
        raise ValueError(f"K={k} must be divisible by {GROUP} for ConvRot")
    sizes = [w.shape[0] for w in weights]
    w_cat = torch.cat([w.to(device=device, dtype=torch.float32) for w in weights], dim=0)
    h = _build_hadamard(GROUP, device=device, dtype=torch.float32)
    w_rot = _rotate_weight(w_cat, h, GROUP)
    q, s = _quantize_weight_int8_perchannel(w_rot)
    return q, s, sizes


# ---------------------------------------------------------------------------
# Fused replacement modules
# ---------------------------------------------------------------------------
class Int8FusedAttention(nn.Module):
    """Drop-in for krea2 Attention: fused QKVG projection + convrot INT8 wo."""

    def __init__(self, attn: nn.Module, device: torch.device | str = "cuda"):
        super().__init__()
        self.heads = attn.heads
        self.kvheads = attn.kvheads
        self.headdim = attn.headdim
        # qknorm is cheap (per-head RMSNorm, fp32) — keep the original module.
        self.qknorm = attn.qknorm

        qkvg_w, qkvg_s, sizes = build_fused_int8_weight(
            [attn.wq.weight.data, attn.wk.weight.data, attn.wv.weight.data, attn.gate.weight.data],
            device,
        )
        self.register_buffer("qkvg_w", qkvg_w)
        self.register_buffer("qkvg_s", qkvg_s)
        self.split_sizes = sizes  # [Nq, Nk, Nv, Ngate]

        wo_w, wo_s, _ = build_fused_int8_weight([attn.wo.weight.data], device)
        self.register_buffer("wo_w", wo_w)
        self.register_buffer("wo_s", wo_s)

    @torch.no_grad()
    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        from einops import rearrange

        from comfy.ldm.flux.math import apply_rope
        from comfy.ldm.modules.attention import optimized_attention_masked

        qkvg = ck.int8_linear(x, self.qkvg_w, self.qkvg_s, convrot=True)
        q, k, v, gate = qkvg.split(self.split_sizes, dim=-1)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = optimized_attention_masked(q, k, v, self.heads, mask=mask, skip_reshape=True,
                                         transformer_options=transformer_options)
        out = out * F.sigmoid(gate)
        return ck.int8_linear(out, self.wo_w, self.wo_s, convrot=True)


class Int8FusedSwiGLU(nn.Module):
    """Drop-in for krea2 SwiGLU: fused gate|up projection + convrot INT8 down."""

    def __init__(self, mlp: nn.Module, device: torch.device | str = "cuda"):
        super().__init__()
        gu_w, gu_s, sizes = build_fused_int8_weight([mlp.gate.weight.data, mlp.up.weight.data], device)
        self.register_buffer("gateup_w", gu_w)
        self.register_buffer("gateup_s", gu_s)
        self.mlpdim = sizes[0]

        down_w, down_s, _ = build_fused_int8_weight([mlp.down.weight.data], device)
        self.register_buffer("down_w", down_w)
        self.register_buffer("down_s", down_s)

    @torch.no_grad()
    def forward(self, x):
        gu = ck.int8_linear(x, self.gateup_w, self.gateup_s, convrot=True)
        gate, up = gu.split([self.mlpdim, self.mlpdim], dim=-1)
        h = F.silu(gate).mul_(up)
        return ck.int8_linear(h, self.down_w, self.down_s, convrot=True)


# ---------------------------------------------------------------------------
# Standalone correctness + microbenchmark (no ComfyUI required)
# ---------------------------------------------------------------------------
def _bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def _rel(out, ref):
    diff = (out.float() - ref.float()).abs()
    return diff.max().item(), diff.mean().item(), (diff.mean() / ref.float().abs().mean().clamp_min(1e-6)).item()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    dev = "cuda"
    dt = torch.bfloat16
    torch.manual_seed(0)
    K = 6144
    M = 4096 + 256  # img tokens (1024^2) + text

    def make_linear(n, k):
        w = (torch.randn(n, k, device=dev, dtype=dt) / (k ** 0.5))
        return w

    print(f"=== Krea2 fused INT8 projection check  (M={M}, K={K}, dtype={dt}) ===\n")

    # ---- QKVG fusion ----
    Nq, Nkv, Ng = 48 * 128, 12 * 128, 6144
    wq, wk, wv, wg = make_linear(Nq, K), make_linear(Nkv, K), make_linear(Nkv, K), make_linear(Ng, K)
    x = torch.randn(M, K, device=dev, dtype=dt)

    qkvg_w, qkvg_s, sizes = build_fused_int8_weight([wq, wk, wv, wg], dev)
    ref = torch.cat([F.linear(x, w) for w in (wq, wk, wv, wg)], dim=-1)
    out = ck.int8_linear(x, qkvg_w, qkvg_s, convrot=True)
    mx, mean, rel = _rel(out, ref)
    print(f"QKVG  [{K}->{sum(sizes)}]  rel_mean={rel:.4f}  max={mx:.4g}  mean={mean:.4g}")

    t_bf16 = _bench(lambda: torch.cat([F.linear(x, w) for w in (wq, wk, wv, wg)], dim=-1))
    t_int8 = _bench(lambda: ck.int8_linear(x, qkvg_w, qkvg_s, convrot=True))
    print(f"      bf16 4xLinear: {t_bf16:.3f} ms   fused int8: {t_int8:.3f} ms   speedup {t_bf16/t_int8:.2f}x\n")

    # ---- gate|up fusion ----
    mlpdim = 16384
    wgate, wup, wdown = make_linear(mlpdim, K), make_linear(mlpdim, K), make_linear(K, mlpdim)
    gu_w, gu_s, gu_sizes = build_fused_int8_weight([wgate, wup], dev)
    down_w, down_s, _ = build_fused_int8_weight([wdown], dev)

    def mlp_bf16(x):
        return F.linear(F.silu(F.linear(x, wgate)).mul_(F.linear(x, wup)), wdown)

    def mlp_int8(x):
        gu = ck.int8_linear(x, gu_w, gu_s, convrot=True)
        g, u = gu.split([mlpdim, mlpdim], dim=-1)
        h = F.silu(g).mul_(u)
        return ck.int8_linear(h, down_w, down_s, convrot=True)

    ref_m = mlp_bf16(x)
    out_m = mlp_int8(x)
    mx, mean, rel = _rel(out_m, ref_m)
    print(f"SwiGLU [{K}->{mlpdim}->{K}]  rel_mean={rel:.4f}  max={mx:.4g}  mean={mean:.4g}")
    t_bf16 = _bench(lambda: mlp_bf16(x))
    t_int8 = _bench(lambda: mlp_int8(x))
    print(f"      bf16: {t_bf16:.3f} ms   fused int8: {t_int8:.3f} ms   speedup {t_bf16/t_int8:.2f}x\n")

    # ---- whole-block proxy (proj GEMMs only) ----
    print("Note: rel_mean is activation-relative; <~0.03 is typical for rotated INT8.")
