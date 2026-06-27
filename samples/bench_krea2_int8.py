"""Validate + benchmark the fused-projection INT8 path on REAL Krea2 checkpoint weights.

Reads block.0 of:
  * krea2_turbo_int8convrot.safetensors  (already Hadamard-rotated + per-channel INT8)
  * krea2_turbo_bf16.safetensors          (original bf16 weights, reference)

Checks:
  1. fused INT8 (concat existing int8 rows+scales) == per-layer INT8   -> lossless fusion
  2. fused INT8 vs bf16 reference                                       -> real quant quality
  3. speed: per-layer INT8  vs  fused INT8  vs  bf16
"""
import torch
import torch.nn.functional as F
from safetensors import safe_open

import comfy_kitchen as ck

CKPT_INT8 = "/home/kijai/AI/ComfyUI/models/diffusion_models/Krea2/krea2_turbo_int8convrot.safetensors"
CKPT_BF16 = "/home/kijai/AI/ComfyUI/models/diffusion_models/Krea2/krea2_turbo_bf16.safetensors"
DEV = "cuda"


def load(f, key, device=DEV):
    return f.get_tensor(key).to(device)


def fused_int8(x, w_list, s_list):
    w = torch.cat(w_list, dim=0).contiguous()
    s = torch.cat([s.reshape(-1) for s in s_list]).contiguous()
    return ck.int8_linear(x, w, s, convrot=True), [t.shape[0] for t in w_list]


def perlayer_int8(x, w_list, s_list):
    return torch.cat([ck.int8_linear(x, w, s.reshape(-1), convrot=True) for w, s in zip(w_list, s_list)], dim=-1)


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def rel(out, ref):
    d = (out.float() - ref.float()).abs()
    return d.max().item(), (d.mean() / ref.float().abs().mean().clamp_min(1e-6)).item()


def main():
    M = 4096 + 256
    K = 6144
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

    fi8 = safe_open(CKPT_INT8, "pt")
    fbf = safe_open(CKPT_BF16, "pt")
    B = "blocks.0"

    print(f"=== Real-weight Krea2 fused INT8 (block 0, M={M}, K={K}) ===\n")

    # ---------- QKVG ----------
    names = ["wq", "wk", "wv", "gate"]
    w_i8 = [load(fi8, f"{B}.attn.{n}.weight") for n in names]
    s_i8 = [load(fi8, f"{B}.attn.{n}.weight_scale") for n in names]
    w_bf = [load(fbf, f"{B}.attn.{n}.weight") for n in names]

    ref_bf16 = torch.cat([F.linear(x, w) for w in w_bf], dim=-1)
    out_fused, sizes = fused_int8(x, w_i8, s_i8)
    out_perlayer = perlayer_int8(x, w_i8, s_i8)

    mx_l, rel_l = rel(out_fused, out_perlayer)
    mx_q, rel_q = rel(out_fused, ref_bf16)
    print(f"QKVG  [{K}->{sum(sizes)}]  split={sizes}")
    print(f"  fused vs per-layer INT8 : rel_mean={rel_l:.2e}  max={mx_l:.2e}   (lossless if ~0)")
    print(f"  fused INT8 vs bf16 ref  : rel_mean={rel_q:.4f}  max={mx_q:.4g}")
    t_bf = bench(lambda: torch.cat([F.linear(x, w) for w in w_bf], dim=-1))
    t_pl = bench(lambda: perlayer_int8(x, w_i8, s_i8))
    t_fu = bench(lambda: fused_int8(x, w_i8, s_i8)[0])
    print(f"  bf16={t_bf:.3f}ms  per-layer int8={t_pl:.3f}ms  fused int8={t_fu:.3f}ms"
          f"   | fused vs bf16 {t_bf/t_fu:.2f}x | fused vs per-layer int8 {t_pl/t_fu:.2f}x\n")

    # ---------- SwiGLU gate|up + down ----------
    wg, wu = load(fi8, f"{B}.mlp.gate.weight"), load(fi8, f"{B}.mlp.up.weight")
    sg, su = load(fi8, f"{B}.mlp.gate.weight_scale"), load(fi8, f"{B}.mlp.up.weight_scale")
    wd, sd = load(fi8, f"{B}.mlp.down.weight"), load(fi8, f"{B}.mlp.down.weight_scale")
    wg_bf, wu_bf, wd_bf = (load(fbf, f"{B}.mlp.{n}.weight") for n in ("gate", "up", "down"))
    mlpdim = wg.shape[0]

    def mlp_bf16():
        return F.linear(F.silu(F.linear(x, wg_bf)).mul_(F.linear(x, wu_bf)), wd_bf)

    def mlp_perlayer():
        g = ck.int8_linear(x, wg, sg.reshape(-1), convrot=True)
        u = ck.int8_linear(x, wu, su.reshape(-1), convrot=True)
        return ck.int8_linear(F.silu(g).mul_(u), wd, sd.reshape(-1), convrot=True)

    def mlp_fused():
        gu, _ = fused_int8(x, [wg, wu], [sg, su])
        g, u = gu.split([mlpdim, mlpdim], dim=-1)
        return ck.int8_linear(F.silu(g).mul_(u), wd, sd.reshape(-1), convrot=True)

    mx_l, rel_l = rel(mlp_fused(), mlp_perlayer())
    mx_q, rel_q = rel(mlp_fused(), mlp_bf16())
    print(f"SwiGLU [{K}->{mlpdim}->{K}]")
    print(f"  fused vs per-layer INT8 : rel_mean={rel_l:.2e}  max={mx_l:.2e}")
    print(f"  fused INT8 vs bf16 ref  : rel_mean={rel_q:.4f}  max={mx_q:.4g}")
    t_bf = bench(mlp_bf16)
    t_pl = bench(mlp_perlayer)
    t_fu = bench(mlp_fused)
    print(f"  bf16={t_bf:.3f}ms  per-layer int8={t_pl:.3f}ms  fused int8={t_fu:.3f}ms"
          f"   | fused vs bf16 {t_bf/t_fu:.2f}x | fused vs per-layer int8 {t_pl/t_fu:.2f}x\n")

    # ---------- full-block projection total (what the patcher changes) ----------
    print("Per-block projection GEMM totals (attn QKVG+wo, mlp gate|up+down):")
    wo, so = load(fi8, f"{B}.attn.wo.weight"), load(fi8, f"{B}.attn.wo.weight_scale")

    def block_perlayer():
        qkvg = perlayer_int8(x, w_i8, s_i8)
        q, k, v, g = qkvg.split(sizes, dim=-1)
        a = ck.int8_linear(q, wo, so.reshape(-1), convrot=True)  # stand-in for attn-out proj
        return a + mlp_perlayer()

    def block_fused():
        qkvg, _ = fused_int8(x, w_i8, s_i8)
        q, k, v, g = qkvg.split(sizes, dim=-1)
        a = ck.int8_linear(q, wo, so.reshape(-1), convrot=True)
        return a + mlp_fused()

    t_pl = bench(block_perlayer)
    t_fu = bench(block_fused)
    print(f"  per-layer int8={t_pl:.3f}ms  fused int8={t_fu:.3f}ms   fusion speedup {t_pl/t_fu:.2f}x")
    print(f"  (x28 blocks => ~{(t_pl - t_fu) * 28:.1f} ms saved/step from projection fusion alone)")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
