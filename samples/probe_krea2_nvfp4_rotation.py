"""Settle the question: does Hadamard rotation help NVFP4 on REAL Krea2 activations?

Loads the bf16 model, runs one forward on a random latent, captures the actual
block-input activations (which carry real outlier-channel structure), then for each
captured activation compares projection output error of bf16 vs nvfp4 vs nvfp4+Hadamard.
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/kijai/AI/dev/ComfyUI")
import comfy.ops as ops
from comfy.ldm.krea2.model import SingleStreamDiT

import comfy_kitchen as ck
from comfy_kitchen.tensor.int8_utils import _build_hadamard

DEV = "cuda"
CKPT = "/home/kijai/AI/ComfyUI/models/diffusion_models/Krea2/krea2_turbo_bf16.safetensors"
F8 = ck.float_utils.F8_E4M3_MAX
F4 = ck.float_utils.F4_E2M1_MAX
GROUP = 256


def nvfp4_lin(x, w):
    m = x.shape[0]
    pad = (-m) % 16
    if pad:
        x = torch.cat([x, x.new_zeros(pad, x.shape[1])], 0)
    def q(t):
        s = (t.abs().amax() / (F8 * F4)).float().to(DEV)
        qd, bs = ck.quantize_nvfp4(t.contiguous(), s)
        return qd, s, bs
    xq, xs, xbs = q(x)
    wq, ws, wbs = q(w)
    out = ck.scaled_mm_nvfp4(xq, wq, xs, ws, xbs, wbs, alpha=(xs * ws))
    return out[:m]


def rotate(t):
    h = _build_hadamard(GROUP, device=t.device, dtype=torch.float32)
    g = t.shape[-1] // GROUP
    return (t.reshape(-1, g, GROUP).float() @ h.float()).reshape(t.shape).to(t.dtype)


def rel(o, r):
    d = (o.float() - r.float()).abs()
    return (d.mean() / r.float().abs().mean().clamp_min(1e-6)).item()


def outlier_ratio(a):
    # per-channel L2 across tokens; ratio of max channel to median channel
    n = a.float().reshape(-1, a.shape[-1]).norm(dim=0)
    return (n.max() / n.median()).item()


@torch.inference_mode()
def main():
    import comfy.ldm.common_dit
    from einops import rearrange
    from comfy.ldm.flux.layers import timestep_embedding

    print("Building model + loading bf16 weights on CPU (blocks streamed to GPU)...")
    model = SingleStreamDiT(operations=ops.disable_weight_init, dtype=torch.bfloat16)
    from safetensors.torch import load_file
    sd = load_file(CKPT)
    model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()}, assign=True)
    del sd
    model.eval()

    # Stem on GPU; blocks stay on CPU and are moved JIT (24GB model, ~19GB free).
    for name, mod in model.named_children():
        if name != "blocks":
            mod.to(DEV)

    capture = {}
    targets = [0, 13, 27]

    torch.manual_seed(0)
    x = torch.randn(1, 16, 48, 48, device=DEV, dtype=torch.bfloat16)
    ts = torch.tensor([0.5], device=DEV, dtype=torch.bfloat16)
    context = torch.randn(1, 64, model.txtlayers * model.txtdim, device=DEV, dtype=torch.bfloat16) * 0.1

    # Also hook the layers that 'actsafe' escalates to bf16 (nvfp4 deemed unsafe).
    esc = {"tmlp.2": model.tmlp[2], "tproj.1": model.tproj[1],
           "txtmlp.3": model.txtmlp[3], "last.linear": model.last.linear}
    esc_cap = {}
    for nm, mod in esc.items():
        mod.register_forward_pre_hook(lambda m, a, nm=nm: esc_cap.__setitem__(nm, a[0].detach()))

    print("Running offloaded forward to capture real activations...")
    patch = model.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch
    context = model._unpack_context(context)
    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    img = model.first(img)
    t = model.tmlp(timestep_embedding(ts, model.tdim).unsqueeze(1).to(img.dtype))
    tvec = model.tproj(t)
    context = model.txtfusion(context, mask=None)
    context = model.txtmlp(context)
    combined = torch.cat((context, img), dim=1)
    bs = 1
    txtlen = context.shape[1]
    txtpos = torch.zeros(bs, txtlen, 3, device=DEV, dtype=torch.float32)
    imgids = torch.zeros(h_, w_, 3, device=DEV, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=DEV, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=DEV, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
    freqs = model.pe_embedder(torch.cat((txtpos, imgpos), dim=1))

    for idx, blk in enumerate(model.blocks):
        blk.to(DEV)
        if idx in targets:
            prescale, preshift, pregate, postscale, postshift, postgate = blk.mod(tvec)
            capture[(idx, "attn")] = ((1 + prescale) * blk.prenorm(combined) + preshift).detach()
            attn_out = combined + pregate * blk.attn((1 + prescale) * blk.prenorm(combined) + preshift, freqs, None)
            capture[(idx, "mlp")] = ((1 + postscale) * blk.postnorm(attn_out) + postshift).detach()
            capture[(idx, "wqkvg")] = torch.cat(
                [blk.attn.wq.weight, blk.attn.wk.weight, blk.attn.wv.weight, blk.attn.gate.weight], 0).clone()
            capture[(idx, "wgu")] = torch.cat([blk.mlp.gate.weight, blk.mlp.up.weight], 0).clone()
        combined = blk(combined, tvec, freqs, None)
        blk.to("cpu")

    print("\n=== NVFP4 vs NVFP4+Hadamard on REAL captured activations ===")
    print(f"{'block':>6} {'site':>5} {'outlier_ratio':>14} {'nvfp4':>8} {'nvfp4+had':>10} {'improve':>8}")
    for idx in targets:
        for site, wkey in (("QKVG", "wqkvg"), ("g|up", "wgu")):
            a = capture[(idx, "attn" if site == "QKVG" else "mlp")].to(DEV)
            a = a.reshape(-1, a.shape[-1])
            w = capture[(idx, wkey)].to(DEV)
            ref = F.linear(a, w)
            e0 = rel(nvfp4_lin(a, w), ref)
            e1 = rel(nvfp4_lin(rotate(a), rotate(w)), ref)
            print(f"{idx:>6} {site:>5} {outlier_ratio(a):>14.1f} {e0:>8.4f} {e1:>10.4f} {e0/max(e1,1e-9):>7.2f}x")

    print("\n=== 'actsafe' ESCALATED layers (kept bf16 in nvfp4 ckpt): can Hadamard rescue them? ===")
    print(f"{'layer':>12} {'M':>5} {'outlier':>8} {'nvfp4':>8} {'nvfp4+had':>10} {'improve':>8}")
    for nm, mod in esc.items():
        if nm not in esc_cap:
            continue
        a = esc_cap[nm].reshape(-1, esc_cap[nm].shape[-1])
        w = mod.weight
        if a.shape[-1] % GROUP or w.shape[-1] % GROUP:
            continue
        ref = F.linear(a, w)
        e0 = rel(nvfp4_lin(a, w), ref)
        e1 = rel(nvfp4_lin(rotate(a), rotate(w)), ref)
        print(f"{nm:>12} {a.shape[0]:>5} {outlier_ratio(a):>8.1f} {e0:>8.4f} {e1:>10.4f} {e0/max(e1,1e-9):>7.2f}x")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
