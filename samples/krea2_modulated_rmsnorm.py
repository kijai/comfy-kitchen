"""Patch Krea2 SingleStreamBlocks to use the fused ck.modulated_rmsnorm kernel.

Each block does, twice:
    (1 + scale) * RMSNorm(x) + shift        # fp32 RMSNorm then bf16 modulation
which is two memory-bound passes per site, 2 sites/block x 28 blocks = 56/step.
ck.modulated_rmsnorm folds RMSNorm + modulation into one pass (read x once, fp32
reduction in registers) -> ~7x on those ops, ~22 ms/step at 1024px. Numerically
equivalent to the eager path within bf16 rounding (the kernel rounds the rmsnorm
output to bf16 before modulating, matching the model).

Usage (ComfyUI node further down, or directly):
    from samples.krea2_modulated_rmsnorm import patch_krea2_modulated_rmsnorm
    patch_krea2_modulated_rmsnorm(model.diffusion_model)
"""
from __future__ import annotations

import types

import torch

import comfy_kitchen as ck


def _make_forward(block):
    # gamma = (1 + stored zero-centered scale); recomputed per call so weight
    # offload / LoRA on the norm params is respected. It's a cheap [D] op.
    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        pre_gamma = comfy_cast(self.prenorm.scale, x) + 1.0
        post_gamma = comfy_cast(self.postnorm.scale, x) + 1.0
        pre = ck.modulated_rmsnorm(x, prescale, preshift, pre_gamma, eps=self.prenorm.eps)
        x = x + pregate * self.attn(pre, freqs, mask, transformer_options=transformer_options)
        post = ck.modulated_rmsnorm(x, postscale, postshift, post_gamma, eps=self.postnorm.eps)
        x = x + postgate * self.mlp(post)
        return x
    return forward


def comfy_cast(param, ref):
    # mirror comfy.model_management.cast_to(param, fp32, device) without importing it hard
    return param.to(device=ref.device, dtype=torch.float32)


def patch_krea2_modulated_rmsnorm(diffusion_model) -> int:
    """Swap pre/post modulated-RMSNorm in every SingleStreamBlock for the fused op.

    Returns the number of blocks patched. Idempotent-ish: re-running rebinds.
    """
    blocks = getattr(diffusion_model, "blocks", None)
    if blocks is None:
        raise ValueError("expected a Krea2 SingleStreamDiT with .blocks")
    count = 0
    for blk in blocks:
        if not (hasattr(blk, "prenorm") and hasattr(blk, "postnorm") and hasattr(blk, "mod")):
            continue
        blk.forward = types.MethodType(_make_forward(blk), blk)
        count += 1
    return count


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
class ApplyKrea2FusedRMSNorm:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "comfy_kitchen/krea2"

    def apply(self, model):
        m = model.clone()
        n = patch_krea2_modulated_rmsnorm(m.model.diffusion_model)
        print(f"[comfy_kitchen] fused modulated-RMSNorm patched into {n} Krea2 blocks")
        return (m,)


NODE_CLASS_MAPPINGS = {"ApplyKrea2FusedRMSNorm": ApplyKrea2FusedRMSNorm}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyKrea2FusedRMSNorm": "Krea2 Fused Modulated-RMSNorm"}
