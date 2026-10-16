from __future__ import annotations

import math

import torch

from .backends import cuda as _cuda_backend

_TILE = 16
_CHUNK = 256   # tile-list entries per block, the fastest measured


def is_available(device: torch.device | None = None) -> bool:
    """Return whether the CUDA gaussian splat kernels can run on the requested device."""
    if device is not None and device.type != "cuda":
        return False
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    return (
        _cuda_backend._EXT_AVAILABLE
        and _cuda_backend._C is not None
        and hasattr(_cuda_backend._C, "gaussian_rasterize_fwd")
    )


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, scales, quats, opacities, viewmat, K, width, height, near, far, dilation):  # noqa: N803
        n = means.shape[0]
        z = means.new_empty(n)
        centre = means.new_empty(n, 2)
        conic = means.new_empty(n, 3)
        boxes = torch.empty(n, 4, device=means.device, dtype=torch.int32)
        wrap = _cuda_backend._wrap_for_dlpack
        _cuda_backend._C.gaussian_project_fwd(
            wrap(means), wrap(scales), wrap(quats), wrap(opacities), wrap(viewmat), wrap(K),
            wrap(z), wrap(centre), wrap(conic), wrap(boxes),
            width, height, near, far, dilation, torch.cuda.current_stream(means.device).cuda_stream,
        )
        ctx.mark_non_differentiable(z, boxes)
        ctx.save_for_backward(means, scales, quats, viewmat, K)
        ctx.args = (width, height, near, far, dilation)
        return z, centre, conic, boxes

    @staticmethod
    def backward(ctx, v_z, v_centre, v_conic, v_boxes):
        means, scales, quats, viewmat, K = ctx.saved_tensors  # noqa: N806
        v_means = torch.empty_like(means)
        v_scales = torch.empty_like(scales)
        v_quats = torch.empty_like(quats)
        v_viewmat = torch.zeros_like(viewmat)
        wrap = _cuda_backend._wrap_for_dlpack
        _cuda_backend._C.gaussian_project_bwd(
            wrap(means), wrap(scales), wrap(quats), wrap(viewmat), wrap(K),
            wrap(v_centre.contiguous()), wrap(v_conic.contiguous()),
            wrap(v_means), wrap(v_scales), wrap(v_quats), wrap(v_viewmat),
            *ctx.args, torch.cuda.current_stream(means.device).cuda_stream,
        )
        return v_means, v_scales, v_quats, None, v_viewmat, None, None, None, None, None, None


def project_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,  # noqa: N803
    width: int,
    height: int,
    near: float,
    far: float,
    dilation: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """EWA-project gaussians (quats wxyz) into a camera, as float32 ``(z, centre, conic, boxes)``: conic is the
    inverse 2D covariance ``(xx, xy, yy)``, boxes the int32 ``(x0, y0, x1, y1)`` extent where alpha reaches 1/255."""
    if not is_available(means.device):
        raise RuntimeError("project_gaussians requires the CUDA extension")
    return _Project.apply(
        means.contiguous(), scales.contiguous(), quats.contiguous(), opacities.contiguous(), viewmat.contiguous(),
        K.contiguous(), width, height, float(near), float(far), float(dilation),
    )


def bin_gaussians(
    z: torch.Tensor,
    boxes: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort projected gaussians into 16x16 tiles, as int32 ``(tile_offsets, flatten_ids)`` front to back."""
    tiles_x = (width + _TILE - 1) // _TILE
    n_tiles = tiles_x * ((height + _TILE - 1) // _TILE)
    order = torch.argsort(z)
    boxes = boxes.contiguous()
    counts = torch.empty(order.shape[0], device=z.device, dtype=torch.int32)
    wrap = _cuda_backend._wrap_for_dlpack
    stream = torch.cuda.current_stream(z.device).cuda_stream
    _cuda_backend._C.gaussian_bin_count(wrap(order), wrap(boxes), wrap(counts), stream)
    ends = counts.cumsum(0)
    total = int(ends[-1]) if ends.numel() else 0
    if total >= 2 ** 31:
        raise RuntimeError(f"bin_gaussians: {total} (tile, gaussian) pairs overflow the int32 offsets")
    # pairs are in depth order, so a stable sort by tile finishes; 16-bit keys halve the passes
    key_dtype = torch.int16 if n_tiles < 2 ** 15 else torch.int32
    keys = torch.empty(total, device=z.device, dtype=key_dtype)
    ids = torch.empty(total, device=z.device, dtype=torch.int32)
    _cuda_backend._C.gaussian_bin(wrap(order), wrap(boxes), wrap(ends.int()), wrap(keys), wrap(ids), tiles_x,
                                  stream)
    keys, perm = torch.sort(keys, stable=True)
    offsets = torch.searchsorted(keys, torch.arange(n_tiles + 1, device=z.device, dtype=key_dtype), out_int32=True)
    return offsets, ids[perm]


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means2d, conics, opacities, colors, background, boxes, tile_offsets, flatten_ids, width, height):
        device = means2d.device
        channels = colors.shape[1]
        image = torch.empty(height, width, channels, device=device, dtype=torch.float32)
        chunks = (torch.diff(tile_offsets) + _CHUNK - 1) // _CHUNK
        chunk_offsets = torch.cat([chunks.new_zeros(1), chunks.cumsum(0, dtype=torch.int32)])
        # an upper bound on the chunk count, so no host read
        n_items = tile_offsets.shape[0] - 1 + flatten_ids.shape[0] // _CHUNK
        cells = n_items * _TILE * _TILE
        # kept for the backward: per chunk pixel (T, colour), then per image pixel T and the last contributor
        state = torch.empty(cells * (channels + 1) + 2 * height * width, device=device, dtype=torch.float32)
        # forward only: per chunk pixel the last contributor (int32) and whether it closed in the chunk (uint8)
        scratch = torch.empty(cells * 5, device=device, dtype=torch.uint8)
        wrap = _cuda_backend._wrap_for_dlpack
        ctx.args = (n_items, width, height, (width + _TILE - 1) // _TILE, _CHUNK)
        _cuda_backend._C.gaussian_rasterize_fwd(
            wrap(means2d), wrap(conics), wrap(opacities), wrap(colors), wrap(background), wrap(boxes),
            wrap(tile_offsets), wrap(flatten_ids), wrap(chunk_offsets), wrap(image), wrap(state), wrap(scratch),
            *ctx.args, torch.cuda.current_stream(device).cuda_stream,
        )
        ctx.save_for_backward(means2d, conics, opacities, colors, background, boxes, tile_offsets, flatten_ids,
                              image, state, chunk_offsets)
        return image

    @staticmethod
    def backward(ctx, v_image):
        means2d, conics, opacities, colors, *rest = ctx.saved_tensors
        v_means2d = torch.zeros_like(means2d)
        v_conics = torch.zeros_like(conics)
        v_opacities = torch.zeros_like(opacities)
        v_colors = torch.zeros_like(colors)
        wrap = _cuda_backend._wrap_for_dlpack
        _cuda_backend._C.gaussian_rasterize_bwd(
            wrap(means2d), wrap(conics), wrap(opacities), wrap(colors), *(wrap(t) for t in rest),
            wrap(v_image.contiguous()), wrap(v_means2d), wrap(v_conics), wrap(v_opacities), wrap(v_colors),
            *ctx.args, torch.cuda.current_stream(means2d.device).cuda_stream,
        )
        return v_means2d, v_conics, v_opacities, v_colors, None, None, None, None, None, None


def rasterize_gaussians(
    means2d: torch.Tensor,
    conics: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    background: torch.Tensor,
    boxes: torch.Tensor,
    tile_offsets: torch.Tensor,
    flatten_ids: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """Composite binned gaussians into a ``(height, width, C)`` float32 image, C = 3 or 5, by gsplat's classic
    rules."""
    if not is_available(means2d.device):
        raise RuntimeError("rasterize_gaussians requires the CUDA extension")
    if colors.shape[1] not in (3, 5) or background.shape != (colors.shape[1],):
        raise ValueError(f"rasterize_gaussians: colors {tuple(colors.shape)} with background {tuple(background.shape)}; "
                         "takes 3 or 5 channels and a background of the same length")
    return _Rasterize.apply(
        means2d.contiguous(), conics.contiguous(), opacities.contiguous(), colors.contiguous(),
        background.contiguous(), boxes.contiguous(), tile_offsets.contiguous(), flatten_ids.contiguous(), width, height,
    )


class _SphericalHarmonics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, campos, sh0, sh_rest, degree):
        colors = means.new_empty(means.shape[0], 3)
        wrap = _cuda_backend._wrap_for_dlpack
        _cuda_backend._C.gaussian_sh_fwd(
            wrap(means), wrap(campos), wrap(sh0), wrap(sh_rest), wrap(colors), degree,
            torch.cuda.current_stream(means.device).cuda_stream,
        )
        ctx.save_for_backward(means, campos, sh0, sh_rest)
        ctx.degree = degree
        return colors

    @staticmethod
    def backward(ctx, v_colors):
        means, campos, sh0, sh_rest = ctx.saved_tensors
        v_means = torch.empty_like(means)
        v_campos = torch.zeros_like(campos)
        v_sh0 = torch.empty_like(sh0)
        v_sh_rest = torch.empty_like(sh_rest)
        wrap = _cuda_backend._wrap_for_dlpack
        _cuda_backend._C.gaussian_sh_bwd(
            wrap(means), wrap(campos), wrap(sh0), wrap(sh_rest), wrap(v_colors.contiguous()), wrap(v_sh0),
            wrap(v_sh_rest), wrap(v_means), wrap(v_campos), ctx.degree, torch.cuda.current_stream(means.device).cuda_stream,
        )
        return v_means, v_campos, v_sh0, v_sh_rest, None


def spherical_harmonics(
    means: torch.Tensor, campos: torch.Tensor, sh0: torch.Tensor, sh_rest: torch.Tensor, degree: int,
) -> torch.Tensor:
    """Raw ``(N, 3)`` colour from real SH up to ``degree`` (0-3) seen from ``campos``, without 3DGS's 0.5 offset.
    ``sh0`` is ``(N, 1, 3)`` and ``sh_rest`` ``(N, R, 3)`` in 3DGS order; bands above ``degree`` are ignored."""
    if not is_available(means.device):
        raise RuntimeError("spherical_harmonics requires the CUDA extension")
    if not 0 <= degree <= 3 or sh_rest.shape[1] < (degree + 1) ** 2 - 1:
        raise ValueError(f"spherical_harmonics: degree {degree} with {sh_rest.shape[1]} higher-band coefficients")
    return _SphericalHarmonics.apply(means.contiguous(), campos.contiguous(), sh0.contiguous(), sh_rest.contiguous(),
                                     degree)


class _SSIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img1, img2):
        ssim = torch.empty_like(img1)
        partials = img1.new_empty(3, *img1.shape)
        wrap = _cuda_backend._wrap_for_dlpack
        _cuda_backend._C.ssim_fwd(
            wrap(img1), wrap(img2), wrap(ssim), wrap(partials), torch.cuda.current_stream(img1.device).cuda_stream,
        )
        ctx.save_for_backward(img1, img2, partials)
        return ssim

    @staticmethod
    def backward(ctx, v_ssim):
        img1, img2, partials = ctx.saved_tensors
        v_img1 = torch.empty_like(img1)
        wrap = _cuda_backend._wrap_for_dlpack
        _cuda_backend._C.ssim_bwd(
            wrap(img1), wrap(img2), wrap(v_ssim.contiguous()), wrap(partials), wrap(v_img1),
            torch.cuda.current_stream(img1.device).cuda_stream,
        )
        return v_img1, None


def ssim_map(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Per-pixel SSIM of two ``(B, C, H, W)`` float32 images with 3DGS's 11x11 window, differentiable in ``img1``."""
    if not is_available(img1.device):
        raise RuntimeError("ssim_map requires the CUDA extension")
    return _SSIM.apply(img1.contiguous(), img2.contiguous())


def inject_position_noise_(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    noise: torch.Tensor,
    scale: float,
) -> None:
    """3DGS-MCMC exploration: move contiguous float32 ``means`` in place by covariance-shaped ``noise``, gated
    to near-transparent splats. ``scales`` are linear and ``opacities`` in [0, 1]."""
    if not is_available(means.device):
        raise RuntimeError("inject_position_noise_ requires the CUDA extension")
    wrap = _cuda_backend._wrap_for_dlpack
    _cuda_backend._C.gaussian_position_noise(
        wrap(means), wrap(scales.contiguous()), wrap(quats.contiguous()), wrap(opacities.contiguous()),
        wrap(noise.contiguous()), float(scale), torch.cuda.current_stream(means.device).cuda_stream,
    )


def sparse_adam_(
    param: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    rows: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> None:
    """``torch.optim.Adam``'s step, in place, on the ``rows`` of a contiguous float32 ``param``; ``step`` counts
    from 1."""
    if not is_available(param.device):
        raise RuntimeError("sparse_adam_ requires the CUDA extension")
    wrap = _cuda_backend._wrap_for_dlpack
    _cuda_backend._C.sparse_adam(
        wrap(param), wrap(grad.contiguous()), wrap(exp_avg), wrap(exp_avg_sq), wrap(rows.contiguous()),
        float(beta1), float(beta2), lr / (1 - beta1 ** step), math.sqrt(1 - beta2 ** step), float(eps),
        torch.cuda.current_stream(param.device).cuda_stream,
    )
