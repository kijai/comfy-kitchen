import math

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen import gaussian_splat

requires_rasterizer = pytest.mark.skipif(
    not ck.gaussian_splat_is_available(),
    reason="requires the CUDA gaussian splat kernels",
)


def _scene(n=60, width=40, height=36, seed=0, opacity=(0.05, 1.0), channels=3):
    gen = torch.Generator().manual_seed(seed)
    means2d = torch.rand(n, 2, generator=gen, dtype=torch.float64) * torch.tensor([width, height], dtype=torch.float64)
    a = torch.rand(n, generator=gen, dtype=torch.float64) * 12 + 0.8
    c = torch.rand(n, generator=gen, dtype=torch.float64) * 12 + 0.8
    b = (torch.rand(n, generator=gen, dtype=torch.float64) * 2 - 1) * 0.8 * (a * c).sqrt()
    det = a * c - b * b
    conics = torch.stack([c / det, -b / det, a / det], dim=-1)
    opacities = torch.rand(n, generator=gen, dtype=torch.float64) * (opacity[1] - opacity[0]) + opacity[0]
    colors = torch.rand(n, channels, generator=gen, dtype=torch.float64)
    depth = torch.rand(n, generator=gen, dtype=torch.float64)
    return means2d, conics, opacities, colors, depth


def _all_tiles(depth, width, height):
    # every splat in every tile; the per-pixel alpha test culls
    order = torch.argsort(depth).to(torch.int32)
    n_tiles = math.ceil(width / 16) * math.ceil(height / 16)
    offsets = torch.arange(n_tiles + 1, dtype=torch.int32) * order.numel()
    return offsets, order.repeat(n_tiles)


def _boxes(means2d, conics, opacities, width, height, tight):
    """Pixel boxes the rasterizer culls with: the whole image, or the alpha >= 1/255 extent padded a pixel."""
    n = means2d.shape[0]
    if not tight:
        return torch.tensor([0, 0, width, height], dtype=torch.int32).repeat(n, 1)
    det = conics[:, 0] * conics[:, 2] - conics[:, 1] ** 2
    reach = (2 * torch.log((opacities * 255).clamp_min(1))).sqrt()
    rx, ry = reach * (conics[:, 2] / det).sqrt(), reach * (conics[:, 0] / det).sqrt()
    x0 = (means2d[:, 0] - rx - 0.5).ceil() - 1
    y0 = (means2d[:, 1] - ry - 0.5).ceil() - 1
    x1 = (means2d[:, 0] + rx - 0.5).floor() + 2
    y1 = (means2d[:, 1] + ry - 0.5).floor() + 2
    return torch.stack([x0.clamp(0, width), y0.clamp(0, height), x1.clamp(0, width), y1.clamp(0, height)], -1).int()


def _reference(means2d, conics, opacities, colors, background, depth, width, height):
    """gsplat's classic compositing, vectorized over pixels."""
    order = torch.argsort(depth)
    means2d, conics, opacities, colors = means2d[order], conics[order], opacities[order], colors[order]
    ys, xs = torch.meshgrid(torch.arange(height, dtype=means2d.dtype) + 0.5,
                            torch.arange(width, dtype=means2d.dtype) + 0.5, indexing="ij")
    dx = means2d[:, 0] - xs.reshape(-1, 1)
    dy = means2d[:, 1] - ys.reshape(-1, 1)
    sigma = 0.5 * (conics[:, 0] * dx * dx + conics[:, 2] * dy * dy) + conics[:, 1] * dx * dy
    raw = opacities * torch.exp(-sigma)
    valid = (sigma >= 0) & (raw >= 1.0 / 255.0)
    alpha = torch.where(valid, raw.clamp(max=0.999), 0.0)
    log_keep = torch.log1p(-alpha)
    csum = torch.cumsum(log_keep, dim=1)
    live = valid & (csum > math.log(1e-4))
    weight = torch.exp(csum - log_keep) * alpha * live
    log_t = (log_keep * live).sum(dim=1, keepdim=True)
    image = weight @ colors + torch.exp(log_t) * background
    return image.reshape(height, width, -1)


BACKGROUND = (0.5, 0.25, 0.75, 0.4, 0.6)


@requires_rasterizer
@pytest.mark.parametrize("channels", [3, 5])
@pytest.mark.parametrize("tight", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rasterize_matches_reference(seed, tight, channels):
    width, height = 40, 36
    means2d, conics, opacities, colors, depth = _scene(width=width, height=height, seed=seed, channels=channels)
    background = torch.tensor(BACKGROUND[:channels], dtype=torch.float64)
    offsets, ids = _all_tiles(depth, width, height)
    boxes = _boxes(means2d, conics, opacities, width, height, tight)

    ref_inputs = [t.clone().requires_grad_(True) for t in (means2d, conics, opacities, colors)]
    ref = _reference(*ref_inputs, background, depth, width, height)
    weights = torch.rand(height, width, channels, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    ref_grads = torch.autograd.grad((ref * weights).sum(), ref_inputs)

    cuda_inputs = [t.float().cuda().requires_grad_(True) for t in (means2d, conics, opacities, colors)]
    image = ck.rasterize_gaussians(*cuda_inputs, background.float().cuda(), boxes.cuda(), offsets.cuda(), ids.cuda(),
                                   width, height)
    grads = torch.autograd.grad((image * weights.float().cuda()).sum(), cuda_inputs)

    assert image.shape == (height, width, channels)
    torch.testing.assert_close(image.double().cpu(), ref.detach(), atol=1e-4, rtol=0)
    for got, want in zip(grads, ref_grads, strict=True):
        assert float((got.double().cpu() - want).norm() / want.norm()) < 1e-3


@requires_rasterizer
@pytest.mark.parametrize("channels", [3, 5])
@pytest.mark.parametrize("opacity", [(0.005, 0.035), (0.05, 0.2)])
@pytest.mark.parametrize("seed", [0, 1])
def test_rasterize_chunks_match_one_chunk(seed, opacity, channels, monkeypatch):
    # faint splats fold many chunks; denser ones close pixels inside replayed chunks
    width, height = 40, 36
    means2d, conics, opacities, colors, depth = _scene(n=3000, width=width, height=height, seed=seed, opacity=opacity,
                                                       channels=channels)
    offsets, ids = _all_tiles(depth, width, height)
    boxes = _boxes(means2d, conics, opacities, width, height, True).cuda()
    background = torch.tensor(BACKGROUND[:channels], device="cuda")
    weights = torch.rand(height, width, channels, generator=torch.Generator().manual_seed(seed)).cuda()
    runs = []
    for chunk in (64, 1 << 20):
        monkeypatch.setattr(gaussian_splat, "_CHUNK", chunk)
        inputs = [t.float().cuda().requires_grad_(True) for t in (means2d, conics, opacities, colors)]
        image = ck.rasterize_gaussians(*inputs, background, boxes, offsets.cuda(), ids.cuda(), width, height)
        runs.append((image, torch.autograd.grad((image * weights).sum(), inputs)))

    (image, grads), (image_one, grads_one) = runs
    # folding chunks reassociates the transmittance products, so the match is to float rounding
    torch.testing.assert_close(image, image_one, rtol=0, atol=2e-6)
    for got, want in zip(grads, grads_one, strict=True):
        assert float((got - want).norm() / want.norm()) < 1e-5


@requires_rasterizer
@pytest.mark.parametrize("channels", [3, 5])
def test_project_bin_rasterize_uneven_tiles(channels, monkeypatch):
    # a crowded corner: uneven and empty tiles, lists spanning several chunks
    gen = torch.Generator().manual_seed(0)
    n, width, height = 1500, 80, 56
    means = torch.rand(n, 3, generator=gen) * torch.tensor([0.6, 0.5, 1.0]) + torch.tensor([-0.9, 0.1, 2.0])
    scales = torch.rand(n, 3, generator=gen) * 0.03 + 0.005
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1)
    opacities = torch.rand(n, generator=gen) * 0.3 + 0.02
    colors = torch.rand(n, channels, generator=gen)
    intrinsics = torch.tensor([[60.0, 0, 40], [0, 60.0, 28], [0, 0, 1]])
    background = torch.tensor(BACKGROUND[:channels])
    z, centre, conic, boxes = ck.project_gaussians(means.cuda(), scales.cuda(), quats.cuda(), opacities.cuda(),
                                                   torch.eye(4).cuda(), intrinsics.cuda(), width, height,
                                                   0.01, 100.0, 0.3)
    offsets, ids = ck.bin_gaussians(z, boxes, width, height)
    lengths = offsets.diff()
    assert int((lengths == 0).sum()) > 0 and int(lengths.max()) > 4 * 64

    weights = torch.rand(height, width, channels, generator=gen).cuda()
    runs = []
    for chunk in (64, 1 << 20):
        monkeypatch.setattr(gaussian_splat, "_CHUNK", chunk)
        inputs = [t.clone().requires_grad_(True) for t in (centre, conic, opacities.cuda(), colors.cuda())]
        image = ck.rasterize_gaussians(*inputs, background.cuda(), boxes, offsets, ids, width, height)
        runs.append((image, torch.autograd.grad((image * weights).sum(), inputs)))

    (image, grads), (image_one, grads_one) = runs
    torch.testing.assert_close(image, image_one, rtol=0, atol=2e-6)
    for got, want in zip(grads, grads_one, strict=True):
        assert float((got - want).norm() / want.norm()) < 1e-5
    # the reference's float64 alphas can cross the 1/255 cut where float32's did not
    ref = _reference(centre.double().cpu(), conic.double().cpu(), opacities.double(), colors.double(),
                     background.double(), z.double().cpu(), width, height)
    torch.testing.assert_close(image.double().cpu(), ref, rtol=0, atol=1e-3)


@requires_rasterizer
def test_rasterize_empty_tiles_show_background():
    width, height = 20, 18
    n_tiles = math.ceil(width / 16) * math.ceil(height / 16)
    empty = torch.zeros(0, device="cuda")
    background = torch.tensor([0.1, 0.2, 0.3], device="cuda")
    image = ck.rasterize_gaussians(empty.reshape(0, 2), empty.reshape(0, 3), empty, empty.reshape(0, 3), background,
                                   torch.zeros(0, 4, dtype=torch.int32, device="cuda"),
                                   torch.zeros(n_tiles + 1, dtype=torch.int32, device="cuda"),
                                   torch.zeros(0, dtype=torch.int32, device="cuda"), width, height)
    torch.testing.assert_close(image, background.expand(height, width, 3))


def _project_reference(means, scales, quats, viewmat, K, width, height, dilation):  # noqa: N803
    """EWA projection with the Jacobian taken at the view ray clamped to 1.3x the field of view."""
    w, x, y, z = quats.unbind(-1)
    rq = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    rot, trans = viewmat[:3, :3], viewmat[:3, 3]
    cam = means @ rot.T + trans
    m = rot @ (rq * scales.unsqueeze(1))
    cov = m @ m.transpose(1, 2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    invz = 1.0 / cam[:, 2]
    tx = cam[:, 2] * (cam[:, 0] * invz).clamp(-(cx / fx + 0.15 * width / fx), (width - cx) / fx + 0.15 * width / fx)
    ty = cam[:, 2] * (cam[:, 1] * invz).clamp(-(cy / fy + 0.15 * height / fy), (height - cy) / fy + 0.15 * height / fy)
    zero = torch.zeros_like(invz)
    j = torch.stack([fx * invz, zero, -fx * tx * invz * invz, zero, fy * invz, -fy * ty * invz * invz], -1).reshape(-1, 2, 3)
    cov2 = j @ cov @ j.transpose(1, 2)
    a, b, c = cov2[:, 0, 0] + dilation, cov2[:, 0, 1], cov2[:, 1, 1] + dilation
    det = a * c - b * b
    centre = torch.stack([fx * cam[:, 0] * invz + cx, fy * cam[:, 1] * invz + cy], dim=-1)
    return cam[:, 2], centre, torch.stack([c / det, -b / det, a / det], dim=-1), torch.stack([a, c], dim=-1)


def _box_reference(centre, std, opacities, width, height):
    """Pixel box holding the pixel centres where alpha reaches 1/255, end exclusive."""
    reach = (2.0 * torch.log((opacities * 255.0).clamp_min(1.0))).sqrt()
    x0 = (centre[:, 0] - reach * std[:, 0] - 0.5).ceil().clamp(0, width)
    y0 = (centre[:, 1] - reach * std[:, 1] - 0.5).ceil().clamp(0, height)
    x1 = (centre[:, 0] + reach * std[:, 0] - 0.5).floor().clamp(-1, width - 1) + 1
    y1 = (centre[:, 1] + reach * std[:, 1] - 0.5).floor().clamp(-1, height - 1) + 1
    empty = (x1 <= x0) | (y1 <= y0)
    return torch.where(empty[:, None], 0, torch.stack([x0, y0, x1, y1], -1)).int()


@requires_rasterizer
def test_project_matches_reference():
    gen = torch.Generator().manual_seed(0)
    n, width, height, near, far = 4000, 64, 48, 0.2, 50.0
    means = torch.randn(n, 3, generator=gen, dtype=torch.float64) * torch.tensor([1.5, 1.0, 1.0], dtype=torch.float64)
    means[:, 2] += 1.2   # a share of the splats lands behind the near plane
    means[::40, 2] += 100.0   # and some beyond the far one
    scales = torch.rand(n, 3, generator=gen, dtype=torch.float64) * 0.05 + 0.005
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen, dtype=torch.float64), dim=-1)
    opacities = torch.rand(n, generator=gen, dtype=torch.float64)
    angle = 0.2
    viewmat = torch.eye(4, dtype=torch.float64)
    viewmat[:3, :3] = torch.tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]],
                                   dtype=torch.float64)
    viewmat[:3, 3] = torch.tensor([0.1, -0.05, 0.2], dtype=torch.float64)
    K = torch.tensor([[50.0, 0, 32], [0, 52.0, 24], [0, 0, 1]], dtype=torch.float64)  # noqa: N806
    v_centre = torch.randn(n, 2, generator=gen, dtype=torch.float64)
    v_conic = torch.randn(n, 3, generator=gen, dtype=torch.float64)

    depth = means @ viewmat[:3, :3].T[:, 2] + viewmat[2, 3]
    keep = (depth > near) & (depth < far)
    ref_inputs = [t.clone().requires_grad_(True) for t in (means, scales, quats, viewmat)]
    _, centre_ref, conic_ref, var_ref = _project_reference(ref_inputs[0][keep], ref_inputs[1][keep], ref_inputs[2][keep],
                                                           ref_inputs[3], K, width, height, 0.3)
    ref_grads = torch.autograd.grad((centre_ref * v_centre[keep]).sum() + (conic_ref * v_conic[keep]).sum(), ref_inputs)
    boxes_ref = _box_reference(centre_ref.detach(), var_ref.detach().sqrt(), opacities[keep], width, height)

    cuda_inputs = [t.float().cuda().requires_grad_(True) for t in (means, scales, quats, viewmat)]
    z, centre, conic, boxes = ck.project_gaussians(*cuda_inputs[:3], opacities.float().cuda(), cuda_inputs[3],
                                                   K.float().cuda(), width, height, near, far, 0.3)
    grads = torch.autograd.grad((centre * v_centre.float().cuda()).sum() + (conic * v_conic.float().cuda()).sum(),
                                cuda_inputs)

    assert 0 < int(keep.sum()) < n and int((depth >= far).sum()) > 0
    keep_cuda = keep.cuda()
    torch.testing.assert_close(z.double().cpu(), depth, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(centre[keep_cuda].double().cpu(), centre_ref.detach(), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(conic[keep_cuda].double().cpu(), conic_ref.detach(), rtol=1e-3, atol=1e-5)
    assert bool((conic[~keep_cuda] == 0).all()) and bool((boxes[~keep_cuda] == 0).all())
    # a box edge can land on the other side of a pixel centre within float32 rounding
    assert float((boxes[keep_cuda].cpu() == boxes_ref).all(-1).double().mean()) > 0.99
    assert int((boxes_ref[:, 2:] > boxes_ref[:, :2]).all(-1).sum()) > n // 4
    for got, want in zip(grads[:3], ref_grads[:3], strict=True):
        assert bool((got[~keep_cuda] == 0).all())
        assert float((got.double().cpu() - want).norm() / want.norm()) < 1e-4
    assert float((grads[3].double().cpu() - ref_grads[3]).norm() / ref_grads[3].norm()) < 1e-4


@requires_rasterizer
@pytest.mark.parametrize("width, height", [(100, 70), (4096, 2112)])   # 16-bit and 32-bit tile keys
def test_bin_matches_reference(width, height):
    gen = torch.Generator().manual_seed(0)
    n = 500
    tiles_x, tiles_y = math.ceil(width / 16), math.ceil(height / 16)
    z = torch.rand(n, generator=gen)
    x0 = torch.randint(0, width, (n,), generator=gen)
    y0 = torch.randint(0, height, (n,), generator=gen)
    x1 = (x0 + torch.randint(1, 40, (n,), generator=gen)).clamp(max=width)
    y1 = (y0 + torch.randint(1, 40, (n,), generator=gen)).clamp(max=height)
    boxes = torch.stack([x0, y0, x1, y1], -1).int()
    boxes[::7] = 0   # culled splats have an empty box
    tiles = torch.stack([boxes[:, 0] // 16, boxes[:, 1] // 16, (boxes[:, 2] + 15) // 16, (boxes[:, 3] + 15) // 16], -1)

    offsets, ids = ck.bin_gaussians(z.cuda(), boxes.cuda(), width, height)

    offsets, ids = offsets.cpu(), ids.cpu()
    assert offsets.dtype == torch.int32 and ids.dtype == torch.int32
    assert offsets.shape == (tiles_x * tiles_y + 1,)
    for t in range(tiles_x * tiles_y):
        tx, ty = t % tiles_x, t // tiles_x
        hit = ((tiles[:, 0] <= tx) & (tx < tiles[:, 2]) & (tiles[:, 1] <= ty) & (ty < tiles[:, 3])).nonzero().flatten()
        want = hit[torch.argsort(z[hit])]
        assert torch.equal(ids[offsets[t]:offsets[t + 1]].long(), want)
    assert int(offsets[-1]) == ids.numel()


def _sh_reference(dirs, coeffs):
    """Real SH to degree 3 at normalised directions, the explicit polynomials of the 3DGS reference."""
    x, y, z = torch.nn.functional.normalize(dirs, dim=-1).unbind(-1)
    basis = [torch.full_like(x, 0.28209479177387814)]
    if coeffs.shape[1] > 1:
        c1 = 0.4886025119029199
        basis += [-c1 * y, c1 * z, -c1 * x]
    if coeffs.shape[1] > 4:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        basis += [1.0925484305920792 * xy, -1.0925484305920792 * yz, 0.31539156525252005 * (2 * zz - xx - yy),
                  -1.0925484305920792 * xz, 0.5462742152960396 * (xx - yy)]
    if coeffs.shape[1] > 9:
        basis += [-0.5900435899266435 * y * (3 * xx - yy), 2.890611442640554 * xy * z,
                  -0.4570457994644658 * y * (4 * zz - xx - yy), 0.3731763325901154 * z * (2 * zz - 3 * xx - 3 * yy),
                  -0.4570457994644658 * x * (4 * zz - xx - yy), 1.445305721320277 * z * (xx - yy),
                  -0.5900435899266435 * x * (xx - 3 * yy)]
    return (torch.stack(basis, -1).unsqueeze(-1) * coeffs).sum(1)


@requires_rasterizer
@pytest.mark.parametrize("rest", ["exact", "all"])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_spherical_harmonics_matches_reference(degree, rest):
    bands = (degree + 1) ** 2
    gen = torch.Generator().manual_seed(bands)
    n = 3000
    means = torch.randn(n, 3, generator=gen, dtype=torch.float64) * 2.0
    campos = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    coeffs = torch.randn(n, 16, 3, generator=gen, dtype=torch.float64)
    v_colors = torch.randn(n, 3, generator=gen, dtype=torch.float64)

    ref_inputs = [t.clone().requires_grad_(True) for t in (means, campos, coeffs)]
    ref = _sh_reference(ref_inputs[0] - ref_inputs[1], ref_inputs[2][:, :bands])
    ref_grads = torch.autograd.grad((ref * v_colors).sum(), ref_inputs, allow_unused=True)

    # the higher bands either hold exactly what the degree uses, or all 15 with the unused ones ignored
    stored = bands if rest == "exact" else 16
    means_cuda = means.float().cuda().requires_grad_(True)
    campos_cuda = campos.float().cuda().requires_grad_(True)
    sh0 = coeffs[:, :1].float().cuda().requires_grad_(True)
    sh_rest = coeffs[:, 1:stored].float().cuda().requires_grad_(True)
    colors = ck.spherical_harmonics(means_cuda, campos_cuda, sh0, sh_rest, degree)
    grads = torch.autograd.grad((colors * v_colors.float().cuda()).sum(), (means_cuda, campos_cuda, sh0, sh_rest))

    torch.testing.assert_close(colors.double().cpu(), ref.detach(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grads[2].double().cpu(), ref_grads[2][:, :1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(grads[3].double().cpu(), ref_grads[2][:, 1:stored], rtol=1e-4, atol=1e-5)
    if degree == 0:
        assert bool((grads[0] == 0).all()) and bool((grads[1] == 0).all())
    else:
        assert float((grads[0].double().cpu() - ref_grads[0]).norm() / ref_grads[0].norm()) < 1e-4
        assert float((grads[1].double().cpu() - ref_grads[1]).norm() / ref_grads[1].norm()) < 1e-4


def _ssim_reference(a, b):
    coord = torch.arange(11, dtype=a.dtype) - 5
    g = torch.exp(-coord.square() / (2 * 1.5 * 1.5))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).expand(a.shape[1], 1, 11, 11).contiguous()
    def conv(t):
        return torch.nn.functional.conv2d(t, win, padding=5, groups=a.shape[1])

    mu_a, mu_b = conv(a), conv(b)
    var_a, var_b, cov = conv(a * a) - mu_a * mu_a, conv(b * b) - mu_b * mu_b, conv(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (2 * mu_a * mu_b + c1) * (2 * cov + c2) / ((mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2))


@requires_rasterizer
def test_ssim_matches_reference():
    gen = torch.Generator().manual_seed(0)
    a = torch.rand(2, 3, 37, 53, generator=gen, dtype=torch.float64)
    b = (a + 0.2 * torch.randn(2, 3, 37, 53, generator=gen, dtype=torch.float64)).clamp(0, 1)
    v = torch.randn(2, 3, 37, 53, generator=gen, dtype=torch.float64)

    a_ref = a.clone().requires_grad_(True)
    ref = _ssim_reference(a_ref, b)
    (ref_grad,) = torch.autograd.grad((ref * v).sum(), a_ref)

    a_cuda = a.float().cuda().requires_grad_(True)
    out = ck.ssim_map(a_cuda, b.float().cuda())
    (grad,) = torch.autograd.grad((out * v.float().cuda()).sum(), a_cuda)

    torch.testing.assert_close(out.double().cpu(), ref.detach(), rtol=0, atol=2e-5)
    assert float((grad.double().cpu() - ref_grad).norm() / ref_grad.norm()) < 1e-4


@requires_rasterizer
def test_position_noise_matches_reference():
    gen = torch.Generator().manual_seed(0)
    n = 2000
    means = torch.randn(n, 3, generator=gen)
    scales = torch.rand(n, 3, generator=gen) * 0.1
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=gen), dim=-1)
    opacities = torch.cat([torch.rand(n // 2, generator=gen) * 0.01, torch.rand(n - n // 2, generator=gen)])
    noise = torch.randn(n, 3, generator=gen)

    w, x, y, z = quats.double().unbind(-1)
    rot = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    m = rot * scales.double()[:, None, :]
    gate = torch.sigmoid(100 * ((1 - opacities.double()) - 0.995)) * 50.0
    want = means.double() + ((m @ m.transpose(1, 2)) @ (noise.double() * gate[:, None])[..., None])[..., 0]

    out = means.cuda()
    ck.inject_position_noise_(out, scales.cuda(), quats.cuda(), opacities.cuda(), noise.cuda(), 50.0)

    torch.testing.assert_close(out.double().cpu(), want, rtol=1e-5, atol=1e-6)
    assert float((want - means.double()).abs().max()) > 1e-3


@requires_rasterizer
def test_sparse_adam_matches_torch_adam_on_selected_rows():
    gen = torch.Generator().manual_seed(0)
    n = 5000
    param = torch.randn(n, 15, 3, generator=gen)
    rows = torch.rand(n, generator=gen) < 0.5
    ref = torch.nn.Parameter(param[rows].cuda())
    opt = torch.optim.Adam([ref], lr=2e-3, betas=(0.9, 0.999), eps=1e-15)
    got = param.cuda()
    exp_avg, exp_avg_sq = torch.zeros_like(got), torch.zeros_like(got)
    for step in range(1, 4):
        grad = torch.randn(n, 15, 3, generator=gen).cuda()
        ref.grad = grad[rows.cuda()]
        opt.step()
        ck.sparse_adam_(got, grad, exp_avg, exp_avg_sq, rows.cuda(), 2e-3, 0.9, 0.999, 1e-15, step)

    torch.testing.assert_close(got[rows.cuda()], ref.detach(), rtol=1e-5, atol=1e-6)
    assert torch.equal(got[~rows.cuda()], param[~rows].cuda())
    assert bool((exp_avg[~rows.cuda()] == 0).all())
