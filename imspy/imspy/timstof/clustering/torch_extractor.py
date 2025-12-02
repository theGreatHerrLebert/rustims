import gc
import math
from typing import Iterable, List

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F


# -------------------------------------------------------------------
# Global numeric constants used in refiners
# -------------------------------------------------------------------

LOG_MIN_SIGMA = math.log(1e-3)
LOG_MIN_AMP = math.log(1e-6)


# ---------------------------
# Small utilities
# ---------------------------

def _gaussian_kernel1d(
    sigma: float,
    truncate: float = 3.0,
    device=None,
    dtype=None,
):
    """Create a 1D Gaussian kernel normalized to sum=1."""
    if sigma is None or sigma <= 0:
        return None
    radius = int(truncate * sigma + 0.5)
    if radius < 1:
        return None
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k  # [2R+1]


def _gaussian_blur_2d(
    X: torch.Tensor,
    sigma_scan: float | None = None,   # along scan axis (dim 0 after transpose)
    sigma_tof: float | None = None,    # along tof axis (dim 1 after transpose)
    truncate: float = 3.0,
):
    """
    Separable Gaussian blur on X of shape (H_scan, W_tof).
    Uses reflect padding, same-shape output. If sigma_* <= 0 or None, that axis is skipped.
    """
    if X.ndim != 2:
        raise ValueError(f"_gaussian_blur_2d expects (H,W), got {X.shape}")

    dev, dt = X.device, X.dtype
    Y = X

    # blur along scan axis (rows)
    if sigma_scan is not None and sigma_scan > 0:
        k = _gaussian_kernel1d(sigma_scan, truncate=truncate, device=dev, dtype=dt)
        if k is not None:
            k2d = k.view(1, 1, -1, 1)  # [out_c, in_c, kH, 1]
            pad = (0, 0, k.shape[0] // 2, k.shape[0] // 2)
            Y4 = Y[None, None]          # [1,1,H,W]
            Y4 = F.pad(Y4, pad, mode="reflect")
            Y = F.conv2d(Y4, k2d)[0, 0]  # back to [H,W]

    # blur along tof axis (cols)
    if sigma_tof is not None and sigma_tof > 0:
        k = _gaussian_kernel1d(sigma_tof, truncate=truncate, device=dev, dtype=dt)
        if k is not None:
            k2d = k.view(1, 1, 1, -1)  # [out_c, in_c, 1, kW]
            pad = (k.shape[0] // 2, k.shape[0] // 2, 0, 0)
            Y4 = Y[None, None]
            Y4 = F.pad(Y4, pad, mode="reflect")
            Y = F.conv2d(Y4, k2d)[0, 0]

    return Y


def _ensure_odd(x: int) -> int:
    x = int(x)
    return x if (x % 2 == 1) else (x + 1)


def _apply_scale_tensor(X: torch.Tensor, scale: str):
    """
    Returns (X_scaled, inv_fn). inv_fn maps scaled -> original.
    """
    if scale in (None, "none"):
        return X, (lambda y: y)
    if scale == "sqrt":
        # classic variance stabilization for Poisson-ish intensities
        Xs = torch.sqrt(torch.clamp(X, min=0.0))
        return Xs, (lambda y: torch.clamp(y, min=0.0) ** 2)
    if scale == "cbrt":
        Xs = torch.cbrt(torch.clamp(X, min=0.0))
        return Xs, (lambda y: torch.clamp(y, min=0.0) ** 3)
    if scale == "log1p":
        Xs = torch.log1p(torch.clamp(X, min=0.0))
        return Xs, (lambda y: torch.expm1(y))
    raise ValueError(f"unknown scale={scale!r}")


# ---------------------------
# Batched refiners (unchanged semantics, cleaned internals)
# ---------------------------

def _batched_refine_adam(
    patches: torch.Tensor,
    mu_i0: torch.Tensor, mu_j0: torch.Tensor,
    sigma_i0: torch.Tensor, sigma_j0: torch.Tensor,
    amp0: torch.Tensor, base0: torch.Tensor,
    *,
    iters: int = 8,
    lr: float = 0.2,
    loss: str = "huber",
    mask_k: float = 2.5,
    refine_scan: bool = True,
    refine_tof: bool = False,
    refine_sigma_scan: bool = True,
    refine_sigma_tof: bool = False,
):
    """
    Adam-based refinement with masked residuals.
    """
    if patches.numel() == 0:
        z = patches.new_zeros((0,))
        return z, z, z, z, z, z

    N, H, W = patches.shape
    dev, dtype = patches.device, patches.dtype
    I = torch.arange(H, device=dev, dtype=dtype).view(1, H, 1).expand(N, H, 1)
    J = torch.arange(W, device=dev, dtype=dtype).view(1, 1, W).expand(N, 1, W)

    P = patches.detach()  # fit to a stopped-copy

    with torch.enable_grad():
        mu_i = mu_i0.detach().to(dev).clone()
        mu_i.requires_grad_(refine_scan)

        mu_j = mu_j0.detach().to(dev).clone()
        mu_j.requires_grad_(refine_tof)

        lsi = sigma_i0.clamp_min(1e-3).log().detach().to(dev).clone()
        lsj = sigma_j0.clamp_min(1e-3).log().detach().to(dev).clone()
        lsi.requires_grad_(refine_scan and refine_sigma_scan)
        lsj.requires_grad_(refine_tof and refine_sigma_tof)

        la = amp0.clamp_min(1e-6).log().detach().to(dev).clone()
        la.requires_grad_(True)

        b = base0.detach().to(dev).clone()
        b.requires_grad_(True)

        params = [p for p in (mu_i, mu_j, lsi, lsj, la, b) if p.requires_grad]
        if not params:
            return (
                mu_i.detach(), mu_j.detach(),
                lsi.exp().detach(), lsj.exp().detach(),
                la.exp().detach(), b.detach()
            )

        opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

        def loss_fn(yhat, y, delta=1.0):
            r = yhat - y
            if loss == "huber":
                absr = r.abs()
                return torch.where(
                    absr <= delta,
                    0.5 * r * r,
                    delta * (absr - 0.5 * delta),
                ).mean()
            elif loss == "cauchy":
                return (delta ** 2 * torch.log1p((r / delta) ** 2)).mean()
            else:
                return (r * r).mean()

        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            si = lsi.exp()
            sj = lsj.exp()
            di = I - mu_i.view(-1, 1, 1)
            dj = J - mu_j.view(-1, 1, 1)

            # detached mask narrows the influence region
            Mi = torch.exp(-0.5 * (di / (mask_k * si.view(-1, 1, 1))) ** 2)
            Mj = torch.exp(-0.5 * (dj / (mask_k * sj.view(-1, 1, 1))) ** 2)
            M = (Mi * Mj).detach()

            q = (di / si.view(-1, 1, 1)) ** 2 + (dj / sj.view(-1, 1, 1)) ** 2
            yhat = b.view(-1, 1, 1) + la.exp().view(-1, 1, 1) * torch.exp(-0.5 * q)
            L = loss_fn(yhat * M, P * M)
            L.backward()
            opt.step()

            # guards on logs
            lsi.data.clamp_(min=LOG_MIN_SIGMA)
            lsj.data.clamp_(min=LOG_MIN_SIGMA)
            la.data.clamp_(min=LOG_MIN_AMP)

    return (
        mu_i.detach(), mu_j.detach(),
        lsi.exp().detach(), lsj.exp().detach(),
        la.exp().detach(), b.detach()
    )


def _batched_refine_gauss_newton(
    patches: torch.Tensor,
    mu_i0: torch.Tensor, mu_j0: torch.Tensor,
    sigma_i0: torch.Tensor, sigma_j0: torch.Tensor,
    amp0: torch.Tensor, base0: torch.Tensor,
    *,
    iters: int = 5,
    damping: float = 1e-2,          # slightly stronger default
    refine_scan: bool = True,
    refine_tof: bool = False,
    refine_sigma_scan: bool = True,
    refine_sigma_tof: bool = False,
    mask_k: float = 2.5,            # match ADAM’s mask behavior
    force_dtype: torch.dtype | None = None,
):
    """
    Gauss–Newton/LM on diagonal Σ, with masked residuals, double precision math,
    and safe clamping. Enabled columns (order): [mu_i?, mu_j?, lsi?, lsj?, la, b]
    """
    if patches.numel() == 0:
        z = patches.new_zeros((0,))
        return z, z, z, z, z, z

    if force_dtype is not None:
        patches = patches.to(dtype=force_dtype)

    # Work in float64 for stability
    P = patches.to(dtype=torch.float64)
    dev = P.device
    N, H, W = P.shape
    I = torch.arange(H, device=dev, dtype=P.dtype).view(1, H, 1).expand(N, H, 1)
    J = torch.arange(W, device=dev, dtype=P.dtype).view(1, 1, W).expand(N, 1, W)

    mu_i = mu_i0.to(dev, dtype=P.dtype).clone()
    mu_j = mu_j0.to(dev, dtype=P.dtype).clone()
    lsi = sigma_i0.clamp_min(1e-3).log().to(dev, dtype=P.dtype).clone()
    lsj = sigma_j0.clamp_min(1e-3).log().to(dev, dtype=P.dtype).clone()
    la = amp0.clamp_min(1e-6).log().to(dev, dtype=P.dtype).clone()
    b = base0.to(dev, dtype=P.dtype).clone()

    # Which parameters to include
    cols: list[str] = []
    if refine_scan:
        cols.append("mu_i")
    if refine_tof:
        cols.append("mu_j")
    if refine_scan and refine_sigma_scan:
        cols.append("lsi")
    if refine_tof and refine_sigma_tof:
        cols.append("lsj")
    cols += ["la", "b"]

    # Per-batch dynamic clamps for amplitude (prevents exp overflow)
    patch_max = torch.amax(P.view(N, -1), dim=1)
    amp_cap = (patch_max - b).clamp_min(1e-3)
    la_max = torch.log(amp_cap) + 2.0   # small safety margin
    la_min = torch.log(torch.full_like(la_max, 1e-6))

    def forward_and_residual(mu_i, mu_j, lsi, lsj, la, b):
        si = lsi.exp()
        sj = lsj.exp()
        di = I - mu_i.view(-1, 1, 1)
        dj = J - mu_j.view(-1, 1, 1)

        # Detached Gaussian mask like ADAM to localize influence
        Mi = torch.exp(-0.5 * (di / (mask_k * si.view(-1, 1, 1))) ** 2)
        Mj = torch.exp(-0.5 * (dj / (mask_k * sj.view(-1, 1, 1))) ** 2)
        M = (Mi * Mj).detach()

        q = (di / si.view(-1, 1, 1)) ** 2 + (dj / sj.view(-1, 1, 1)) ** 2
        A = la.exp().view(-1, 1, 1)
        G = torch.exp(-0.5 * q)
        yhat = b.view(-1, 1, 1) + A * G
        R = (P - yhat) * M
        return R, M, di, dj, si, sj, A, G

    R, M, di, dj, si, sj, A, G = forward_and_residual(mu_i, mu_j, lsi, lsj, la, b)
    prev_loss = torch.mean(R * R)

    lam = torch.as_tensor(damping, device=dev, dtype=P.dtype)

    for _ in range(iters):
        # Partials for enabled params with mask M applied
        Jcols: list[torch.Tensor] = []
        if "mu_i" in cols:
            J_mu_i = -(A * G * (di / (si.view(-1, 1, 1) ** 2))) * M
            Jcols.append(J_mu_i)
        if "mu_j" in cols:
            J_mu_j = -(A * G * (dj / (sj.view(-1, 1, 1) ** 2))) * M
            Jcols.append(J_mu_j)
        if "lsi" in cols:
            J_lsi = -(A * G * (di ** 2) / (si.view(-1, 1, 1) ** 2)) * M
            Jcols.append(J_lsi)
        if "lsj" in cols:
            J_lsj = -(A * G * (dj ** 2) / (sj.view(-1, 1, 1) ** 2)) * M
            Jcols.append(J_lsj)

        J_la = -(A * G) * M
        J_b = -torch.ones_like(R) * M
        Jcols.extend([J_la, J_b])

        # Shape to [N, HW, P]
        Jmat = torch.stack([c.reshape(N, -1) for c in Jcols], dim=-1)  # [N, HW, P]
        rvec = R.reshape(N, -1, 1)                                     # [N, HW, 1]

        JT = Jmat.transpose(1, 2)                                      # [N, P, HW]
        JTJ = torch.matmul(JT, Jmat)                                   # [N, P, P]
        JTr = torch.matmul(JT, rvec)                                   # [N, P, 1]

        # LM damping
        eye = torch.eye(JTJ.shape[-1], device=dev, dtype=P.dtype).unsqueeze(0).expand_as(JTJ)
        JTJ_damped = JTJ + lam.view(-1, 1, 1) * eye

        # Solve; fall back to pinv if needed
        try:
            delta = torch.linalg.solve(JTJ_damped, JTr).squeeze(-1)     # [N, P]
        except RuntimeError:
            delta = torch.matmul(torch.linalg.pinv(JTJ_damped), JTr).squeeze(-1)

        # Propose update
        k = 0
        mu_i_new, mu_j_new, lsi_new, lsj_new, la_new, b_new = mu_i, mu_j, lsi, lsj, la, b
        if "mu_i" in cols:
            mu_i_new = (mu_i + delta[:, k]).clamp(-1e6, 1e6)
            k += 1
        if "mu_j" in cols:
            mu_j_new = (mu_j + delta[:, k]).clamp(-1e6, 1e6)
            k += 1
        if "lsi" in cols:
            lsi_new = (lsi + delta[:, k]).clamp_min(LOG_MIN_SIGMA)
            k += 1
        if "lsj" in cols:
            lsj_new = (lsj + delta[:, k]).clamp_min(LOG_MIN_SIGMA)
            k += 1
        la_new = (la + delta[:, k]).clamp_min(LOG_MIN_AMP)
        k += 1
        b_new = b + delta[:, k]  # last

        # Amplitude upper clamp to avoid exp overflow
        la_new = torch.minimum(la_new, la_max)
        la_new = torch.maximum(la_new, la_min)

        # Evaluate new residual / accept or backtrack
        R_new, M_new, di, dj, si, sj, A, G = forward_and_residual(
            mu_i_new, mu_j_new, lsi_new, lsj_new, la_new, b_new
        )
        loss_new = torch.mean(R_new * R_new)

        bad = torch.isnan(loss_new) | torch.isinf(loss_new) | (loss_new > prev_loss)
        if bool(bad):
            lam = lam * 10.0
            continue
        else:
            # Accept
            mu_i, mu_j, lsi, lsj, la, b = (
                mu_i_new, mu_j_new, lsi_new, lsj_new, la_new, b_new
            )
            prev_loss = loss_new
            lam = torch.clamp(lam / 3.0, min=1e-6)

    # Cast back to original dtype
    out_dtype = patches.dtype
    return (
        mu_i.to(out_dtype),
        mu_j.to(out_dtype),
        lsi.exp().to(out_dtype),
        lsj.exp().to(out_dtype),
        la.exp().to(out_dtype),
        b.to(out_dtype),
    )


# ---------------------------
# Main detector (streaming)
# ---------------------------

def iter_detect_peaks_from_blurred(
    B_blurred: np.ndarray,
    *,
    device: str = "cuda",
    pool_scan: int = 15,
    pool_tof: int = 3,
    min_intensity: float = 75.0,            # interpreted in the SCALED domain
    tile_rows: int = 80_000,
    tile_overlap: int = 512,
    fit_h: int = 13,
    fit_w: int = 7,
    topk_per_tile: int | None = None,
    patch_batch_target_mb: int = 128,
    # refinement knobs
    refine: str = "none",               # "none" | "adam" | "gauss_newton" | "gn"
    refine_iters: int = 8,
    refine_lr: float = 0.2,
    refine_mask_k: float = 2.5,
    refine_scan: bool = True,
    refine_tof: bool = False,
    refine_sigma_scan: bool = True,
    refine_sigma_tof: bool = False,
    # scaling + numeric stability
    scale: str = "none",                # "none" | "sqrt" | "log1p"
    output_units: str = "scaled",       # "scaled" | "original"
    gn_float64: bool = False,           # leave False; sqrt scaling keeps fp32 stable
    blur_sigma_scan: float | None = None,
    blur_sigma_tof: float | None = None,
    blur_truncate: float = 3.0,
):
    """
    Stream peak stats on a 2D image derived from TOF×scan.

    Expected orientation of B_blurred (from TofScanWindowGrid.data):
      - rows: TOF bins (tof_row)
      - cols: global scans (scan_idx)

    Internally we transpose to (scan, tof) so that `mu_scan` is always along the
    0th axis and `mu_tof` along the 1st axis.

    Returned fields:
      - mu_scan,  mu_tof        (float, in ORIGINAL orientation)
      - sigma_scan, sigma_tof   (float, patch moments)
      - amplitude, baseline, area
      - tof_row, scan_idx       (indices in ORIGINAL orientation; kept as floats
                                 for compatibility with ImPeak1D.batch_from_detected)
    """
    assert B_blurred.ndim == 2

    # Original orientation: (tof_row, scan_idx) -> internal: (scan, tof_row)
    B_np_int = B_blurred.T  # (scan, tof)

    H, W = B_np_int.shape

    fit_h = _ensure_odd(max(fit_h, 2 * pool_scan + 5))
    fit_w = _ensure_odd(max(fit_w, 2 * pool_tof + 5))
    hr, wr = fit_h // 2, fit_w // 2

    # Torch tensors (UNSCALED initially)
    B_int = torch.from_numpy(B_np_int).to(device=device, dtype=torch.float32)

    # Optional Gaussian blur in scan/tof *before* scaling for detection
    if (blur_sigma_scan and blur_sigma_scan > 0) or (blur_sigma_tof and blur_sigma_tof > 0):
        B_int = _gaussian_blur_2d(
            B_int,
            sigma_scan=blur_sigma_scan or 0.0,
            sigma_tof=blur_sigma_tof or 0.0,
            truncate=blur_truncate,
        )

    # Keep a reference to the unscaled intensities if we need original outputs
    need_orig = (output_units == "original")
    B_int_unscaled = B_int if need_orig else None

    # Apply scaling for detection
    B_int, inv_scale = _apply_scale_tensor(B_int, scale)

    # pooling across scan (rows) x TOF (cols) on the **scaled** image
    kH, kW = pool_scan, pool_tof
    padH, padW = kH // 2, kW // 2

    # ----------------------------------------------------------------
    # Precompute spatial tensors used in _fit_moments / patch logic
    # ----------------------------------------------------------------
    # NOTE: we intentionally keep these in a "base" dtype/device and
    #       cast (.to) inside the helpers to match the patch dtype.
    ii_base = torch.arange(fit_h, dtype=torch.float32).view(1, fit_h, 1)
    jj_base = torch.arange(fit_w, dtype=torch.float32).view(1, 1, fit_w)

    ci = (fit_h - 1) / 2.0
    cj = (fit_w - 1) / 2.0
    wi_base = (ii_base - ci) / (0.6 * max(1.0, float(kH)))
    wj_base = (jj_base - cj) / (0.6 * max(1.0, float(kW)))
    Wsoft_base = torch.exp(-0.5 * (wi_base ** 2 + wj_base ** 2))

    off_i_base = torch.arange(fit_h, dtype=torch.long).view(1, fit_h, 1)
    off_j_base = torch.arange(fit_w, dtype=torch.long).view(1, 1, fit_w)

    def _extract_patches_pad(tile: torch.Tensor, peaks_ij: torch.Tensor) -> torch.Tensor:
        if peaks_ij.numel() == 0:
            return tile.new_zeros((0, fit_h, fit_w))
        h, w = tile.shape

        # pad tile once
        pad_tile = F.pad(tile[None, None], (wr, wr, hr, hr), mode="reflect")
        ph, pw = pad_tile.shape[-2:]

        # offsets/indices (on correct device/dtype)
        off_i = off_i_base.to(tile.device)
        off_j = off_j_base.to(tile.device)

        # patch centers in padded coords
        pi = peaks_ij[:, 0] + hr
        pj = peaks_ij[:, 1] + wr
        tl_i = pi - hr
        tl_j = pj - wr

        abs_i = (tl_i.view(-1, 1, 1) + off_i).clamp_(0, ph - 1)
        abs_j = (tl_j.view(-1, 1, 1) + off_j).clamp_(0, pw - 1)
        lin = abs_i * pw + abs_j

        flat = pad_tile.view(-1).contiguous()
        return flat.take(lin.view(-1)).view(-1, fit_h, fit_w)

    def _fit_moments(patches: torch.Tensor):
        if patches.numel() == 0:
            z = patches.new_zeros((0,))
            return z, z, z, z, z, z, z

        n = patches.shape[0]
        P = patches

        # robust baseline in the **current domain** (scaled)
        base = torch.quantile(P.view(n, -1), 0.10, dim=1)
        Y = (P - base.view(-1, 1, 1)).clamp_min_(0.0)

        # reuse precomputed spatial weights on correct device/dtype
        ii = ii_base.to(P.device, P.dtype)
        jj = jj_base.to(P.device, P.dtype)
        Wsoft = Wsoft_base.to(P.device, P.dtype)

        Yw = Y * Wsoft
        s = Yw.sum(dim=(1, 2)) + 1e-12

        mu_i = (Yw * ii).sum(dim=(1, 2)) / s
        mu_j = (Yw * jj).sum(dim=(1, 2)) / s

        var_i = (Yw * (ii - mu_i.view(-1, 1, 1)) ** 2).sum(dim=(1, 2)) / s
        var_j = (Yw * (jj - mu_j.view(-1, 1, 1)) ** 2).sum(dim=(1, 2)) / s
        sigma_i = var_i.clamp_min_(0).sqrt_()
        sigma_j = var_j.clamp_min_(0).sqrt_()

        i_n = mu_i.round().clamp_(0, fit_h - 1).to(torch.long)
        j_n = mu_j.round().clamp_(0, fit_w - 1).to(torch.long)
        amp = P[torch.arange(n, device=P.device), i_n, j_n] - base
        area = Y.sum(dim=(1, 2))

        return mu_i, mu_j, sigma_i, sigma_j, amp, base, area

    bytes_per_patch = fit_h * fit_w * 4  # float32
    target_bytes = max(16, int(patch_batch_target_mb)) * (1024 ** 2)
    patch_batch = max(512, min(1_000_000, target_bytes // max(1, bytes_per_patch)))

    i = 0
    while i < H:
        lo = max(0, i - tile_overlap)
        hi = min(H, i + tile_rows + tile_overlap)
        tile_scaled = B_int[lo:hi]  # scaled internal (scan, tof)

        thr = torch.tensor(float(min_intensity), dtype=tile_scaled.dtype, device=tile_scaled.device)
        pooled = F.max_pool2d(
            tile_scaled[None, None],
            kernel_size=(kH, kW),
            stride=1,
            padding=(padH, padW),
        )[0, 0]
        mask = (tile_scaled >= thr) & (tile_scaled == pooled)
        idxs = mask.nonzero(as_tuple=False)  # [N, 2] in (scan_row, tof_col) INTERNAL

        tile_orig = B_int_unscaled[lo:hi] if need_orig else None

        if idxs.numel() > 0:
            if topk_per_tile is not None and idxs.shape[0] > topk_per_tile:
                vals = tile_scaled[idxs[:, 0], idxs[:, 1]]
                topk = torch.topk(vals, k=topk_per_tile, largest=True)
                idxs = idxs[topk.indices]

            n = idxs.shape[0]
            for b0 in range(0, n, patch_batch):
                b1 = min(n, b0 + patch_batch)
                idxb = idxs[b0:b1]

                # patches in SCALED domain for fitting
                patches = _extract_patches_pad(tile_scaled, idxb)
                mu_i, mu_j, s_i, s_j, amp, base, area = _fit_moments(patches)

                # optional refinement
                if refine in ("adam",):
                    mu_i, mu_j, s_i, s_j, amp, base = _batched_refine_adam(
                        patches, mu_i, mu_j, s_i, s_j, amp, base,
                        iters=refine_iters, lr=refine_lr, mask_k=refine_mask_k,
                        refine_scan=refine_scan, refine_tof=refine_tof,
                        refine_sigma_scan=refine_sigma_scan,
                        refine_sigma_tof=refine_sigma_tof,
                    )
                elif refine in ("gauss_newton", "gn"):
                    mu_i, mu_j, s_i, s_j, amp, base = _batched_refine_gauss_newton(
                        patches, mu_i, mu_j, s_i, s_j, amp, base,
                        iters=max(1, refine_iters),
                        damping=1e-2,
                        refine_scan=refine_scan, refine_tof=refine_tof,
                        refine_sigma_scan=refine_sigma_scan,
                        refine_sigma_tof=refine_sigma_tof,
                        mask_k=refine_mask_k,
                        force_dtype=(torch.float64 if gn_float64 else None),
                    )

                # If caller wants original units, recompute amp/base/area on unscaled patches.
                if need_orig:
                    patches_o = _extract_patches_pad(tile_orig, idxb)
                    n_o = patches_o.shape[0]
                    base_o = torch.quantile(patches_o.view(n_o, -1), 0.10, dim=1)
                    i_n = mu_i.round().clamp_(0, fit_h - 1).to(torch.long)
                    j_n = mu_j.round().clamp_(0, fit_w - 1).to(torch.long)
                    amp_o = patches_o[
                        torch.arange(n_o, device=patches_o.device), i_n, j_n
                    ] - base_o
                    area_o = (
                        (patches_o - base_o.view(-1, 1, 1))
                        .clamp_min_(0.0)
                        .sum(dim=(1, 2))
                    )
                    amp, base, area = (
                        amp_o.to(amp.dtype),
                        base_o.to(base.dtype),
                        area_o.to(area.dtype),
                    )

                # INTERNAL absolute coords (scan,tof)
                abs_ij = idxb.clone()
                abs_ij[:, 0] += lo

                # convert patch-local μ to global INTERNAL coords
                mu_scan_int = abs_ij[:, 0].to(patches.dtype) + (mu_i - hr)
                mu_tof_int = abs_ij[:, 1].to(patches.dtype) + (mu_j - wr)

                # clamp to valid image bounds to be safe
                mu_scan_int = mu_scan_int.clamp(0, H - 1)
                mu_tof_int = mu_tof_int.clamp(0, W - 1)

                # Map back to ORIGINAL orientation (tof_row, scan_idx)
                mu_scan = mu_scan_int           # original "scan" (columns)
                mu_tof = mu_tof_int             # original "tof_row" (rows)

                tof_row_idx = abs_ij[:, 1].to(patches.dtype)   # original row index
                scan_idx = abs_ij[:, 0].to(patches.dtype)      # original column index

                out = torch.stack(
                    [
                        mu_scan,
                        mu_tof,
                        s_i,
                        s_j,
                        amp,
                        base,
                        area,
                        tof_row_idx,
                        scan_idx,
                    ],
                    dim=1,
                ).detach().cpu().numpy()

                yield {
                    "mu_scan": out[:, 0].astype(np.float32),
                    "mu_tof": out[:, 1].astype(np.float32),
                    "sigma_scan": out[:, 2].astype(np.float32),
                    "sigma_tof": out[:, 3].astype(np.float32),
                    "amplitude": out[:, 4].astype(np.float32),
                    "baseline": out[:, 5].astype(np.float32),
                    "area": out[:, 6].astype(np.float32),
                    "tof_row": out[:, 7].astype(np.float32),
                    "scan_idx": out[:, 8].astype(np.float32),
                }

                # free transients early; let PyTorch reuse cuda memory
                del patches, mu_i, mu_j, s_i, s_j, amp, base, area
                if need_orig:
                    del patches_o

        i += tile_rows
        # no empty_cache here; reuse allocator
        gc.collect()


# ---- plan -> batches of window groups ---------------------------------------

def iter_plan_batches(plan, batch_size: int):
    """
    Iterate over plan in chunks, using get_batch_par / get_batch if available.

    Compatible with the new TofScanPlan / TofScanPlanGroup wrappers:
      - get_batch_par(start, count) -> list[TofScanWindowGrid]
      - get_batch(start, count)     -> list[TofScanWindowGrid]
    """
    total = len(plan)
    for start in range(0, total, batch_size):
        count = min(batch_size, total - start)
        if hasattr(plan, "get_batch_par"):
            wgs = plan.get_batch_par(start, count)
        elif hasattr(plan, "get_batch"):
            wgs = plan.get_batch(start, count)
        else:
            # fallback: plain slicing
            wgs = [plan[i] for i in range(start, start + count)]
        yield wgs


# --- collect streaming chunks into dict-of-arrays -----------------------------

def _collect_stream(peaks_iter: Iterable[dict[str, np.ndarray]]):
    acc = {
        k: []
        for k in [
            "mu_scan",
            "mu_tof",
            "sigma_scan",
            "sigma_tof",
            "amplitude",
            "baseline",
            "area",
            "tof_row",
            "scan_idx",
        ]
    }
    for chunk in peaks_iter:
        for k, v in chunk.items():
            if k in acc:
                acc[k].append(v)
    return {
        k: (np.concatenate(vs, axis=0) if vs else np.empty((0,), np.float32))
        for k, vs in acc.items()
    }


# --- light dedup across tiles (keep max amplitude per coarse cell) ------------

def _dedup_peaks(peaks, tol_scan=0.75, tol_tof=0.25):
    if peaks["mu_scan"].size == 0:
        return peaks
    s = peaks["mu_scan"]
    t = peaks["mu_tof"]
    amp = peaks["amplitude"]

    g_s = np.floor(s / tol_scan).astype(np.int64)
    g_t = np.floor(t / tol_tof).astype(np.int64)
    key = (g_s << 32) ^ (g_t & 0xFFFFFFFF)

    # iterate from smallest -> largest amplitude, so last one kept per cell
    order = np.argsort(amp)
    seen: dict[int, int] = {}
    for idx in order:
        seen[key[idx]] = idx

    keep = np.fromiter(seen.values(), dtype=np.int64)
    # optional: impose RT-ish ordering (by mu_scan) for nicer downstream behaviour
    order2 = np.argsort(peaks["mu_scan"][keep])
    keep = keep[order2]

    return {k: v[keep] for k, v in peaks.items()}


# --- per-WG detection -> list[ImPeak1D] --------------------------------------

def _detect_im_peaks_for_wgs(
    wgs,
    plan_group,
    *,
    device="cuda",
    pool_scan=15,
    pool_tof=3,
    min_intensity_scaled=1.0,
    tile_rows=433_873,
    tile_overlap=64,
    fit_h=35,
    fit_w=11,
    refine="adam",
    refine_iters=8,
    refine_lr=0.2,
    refine_mask_k=2.5,
    refine_scan=True,
    refine_tof=True,
    refine_sigma_scan=True,
    refine_sigma_tof=True,
    scale="sqrt",
    output_units="original",
    gn_float64=False,
    do_dedup=True,
    tol_scan=0.75,
    tol_tof=0.25,
    k_sigma=2.0,                 # was 3.0, narrower window cap for IM
    min_width=3,
    blur_sigma_scan: float | None = None,
    blur_sigma_tof: float | None = None,
    blur_truncate: float = 3.0,
    topk_per_tile: int | None = None,
    patch_batch_target_mb: int = 128,
):
    from imspy.timstof.clustering.data import ImPeak1D

    batch_objs = []

    for wg in tqdm(wgs, desc="Detecting peaks (BATCH)", leave=False, ncols=80):
        # wg is a TofScanWindowGrid wrapper; .data moves the Rust buffer
        B = wg.data

        peaks_iter = iter_detect_peaks_from_blurred(
            B_blurred=B,
            device=device,
            pool_scan=pool_scan,
            pool_tof=pool_tof,
            min_intensity=min_intensity_scaled,
            tile_rows=tile_rows,
            tile_overlap=tile_overlap,
            fit_h=fit_h,
            fit_w=fit_w,
            refine=refine,
            refine_iters=refine_iters,
            refine_lr=refine_lr,
            refine_mask_k=refine_mask_k,
            refine_scan=refine_scan,
            refine_tof=refine_tof,
            refine_sigma_scan=refine_sigma_scan,
            refine_sigma_tof=refine_sigma_tof,
            scale=scale,
            output_units=output_units,
            gn_float64=gn_float64,
            blur_sigma_scan=blur_sigma_scan,
            blur_sigma_tof=blur_sigma_tof,
            blur_truncate=blur_truncate,
            topk_per_tile=topk_per_tile,
            patch_batch_target_mb=patch_batch_target_mb,
        )

        peaks = _collect_stream(peaks_iter)
        if do_dedup:
            peaks = _dedup_peaks(peaks, tol_scan=tol_scan, tol_tof=tol_tof)

        # IM geometry: typical FWHM in scan ≈ 40, but we don't want insane widths
        factor = 3.0
        expected_fwhm_scan = 55.0
        expected_fwhm_tof = 3.0

        max_sigma_scan = (expected_fwhm_scan / 2.355) * factor
        max_sigma_tof = (expected_fwhm_tof / 2.355) * factor

        sigma_scan = np.minimum(peaks["sigma_scan"], max_sigma_scan)
        sigma_tof = np.minimum(peaks["sigma_tof"], max_sigma_tof)
        peaks["sigma_scan"] = sigma_scan
        peaks["sigma_tof"] = sigma_tof

        objs = ImPeak1D.batch_from_detected(
            peaks,
            window_grid=wg,        # TofScanWindowGrid wrapper
            plan_group=plan_group,  # TofScanPlanGroup wrapper
            k_sigma=k_sigma,
            min_width=min_width,
        )
        batch_objs.extend(objs)

        # free WG buffers
        try:
            if hasattr(wg, "clear_cache") and callable(wg.clear_cache):
                wg.clear_cache()
        except Exception:
            pass

        del B, peaks, objs, peaks_iter, wg
        gc.collect()

    return batch_objs


# --- PUBLIC: iterate batches -> yield list[ImPeak1D] per batch ----------------

def iter_im_peaks_batches(
    plan,
    *,
    batch_size=64,
    device="cuda",
    # detector / refinement
    pool_scan=15,
    pool_tof=3,
    min_intensity_scaled=1.0,
    tile_rows=433_873,
    tile_overlap=64,
    fit_h=35,
    fit_w=11,
    refine="adam",
    refine_iters=8,
    refine_lr=0.2,
    refine_mask_k=2.5,
    refine_scan=True,
    refine_tof=True,
    refine_sigma_scan=True,
    refine_sigma_tof=True,
    # scaling + stability
    scale="sqrt",
    output_units="original",
    gn_float64=False,
    # dedup
    do_dedup=True,
    tol_scan=0.75,
    tol_tof=0.25,
    # conversion
    k_sigma=2.0,              # IM: match _detect_im_peaks_for_wgs default
    min_width=3,
    topk_per_tile: int | None = None,
    patch_batch_target_mb: int = 128,
    blur_sigma_scan: float | None = None,
    blur_sigma_tof: float | None = None,
    blur_truncate: float = 3.0,
):
    """
    Yields one list[ImPeak1D] per plan batch (e.g., 64 WGs).

    `plan` is expected to be a TofScanPlanGroup wrapper (or something with
    len(), get_batch_par/get_batch, and __getitem__ defined).
    """
    num_batches = (len(plan) + batch_size - 1) // batch_size
    for wgs in tqdm(
        iter_plan_batches(plan, batch_size),
        total=num_batches,
        desc="Batches",
        ncols=80,
    ):
        batch_objs = _detect_im_peaks_for_wgs(
            wgs,
            plan_group=plan,
            device=device,
            pool_scan=pool_scan,
            pool_tof=pool_tof,
            min_intensity_scaled=min_intensity_scaled,
            tile_rows=tile_rows,
            tile_overlap=tile_overlap,
            fit_h=fit_h,
            fit_w=fit_w,
            refine=refine,
            refine_iters=refine_iters,
            refine_lr=refine_lr,
            refine_mask_k=refine_mask_k,
            refine_scan=refine_scan,
            refine_tof=refine_tof,
            refine_sigma_scan=refine_sigma_scan,
            refine_sigma_tof=refine_sigma_tof,
            scale=scale,
            output_units=output_units,
            gn_float64=gn_float64,
            do_dedup=do_dedup,
            tol_scan=tol_scan,
            tol_tof=tol_tof,
            k_sigma=k_sigma,
            min_width=min_width,
            topk_per_tile=topk_per_tile,
            patch_batch_target_mb=patch_batch_target_mb,
            blur_sigma_scan=blur_sigma_scan,
            blur_sigma_tof=blur_sigma_tof,
            blur_truncate=blur_truncate,
        )

        yield batch_objs

        try:
            if hasattr(plan, "evict_views") and callable(plan.evict_views):
                plan.evict_views(wgs)
        except Exception:
            pass

        del wgs, batch_objs
        gc.collect()


# ---------------------------
# RT case: TOF×RT grids
# ---------------------------

def detect_rt_peaks_for_grid(
    grid: "TofRtGrid",
    *,
    device: str = "cuda",
    pool_rt: int = 5,                  # narrower LC peaks: ~3–5 frames
    pool_tof: int = 3,
    min_intensity_scaled: float = 1.0,
    tile_rows: int = 50_000,           # RT grids smaller; no need for 200k+
    tile_overlap: int = 32,
    fit_h: int = 19,                   # ~2*pool_rt + margin, odd
    fit_w: int = 11,
    refine: str = "adam",
    refine_iters: int = 8,
    refine_lr: float = 0.2,
    refine_mask_k: float = 2.5,
    refine_rt: bool = True,            # like refine_scan
    refine_tof: bool = True,
    refine_sigma_rt: bool = True,
    refine_sigma_tof: bool = True,
    scale: str = "sqrt",
    output_units: str = "original",
    gn_float64: bool = False,
    do_dedup: bool = True,
    tol_rt: float = 0.5,               # stricter dedup in RT than IM
    tol_tof: float = 0.25,
    k_sigma: float = 2.5,              # narrower window cap in frames
    min_width_frames: int = 3,
    blur_sigma_rt: float | None = None,
    blur_sigma_tof: float | None = None,
    blur_truncate: float = 3.0,
    topk_per_tile: int | None = None,
    patch_batch_target_mb: int = 128,
) -> List["RtPeak1D"]:
    """
    Detect RT 1D peaks from a single TofRtGrid using the same 2D detector
    as for TOF×SCAN, interpreting the column axis as RT frames.

    Here we assume narrow LC peaks in RT frames (~3–5 frames typical).
    """
    from imspy.timstof.clustering.data import RtPeak1D

    # Dense TOF×RT matrix (rows = TOF bins, cols = RT frames)
    B = grid.data  # np.ndarray, shape (tof_row, rt_frame)

    # Use the shared detector; blur happens inside via blur_sigma_* args.
    peaks_iter = iter_detect_peaks_from_blurred(
        B_blurred=B,
        device=device,
        pool_scan=pool_rt,              # "scan" axis is RT here
        pool_tof=pool_tof,
        min_intensity=min_intensity_scaled,
        tile_rows=tile_rows,
        tile_overlap=tile_overlap,
        fit_h=fit_h,
        fit_w=fit_w,
        refine=refine,
        refine_iters=refine_iters,
        refine_lr=refine_lr,
        refine_mask_k=refine_mask_k,
        refine_scan=refine_rt,
        refine_tof=refine_tof,
        refine_sigma_scan=refine_sigma_rt,
        refine_sigma_tof=refine_sigma_tof,
        scale=scale,
        output_units=output_units,
        gn_float64=gn_float64,
        blur_sigma_scan=blur_sigma_rt,
        blur_sigma_tof=blur_sigma_tof,
        blur_truncate=blur_truncate,
        topk_per_tile=topk_per_tile,
        patch_batch_target_mb=patch_batch_target_mb,
    )

    peaks = _collect_stream(peaks_iter)

    if peaks["mu_scan"].size == 0:
        # no detections → empty list
        return []

    if do_dedup:
        # Same logic as IM: "scan" is RT frames here.
        peaks = _dedup_peaks(
            peaks,
            tol_scan=tol_rt,
            tol_tof=tol_tof,
        )

    # Optional sigma caps – now for RT instead of IM.
    # Typical LC FWHM ~4 frames; allow up to ~2×FWHM in window.
    factor = 3.0
    expected_fwhm_rt = 3.0
    max_sigma_rt = (expected_fwhm_rt / 2.355) * factor
    expected_fwhm_tof = 3.0  # tweak if needed
    max_sigma_tof = (expected_fwhm_tof / 2.355) * factor

    peaks["sigma_scan"] = np.minimum(peaks["sigma_scan"], max_sigma_rt)
    peaks["sigma_tof"] = np.minimum(peaks["sigma_tof"], max_sigma_tof)

    # Turn dict-of-arrays into RtPeak1D objects
    objs = RtPeak1D.from_batch_detected(
        peaks,
        window_grid=grid,    # TofRtGrid wrapper
        plan_group=None,
        k_sigma=k_sigma,
        min_width=min_width_frames,
    )

    # free grid buffers if it has a cache (mirrors IM code)
    try:
        if hasattr(grid, "clear_cache") and callable(grid.clear_cache):
            grid.clear_cache()
    except Exception:
        pass

    del B, peaks, peaks_iter
    gc.collect()

    return objs