import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------
# Small utilities
# ---------------------------
def _ensure_odd(x: int) -> int:
    return int(x) if (x % 2 == 1) else (int(x) + 1)

def _apply_scale_tensor(X: torch.Tensor, scale: str):
    """
    Returns (X_scaled, inv_fn). inv_fn maps scaled -> original.
    """
    if scale in (None, "none"):
        return X, (lambda y: y)
    if scale == "sqrt":
        # classic variance stabilization for Poisson-ish intensities
        Xs = torch.sqrt(torch.clamp(X, min=0.0))
        return Xs, (lambda y: torch.clamp(y, min=0.0)**2)
    if scale == "cbrt":
        Xs = torch.cbrt(torch.clamp(X, min=0.0))
        return Xs, (lambda y: torch.clamp(y, min=0.0)**3)
    if scale == "log1p":
        Xs = torch.log1p(torch.clamp(X, min=0.0))
        return Xs, (lambda y: torch.expm1(y))
    raise ValueError(f"unknown scale={scale!r}")

# ---------------------------
# Batched refiners
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
    refine_mz: bool = False,
    refine_sigma_scan: bool = True,
    refine_sigma_mz: bool = False,
):
    if patches.numel() == 0:
        z = patches.new_zeros((0,))
        return z, z, z, z, z, z

    N, H, W = patches.shape
    dev, dtype = patches.device, patches.dtype
    I = torch.arange(H, device=dev, dtype=dtype).view(1, H, 1).expand(N, H, 1)
    J = torch.arange(W, device=dev, dtype=dtype).view(1, 1, W).expand(N, 1, W)

    P = patches.detach()  # fit to a stopped-copy

    with torch.enable_grad():
        mu_i = mu_i0.detach().to(dev).clone(); mu_i.requires_grad_(refine_scan)
        mu_j = mu_j0.detach().to(dev).clone(); mu_j.requires_grad_(refine_mz)
        lsi  = sigma_i0.clamp_min(1e-3).log().detach().to(dev).clone()
        lsj  = sigma_j0.clamp_min(1e-3).log().detach().to(dev).clone()
        lsi.requires_grad_(refine_scan and refine_sigma_scan)
        lsj.requires_grad_(refine_mz   and refine_sigma_mz)
        la   = amp0.clamp_min(1e-6).log().detach().to(dev).clone(); la.requires_grad_(True)
        b    = base0.detach().to(dev).clone(); b.requires_grad_(True)

        params = [p for p in (mu_i, mu_j, lsi, lsj, la, b) if p.requires_grad]
        if not params:
            return mu_i.detach(), mu_j.detach(), lsi.exp().detach(), lsj.exp().detach(), la.exp().detach(), b.detach()

        opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

        def loss_fn(yhat, y, delta=1.0):
            r = yhat - y
            if loss == "huber":
                absr = r.abs()
                return torch.where(absr <= delta, 0.5*r*r, delta*(absr - 0.5*delta)).mean()
            elif loss == "cauchy":
                return (delta**2 * torch.log1p((r/delta)**2)).mean()
            else:
                return (r*r).mean()

        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            si = lsi.exp(); sj = lsj.exp()
            di = I - mu_i.view(-1,1,1)
            dj = J - mu_j.view(-1,1,1)

            # detached mask narrows the influence region
            Mi = torch.exp(-0.5*(di/(mask_k*si.view(-1,1,1)))**2)
            Mj = torch.exp(-0.5*(dj/(mask_k*sj.view(-1,1,1)))**2)
            M = (Mi*Mj).detach()

            q = (di/si.view(-1,1,1))**2 + (dj/sj.view(-1,1,1))**2
            yhat = b.view(-1,1,1) + la.exp().view(-1,1,1) * torch.exp(-0.5*q)
            L = loss_fn(yhat*M, P*M)
            L.backward()
            opt.step()

            # guards on logs
            lsi.data.clamp_(min=np.log(1e-3))
            lsj.data.clamp_(min=np.log(1e-3))
            la.data.clamp_(min=np.log(1e-6))

    return mu_i.detach(), mu_j.detach(), lsi.exp().detach(), lsj.exp().detach(), la.exp().detach(), b.detach()


def _batched_refine_gauss_newton(
    patches: torch.Tensor,
    mu_i0: torch.Tensor, mu_j0: torch.Tensor,
    sigma_i0: torch.Tensor, sigma_j0: torch.Tensor,
    amp0: torch.Tensor, base0: torch.Tensor,
    *,
    iters: int = 5,
    damping: float = 1e-2,          # slightly stronger default
    refine_scan: bool = True,
    refine_mz: bool = False,
    refine_sigma_scan: bool = True,
    refine_sigma_mz: bool = False,
    mask_k: float = 2.5,            # NEW: match ADAM’s mask behavior
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
    lsi  = sigma_i0.clamp_min(1e-3).log().to(dev, dtype=P.dtype).clone()
    lsj  = sigma_j0.clamp_min(1e-3).log().to(dev, dtype=P.dtype).clone()
    la   = amp0.clamp_min(1e-6).log().to(dev, dtype=P.dtype).clone()
    b    = base0.to(dev, dtype=P.dtype).clone()

    # Which parameters to include
    cols = []
    if refine_scan: cols.append("mu_i")
    if refine_mz:   cols.append("mu_j")
    if refine_scan and refine_sigma_scan: cols.append("lsi")
    if refine_mz   and refine_sigma_mz:   cols.append("lsj")
    cols += ["la", "b"]

    # Per-batch dynamic clamps for amplitude (prevents exp overflow)
    # Use an empirical cap: log(max( patch_max - base, 1e-3 )) + margin
    patch_max = torch.amax(P.view(N, -1), dim=1)
    amp_cap   = (patch_max - b).clamp_min(1e-3)
    la_max    = torch.log(amp_cap) + 2.0   # small safety margin
    la_min    = torch.log(torch.full_like(la_max, 1e-6))

    # Initial residual (for LM acceptance test)
    def forward_and_residual(mu_i, mu_j, lsi, lsj, la, b):
        si = lsi.exp(); sj = lsj.exp()
        di = I - mu_i.view(-1,1,1)
        dj = J - mu_j.view(-1,1,1)

        # Detached Gaussian mask like ADAM to localize influence
        Mi = torch.exp(-0.5*(di/(mask_k*si.view(-1,1,1)))**2)
        Mj = torch.exp(-0.5*(dj/(mask_k*sj.view(-1,1,1)))**2)
        M  = (Mi*Mj).detach()

        q    = (di/si.view(-1,1,1))**2 + (dj/sj.view(-1,1,1))**2
        A    = la.exp().view(-1,1,1)
        G    = torch.exp(-0.5*q)
        yhat = b.view(-1,1,1) + A*G
        R    = (P - yhat) * M
        return R, M, di, dj, si, sj, A, G

    R, M, di, dj, si, sj, A, G = forward_and_residual(mu_i, mu_j, lsi, lsj, la, b)
    prev_loss = torch.mean(R*R)

    lam = torch.as_tensor(damping, device=dev, dtype=P.dtype)

    for _ in range(iters):
        # Partials for enabled params (Jacobian of residuals: J = dR/dθ = -dyhat/dθ) with mask M applied
        Jcols = []
        if "mu_i" in cols:
            J_mu_i = -(A*G * (di/(si.view(-1,1,1)**2))) * M
            Jcols.append(J_mu_i)
        if "mu_j" in cols:
            J_mu_j = -(A*G * (dj/(sj.view(-1,1,1)**2))) * M
            Jcols.append(J_mu_j)
        if "lsi" in cols:
            J_lsi  = -(A*G * (di**2) / (si.view(-1,1,1)**2)) * M
            Jcols.append(J_lsi)
        if "lsj" in cols:
            J_lsj  = -(A*G * (dj**2) / (sj.view(-1,1,1)**2)) * M
            Jcols.append(J_lsj)
        J_la = -(A*G) * M
        J_b  = -(torch.ones_like(R)) * M
        Jcols.extend([J_la, J_b])

        # Shape to [N, HW, P]
        Jmat = torch.stack([c.reshape(N, -1) for c in Jcols], dim=-1)  # [N, HW, P]
        rvec = R.reshape(N, -1, 1)                                     # [N, HW, 1]

        JT  = Jmat.transpose(1, 2)                                     # [N, P, HW]
        JTJ = torch.matmul(JT, Jmat)                                    # [N, P, P]
        JTr = torch.matmul(JT, rvec)                                    # [N, P, 1]

        # LM damping
        eye = torch.eye(JTJ.shape[-1], device=dev, dtype=P.dtype).unsqueeze(0).expand_as(JTJ)
        JTJ_damped = JTJ + lam.view(-1,1,1) * eye

        # Solve; fall back to pinv if needed
        try:
            delta = torch.linalg.solve(JTJ_damped, JTr).squeeze(-1)     # [N, P]
        except RuntimeError:
            delta = torch.matmul(torch.linalg.pinv(JTJ_damped), JTr).squeeze(-1)

        # Propose update
        k = 0
        mu_i_new, mu_j_new, lsi_new, lsj_new, la_new, b_new = mu_i, mu_j, lsi, lsj, la, b
        if "mu_i" in cols:
            mu_i_new = (mu_i + delta[:, k]).clamp(-1e6, 1e6); k += 1
        if "mu_j" in cols:
            mu_j_new = (mu_j + delta[:, k]).clamp(-1e6, 1e6); k += 1
        if "lsi" in cols:
            lsi_new  = (lsi  + delta[:, k]).clamp_min(np.log(1e-3));   k += 1
        if "lsj" in cols:
            lsj_new  = (lsj  + delta[:, k]).clamp_min(np.log(1e-3));   k += 1
        la_new  = (la + delta[:, k]).clamp_min(np.log(1e-6));           k += 1
        b_new   =  b + delta[:, k]                                     # last

        # Amplitude upper clamp to avoid exp overflow
        la_new = torch.minimum(la_new, la_max)
        la_new = torch.maximum(la_new, la_min)

        # Evaluate new residual / accept or backtrack
        R_new, M_new, di, dj, si, sj, A, G = forward_and_residual(
            mu_i_new, mu_j_new, lsi_new, lsj_new, la_new, b_new
        )
        loss_new = torch.mean(R_new*R_new)

        # If NaN/Inf or not improved, increase damping and retry this iteration
        bad = torch.isnan(loss_new) | torch.isinf(loss_new) | (loss_new > prev_loss)
        if bool(bad):
            lam = lam * 10.0
            # Recompute with stronger damping next loop (no state change)
            continue
        else:
            # Accept
            mu_i, mu_j, lsi, lsj, la, b = mu_i_new, mu_j_new, lsi_new, lsj_new, la_new, b_new
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
    pool_mz: int = 3,
    min_intensity: float = 75.0,            # interpreted in the SCALED domain
    tile_rows: int = 80_000,
    tile_overlap: int = 512,
    fit_h: int = 13,
    fit_w: int = 7,
    topk_per_tile: int | None = None,
    patch_batch_target_mb: int = 128,
    rows_are_mz: bool = True,
    # refinement knobs
    refine: str = "none",               # "none" | "adam" | "gauss_newton" | "gn"
    refine_iters: int = 8,
    refine_lr: float = 0.2,
    refine_mask_k: float = 2.5,
    refine_scan: bool = True,
    refine_mz: bool = False,
    refine_sigma_scan: bool = True,
    refine_sigma_mz: bool = False,
    # --- NEW ---
    scale: str = "none",                # "none" | "sqrt" | "log1p"
    output_units: str = "scaled",       # "scaled" | "original"
    gn_float64: bool = False,           # leave False; sqrt scaling keeps fp32 stable
):
    """
    Stream peak stats. If rows_are_mz=True, input is (mz, scan) and will be
    transposed internally so the algorithm always works on (scan, mz).
    Returned fields (mu_scan, mu_mz, i, j) are in the ORIGINAL orientation.
    Amplitude/baseline/area are in `output_units` (scaled or original).
    """
    assert B_blurred.ndim == 2
    B_np_int = B_blurred.T if rows_are_mz else B_blurred  # internal: (scan, mz)

    H, W = B_np_int.shape
    fit_h = _ensure_odd(max(fit_h, 2*pool_scan + 5))
    fit_w = _ensure_odd(max(fit_w, 2*pool_mz   + 5))
    hr, wr = fit_h // 2, fit_w // 2

    # Torch tensors
    B_int = torch.from_numpy(B_np_int).to(device=device, dtype=torch.float32)

    # Keep a reference to the unscaled intensities if we need original outputs
    need_orig = (output_units == "original")
    B_int_unscaled = B_int if need_orig else None

    # Apply scaling for detection
    B_int, inv_scale = _apply_scale_tensor(B_int, scale)

    # pooling across scan (rows) x m/z (cols) on the **scaled** image
    kH, kW = pool_scan, pool_mz
    padH, padW = kH // 2, kW // 2

    def _extract_patches_pad(tile: torch.Tensor, peaks_ij: torch.Tensor) -> torch.Tensor:
        if peaks_ij.numel() == 0:
            return tile.new_zeros((0, fit_h, fit_w))
        h, w = tile.shape
        pad_tile = F.pad(tile[None, None], (wr, wr, hr, hr), mode="reflect")
        ph, pw = pad_tile.shape[-2:]
        pi = peaks_ij[:, 0] + hr
        pj = peaks_ij[:, 1] + wr
        tl_i = pi - hr
        tl_j = pj - wr
        off_i = torch.arange(fit_h, device=tile.device, dtype=torch.long).view(1, fit_h, 1)
        off_j = torch.arange(fit_w, device=tile.device, dtype=torch.long).view(1, 1, fit_w)
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
        ii = torch.arange(fit_h, device=P.device, dtype=P.dtype).view(1, fit_h, 1)
        jj = torch.arange(fit_w, device=P.device, dtype=P.dtype).view(1, 1, fit_w)
        ci = (fit_h - 1) / 2.0
        cj = (fit_w - 1) / 2.0
        wi = (ii - ci) / (0.6 * max(1.0, float(kH)))
        wj = (jj - cj) / (0.6 * max(1.0, float(kW)))
        Wsoft = torch.exp(-0.5 * (wi**2 + wj**2))
        Yw = Y * Wsoft
        s   = Yw.sum(dim=(1, 2)) + 1e-12
        mu_i = (Yw * ii).sum(dim=(1, 2)) / s
        mu_j = (Yw * jj).sum(dim=(1, 2)) / s
        var_i = (Yw * (ii - mu_i.view(-1, 1, 1))**2).sum(dim=(1, 2)) / s
        var_j = (Yw * (jj - mu_j.view(-1, 1, 1))**2).sum(dim=(1, 2)) / s
        sigma_i = var_i.clamp_min_(0).sqrt_()
        sigma_j = var_j.clamp_min_(0).sqrt_()
        i_n = mu_i.round().clamp_(0, fit_h - 1).to(torch.long)
        j_n = mu_j.round().clamp_(0, fit_w - 1).to(torch.long)
        amp = P[torch.arange(n, device=P.device), i_n, j_n] - base
        area = (Y).sum(dim=(1,2))
        return mu_i, mu_j, sigma_i, sigma_j, amp, base, area

    bytes_per_patch = fit_h * fit_w * 4
    target_bytes = max(16, int(patch_batch_target_mb)) * (1024**2)
    patch_batch = max(512, min(1_000_000, target_bytes // max(1, bytes_per_patch)))

    i = 0
    while i < H:
        lo = max(0, i - tile_overlap)
        hi = min(H, i + tile_rows + tile_overlap)
        tile_scaled = B_int[lo:hi]  # scaled internal (scan,mz)

        thr = torch.tensor(float(min_intensity), dtype=tile_scaled.dtype, device=tile_scaled.device)
        pooled = F.max_pool2d(tile_scaled[None, None], kernel_size=(kH, kW),
                              stride=1, padding=(padH, padW))[0, 0]
        mask = (tile_scaled >= thr) & (tile_scaled == pooled)
        idxs = mask.nonzero(as_tuple=False)  # [N, 2] in (scan_row, mz_col)

        # optional access to original (unscaled) tile
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
                        refine_scan=refine_scan, refine_mz=refine_mz,
                        refine_sigma_scan=refine_sigma_scan, refine_sigma_mz=refine_sigma_mz,
                    )
                elif refine in ("gauss_newton", "gn"):
                    mu_i, mu_j, s_i, s_j, amp, base = _batched_refine_gauss_newton(
                        patches, mu_i, mu_j, s_i, s_j, amp, base,
                        iters=max(1, refine_iters),
                        damping=1e-2,
                        refine_scan=refine_scan, refine_mz=refine_mz,
                        refine_sigma_scan=refine_sigma_scan, refine_sigma_mz=refine_sigma_mz,
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
                    amp_o = patches_o[torch.arange(n_o, device=patches_o.device), i_n, j_n] - base_o
                    area_o = (patches_o - base_o.view(-1,1,1)).clamp_min_(0.0).sum(dim=(1,2))
                    amp, base, area = amp_o.to(amp.dtype), base_o.to(base.dtype), area_o.to(area.dtype)

                # INTERNAL absolute coords (scan,mz)
                abs_ij = idxb.clone()
                abs_ij[:, 0] += lo

                # convert patch-local μ to global INTERNAL coords
                mu_scan_int = abs_ij[:, 0].to(patches.dtype) + (mu_i - hr)
                mu_mz_int   = abs_ij[:, 1].to(patches.dtype) + (mu_j - wr)

                # Map back to ORIGINAL orientation
                if rows_are_mz:
                    mu_scan = mu_scan_int
                    mu_mz   = mu_mz_int
                    i_idx   = abs_ij[:, 1].to(patches.dtype)  # integer mz row (orig)
                    j_idx   = abs_ij[:, 0].to(patches.dtype)  # integer scan col (orig)
                else:
                    mu_scan = mu_scan_int
                    mu_mz   = mu_mz_int
                    i_idx   = abs_ij[:, 0].to(patches.dtype)
                    j_idx   = abs_ij[:, 1].to(patches.dtype)

                out = torch.stack([
                    mu_scan, mu_mz, s_i, s_j, amp, base, area, i_idx, j_idx
                ], dim=1).detach().cpu().numpy()

                yield {
                    "mu_scan":   out[:, 0].astype(np.float32),
                    "mu_mz":     out[:, 1].astype(np.float32),
                    "sigma_scan":out[:, 2].astype(np.float32),
                    "sigma_mz":  out[:, 3].astype(np.float32),
                    "amplitude": out[:, 4].astype(np.float32),  # scaled or original per output_units
                    "baseline":  out[:, 5].astype(np.float32),
                    "area":      out[:, 6].astype(np.float32),
                    "i":         out[:, 7].astype(np.float32),
                    "j":         out[:, 8].astype(np.float32),
                }

                # free transients early
                del patches, mu_i, mu_j, s_i, s_j, amp, base, area
                if need_orig:
                    del patches_o
                if tile_scaled.is_cuda:
                    torch.cuda.empty_cache()

        i += tile_rows
        if B_int.is_cuda:
            torch.cuda.empty_cache()


def detect_peaks_from_blurred_streaming(**kwargs):
    acc = {k: [] for k in ["mu_scan","mu_mz","sigma_scan","sigma_mz",
                           "amplitude","baseline","area","i","j"]}
    for chunk in iter_detect_peaks_from_blurred(**kwargs):
        for k, v in chunk.items():
            acc[k].append(v)
    return {k: (np.concatenate(v, axis=0) if len(v) else np.empty((0,), np.float32))
            for k, v in acc.items()}