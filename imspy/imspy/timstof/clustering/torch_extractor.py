import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------
# Small utilities
# ---------------------------
def _ensure_odd(x: int) -> int:
    return int(x) if (x % 2 == 1) else (int(x) + 1)

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
    damping: float = 1e-3,
    refine_scan: bool = True,
    refine_mz: bool = False,
    refine_sigma_scan: bool = True,
    refine_sigma_mz: bool = False,
):
    """
    Gauss–Newton on diagonal Σ (no correlation). θ subset depends on flags.
    Enabled columns (order): [mu_i?, mu_j?, lsi?, lsj?, la, b]
    """
    if patches.numel() == 0:
        z = patches.new_zeros((0,))
        return z, z, z, z, z, z

    N, H, W = patches.shape
    dev, dtype = patches.device, patches.dtype
    I = torch.arange(H, device=dev, dtype=dtype).view(1, H, 1).expand(N, H, 1)
    J = torch.arange(W, device=dev, dtype=dtype).view(1, 1, W).expand(N, 1, W)

    mu_i = mu_i0.clone().to(dev)
    mu_j = mu_j0.clone().to(dev)
    lsi  = sigma_i0.clamp_min(1e-3).log().clone().to(dev)
    lsj  = sigma_j0.clamp_min(1e-3).log().clone().to(dev)
    la   = amp0.clamp_min(1e-6).log().clone().to(dev)
    b    = base0.clone().to(dev)

    # which columns to include
    cols = []
    if refine_scan:             cols.append("mu_i")
    if refine_mz:               cols.append("mu_j")
    if refine_scan and refine_sigma_scan: cols.append("lsi")
    if refine_mz   and refine_sigma_mz:   cols.append("lsj")
    cols += ["la", "b"]  # always fit amplitude and baseline
    P = patches

    for _ in range(iters):
        si = lsi.exp(); sj = lsj.exp()
        di = I - mu_i.view(-1,1,1)
        dj = J - mu_j.view(-1,1,1)

        q   = (di/si.view(-1,1,1))**2 + (dj/sj.view(-1,1,1))**2
        A   = la.exp().view(-1,1,1)
        G   = torch.exp(-0.5*q)
        yhat= b.view(-1,1,1) + A*G
        R   = (P - yhat)  # residuals

        # partials for enabled params
        Jcols = []
        if "mu_i" in cols:
            J_mu_i = -(A*G * (di/(si.view(-1,1,1)**2)))
            Jcols.append(J_mu_i)
        if "mu_j" in cols:
            J_mu_j = -(A*G * (dj/(sj.view(-1,1,1)**2)))
            Jcols.append(J_mu_j)
        if "lsi" in cols:
            J_lsi = -(A*G * (di**2) / (si.view(-1,1,1)**2))
            Jcols.append(J_lsi)
        if "lsj" in cols:
            J_lsj = -(A*G * (dj**2) / (sj.view(-1,1,1)**2))
            Jcols.append(J_lsj)
        # la (logA) and b always included
        J_la = -(A*G)
        J_b  = -torch.ones_like(R)
        Jcols.extend([J_la, J_b])

        # shape to [N, HW, P]
        Jmat = torch.stack([c.reshape(N, -1) for c in Jcols], dim=-1)  # [N, HW, P]
        rvec = R.reshape(N, -1, 1)                                     # [N, HW, 1]

        JTJ = torch.matmul(Jmat.transpose(1,2), Jmat)                  # [N, P, P]
        JTr = torch.matmul(Jmat.transpose(1,2), rvec)                  # [N, P, 1]

        eye = torch.eye(JTJ.shape[-1], device=dev, dtype=dtype).unsqueeze(0).expand_as(JTJ)
        JTJ = JTJ + damping * eye

        try:
            delta = torch.linalg.solve(JTJ, JTr).squeeze(-1)           # [N, P]
        except RuntimeError:
            delta = torch.matmul(torch.linalg.pinv(JTJ), JTr).squeeze(-1)

        # scatter updates back to full parameter set in the order of 'cols'
        k = 0
        if "mu_i" in cols:
            mu_i = (mu_i + delta[:, k]).clamp(-1e6, 1e6); k += 1
        if "mu_j" in cols:
            mu_j = (mu_j + delta[:, k]).clamp(-1e6, 1e6); k += 1
        if "lsi" in cols:
            lsi  = (lsi  + delta[:, k]).clamp_min(np.log(1e-3)); k += 1
        if "lsj" in cols:
            lsj  = (lsj  + delta[:, k]).clamp_min(np.log(1e-3)); k += 1
        la   = (la   + delta[:, k]).clamp_min(np.log(1e-6)); k += 1
        b    =  b    + delta[:, k]                                   # last

    return mu_i, mu_j, lsi.exp(), lsj.exp(), la.exp(), b


# ---------------------------
# Main detector (streaming)
# ---------------------------

def iter_detect_peaks_from_blurred(
    B_blurred: np.ndarray,
    *,
    device: str = "cuda",
    pool_scan: int = 15,
    pool_mz: int = 3,
    min_intensity: float = 75.0,
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
    refine_scan: bool = True,           # refine μ/σ in scan (rows)
    refine_mz: bool = False,            # refine μ/σ in m/z (cols)  ← default off
    refine_sigma_scan: bool = True,     # allow σ_scan to move
    refine_sigma_mz: bool = False,      # keep σ_mz from moments
):
    """
    Stream peak stats. If rows_are_mz=True, input is (mz, scan) and will be
    transposed internally so the algorithm always works on (scan, mz).
    Returned fields (mu_scan, mu_mz, i, j) are in the ORIGINAL orientation.
    """
    assert B_blurred.ndim == 2
    B_np_int = B_blurred.T if rows_are_mz else B_blurred  # internal: (scan, mz)

    H, W = B_np_int.shape
    fit_h = _ensure_odd(max(fit_h, 2*pool_scan + 5))   # reduce σ bias
    fit_w = _ensure_odd(max(fit_w, 2*pool_mz   + 5))
    hr, wr = fit_h // 2, fit_w // 2

    B = torch.from_numpy(B_np_int).to(device=device, dtype=torch.float32)

    # pooling across scan (rows) x m/z (cols)
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
        """
        Moment init with soft elliptical window to reduce σ_scan underestimation.
        Returns (mu_i, mu_j, sigma_i, sigma_j, amp, base, area) in patch coords.
        """
        if patches.numel() == 0:
            z = patches.new_zeros((0,))
            return z, z, z, z, z, z, z
        n = patches.shape[0]
        P = patches

        # robust local baseline
        base = torch.quantile(P.view(n, -1), 0.10, dim=1)

        Y = (P - base.view(-1, 1, 1)).clamp_min_(0.0)
        ii = torch.arange(fit_h, device=P.device, dtype=P.dtype).view(1, fit_h, 1)
        jj = torch.arange(fit_w, device=P.device, dtype=P.dtype).view(1, 1, fit_w)

        # soft elliptical weight around patch center (not μ)
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

        area = (Y).sum(dim=(1,2))  # positive mass (unweighted) as proxy
        return mu_i, mu_j, sigma_i, sigma_j, amp, base, area

    # choose micro-batch size for patches from a target MB
    bytes_per_patch = fit_h * fit_w * 4
    target_bytes = max(16, int(patch_batch_target_mb)) * (1024**2)
    patch_batch = max(512, min(1_000_000, target_bytes // max(1, bytes_per_patch)))

    i = 0
    while i < H:
        lo = max(0, i - tile_overlap)
        hi = min(H, i + tile_rows + tile_overlap)
        tile = B[lo:hi]  # [h, W] internal (scan,mz)

        thr = torch.tensor(float(min_intensity), dtype=tile.dtype, device=tile.device)
        pooled = F.max_pool2d(tile[None, None], kernel_size=(kH, kW),
                              stride=1, padding=(padH, padW))[0, 0]
        mask = (tile >= thr) & (tile == pooled)
        idxs = mask.nonzero(as_tuple=False)  # [N, 2] internal coords (scan_row, mz_col)

        if idxs.numel() > 0:
            if topk_per_tile is not None and idxs.shape[0] > topk_per_tile:
                vals = tile[idxs[:, 0], idxs[:, 1]]
                topk = torch.topk(vals, k=topk_per_tile, largest=True)
                idxs = idxs[topk.indices]

            n = idxs.shape[0]
            for b0 in range(0, n, patch_batch):
                b1 = min(n, b0 + patch_batch)
                idxb = idxs[b0:b1]

                patches = _extract_patches_pad(tile, idxb)
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
                        iters=max(1, refine_iters), damping=1e-3,
                        refine_scan=refine_scan, refine_mz=refine_mz,
                        refine_sigma_scan=refine_sigma_scan, refine_sigma_mz=refine_sigma_mz,
                    )

                # INTERNAL absolute coords (scan,mz)
                abs_ij = idxb.clone()
                abs_ij[:, 0] += lo

                # convert patch-local μ to global INTERNAL coords
                mu_scan_int = abs_ij[:, 0].to(patches.dtype) + (mu_i - hr)
                mu_mz_int   = abs_ij[:, 1].to(patches.dtype) + (mu_j - wr)

                # Map back to ORIGINAL orientation
                if rows_are_mz:
                    # original: rows=mz, cols=scan
                    mu_scan = mu_scan_int                 # → original COL
                    mu_mz   = mu_mz_int                   # → original ROW
                    i_idx   = abs_ij[:, 1].to(patches.dtype)  # integer mz row (orig)
                    j_idx   = abs_ij[:, 0].to(patches.dtype)  # integer scan col (orig)
                else:
                    mu_scan = mu_scan_int
                    mu_mz   = mu_mz_int
                    i_idx   = abs_ij[:, 0].to(patches.dtype)
                    j_idx   = abs_ij[:, 1].to(patches.dtype)

                out = torch.stack([
                    mu_scan, mu_mz, s_i, s_j, amp, base, area,
                    i_idx, j_idx
                ], dim=1).detach().cpu().numpy()

                yield {
                    "mu_scan":   out[:, 0].astype(np.float32),
                    "mu_mz":     out[:, 1].astype(np.float32),
                    "sigma_scan":out[:, 2].astype(np.float32),
                    "sigma_mz":  out[:, 3].astype(np.float32),
                    "amplitude": out[:, 4].astype(np.float32),
                    "baseline":  out[:, 5].astype(np.float32),
                    "area":      out[:, 6].astype(np.float32),
                    "i":         out[:, 7].astype(np.float32),  # integer ROW in ORIGINAL
                    "j":         out[:, 8].astype(np.float32),  # integer COL in ORIGINAL
                }

                # free transients early
                del patches, mu_i, mu_j, s_i, s_j, amp, base, area, abs_ij, out
                if isinstance(tile, torch.Tensor) and tile.is_cuda:
                    torch.cuda.empty_cache()

        i += tile_rows
        if isinstance(B, torch.Tensor) and B.is_cuda:
            torch.cuda.empty_cache()


def detect_peaks_from_blurred_streaming(**kwargs):
    acc = {k: [] for k in ["mu_scan","mu_mz","sigma_scan","sigma_mz",
                           "amplitude","baseline","area","i","j"]}
    for chunk in iter_detect_peaks_from_blurred(**kwargs):
        for k, v in chunk.items():
            acc[k].append(v)
    return {k: (np.concatenate(v, axis=0) if len(v) else np.empty((0,), np.float32))
            for k, v in acc.items()}