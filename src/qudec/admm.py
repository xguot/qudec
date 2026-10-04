"""Batched ADMM decoding following Gu and Soleimanifar, ISIT 2026.

Plain ADMM solves the parity-polytope relaxation (their QP (8) with g = 0
and c = 1): per check, the neighborhood indicator z_j is projected onto the
parity polytope of the syndrome bit, and the qubit indicators minimize a
separable quadratic. The relaxation equals the exact LP (2) of the TIT
paper, so this solver replaces HiGHS where the subset formulation is too
large, such as detector error models. OSD post-processing reuses the LP
decoder's ordering, with integrality |x_i - 1/2| as reliability.
"""

import numpy as np
import torch

from qudec.codes import gf2_rref
from qudec.lp import syndrome_distance


def build_tanner(h):
    """Return (idx, mask) for check-neighbor gather/scatter on h (m, n).

    idx (m, d_max): variable indices per check slot, padded with column n;
    mask (m, d_max): True on real slots.
    """
    m, n = h.shape
    d_max = max(len(np.nonzero(h[j])[0]) for j in range(m)) if m else 0
    idx = np.full((m, d_max), n, dtype=np.int64)
    mask = np.zeros((m, d_max), dtype=bool)
    for j in range(m):
        nbr = np.nonzero(h[j])[0]
        idx[j, :len(nbr)] = nbr
        mask[j, :len(nbr)] = True
    return idx, mask


def parity_polytope_projection(z, parity, mask, grid=33, bisect_iters=24):
    """Project z (batch, m, d) onto per-check parity polytopes.

    Uses the two-slice representation of the parity polytope (Barman et
    al. 2013): in sorted space, membership for constituent parity r is
    the single constraint 2 sum_{k<=r+1} x - sum x <= r. The KKT
    correction for a candidate r is x = clip(v - mu n_slice - delta
    n_balance) with n_slice = +1 on the first r+1 coordinates and
    n_balance = +1 on the first coordinate only. Solve by nested search:
    inner bisection in mu (monotone), outer grid scan for the smallest
    delta root. Enumerate candidate parities and keep the nearest
    feasible point.

    parity (batch, m): required parity b of each check.
    mask (m, d): valid slots per check.
    """
    batch, m, d = z.shape
    big = 1e6
    vals = torch.where(mask.unsqueeze(0), z, torch.full_like(z, -big))
    deg = mask.sum(dim=1).long()
    kk = torch.arange(1, d + 1, dtype=torch.float32,
                      device=vals.device).view(1, 1, d)
    vals, sort_idx = vals.sort(dim=2, descending=True)

    def g_of(v, r):
        """Constraint value 2 sum_{k<=r+1} v - sum v - r in sorted space."""
        v = v.sort(dim=2, descending=True).values.clamp(0.0, 1.0)
        pre = (v * (kk <= r + 1)).sum(dim=2)
        return 2.0 * pre - v.sum(dim=2) - r

    def d2_of(x):
        return ((x - vals) ** 2).sum(dim=2)

    best_x = None
    best_d2 = None
    best_ok = None

    n_outer = torch.where(kk <= parity.unsqueeze(2) + 1, 1.0, -1.0)
    lam_hi = 8.0
    dgrid = torch.linspace(0.0, lam_hi, grid, dtype=torch.float32,
                           device=vals.device)
    max_r = min(d - 1, int(deg.max().item()) - 1)

    def g_outer(x):
        pre = (x * (kk <= parity.unsqueeze(2) + 1)).sum(dim=2)
        return 2.0 * pre - x.sum(dim=2) - parity

    for r in range(max_r + 1):
        n_slice = torch.where(kk <= r + 1, 1.0, -1.0)

        def x_of(mu, delta):
            if mu.ndim == 2:
                mu = mu.unsqueeze(2)
            if delta.ndim == 2:
                delta = delta.unsqueeze(2)
            return (vals - mu * n_slice - delta * n_outer).clamp(0.0, 1.0)

        # inner: smallest mu >= 0 with g_slice <= 0, by exact segment
        # walk over the piecewise-linear g. Breakpoints are the clip
        # events of the prefix and tail coordinates and the sort switches
        # between them; roots can lie at breakpoints, so grids miss them.
        def mu_star(delta):
            if delta.ndim == 2:
                delta = delta.unsqueeze(2)
            base = vals - delta * n_outer
            pre = base[:, :, :r + 1]
            tail = base[:, :, r + 1:]
            bps = torch.cat([pre, pre - 1.0, 1.0 - tail, -tail], dim=2)
            sw = (pre.unsqueeze(3) - tail.unsqueeze(2)) / 2.0
            bps = torch.cat([bps, sw.reshape(batch, m, -1)], dim=2)
            bps = bps.clamp(min=0.0)
            bps, _ = bps.sort(dim=2)
            n_pts = bps.shape[2]
            g0 = g_of(x_of(vals.new_zeros(batch, m), delta), r)
            best = vals.new_full((batch, m), float("inf"))
            found = g0 <= 1e-6
            best = torch.where(found, torch.zeros_like(best), best)
            prev_mu = vals.new_zeros(batch, m)
            prev_g = g0
            for t in range(n_pts):
                mu_t = bps[:, :, t]
                g_t = g_of(x_of(mu_t, delta), r)
                cand_t = torch.where((g_t <= 1e-6) & (prev_g > 1e-6), mu_t,
                                     torch.full_like(mu_t, float("inf")))
                denom = (prev_g - g_t).clamp(min=1e-12)
                interior = prev_mu + prev_g * (mu_t - prev_mu) / denom
                cand_i = torch.where(
                    (prev_g > 1e-6) & (g_t < -1e-6) & (prev_g > g_t),
                    interior, torch.full_like(mu_t, float("inf")))
                cand = torch.minimum(cand_t, cand_i)
                take = (~found) & (cand < best)
                best = torch.where(take, cand, best)
                found = found | (cand < float("inf"))
                keep = g_t > 1e-6
                prev_mu = torch.where(keep, mu_t, prev_mu)
                prev_g = torch.where(keep, g_t, prev_g)
            return torch.where(best.isinf(), vals.new_zeros(batch, m), best)

        # outer: smallest delta >= 0 with g_outer(mu*(delta), delta) <= 0
        h = torch.stack(
            [g_outer(x_of(mu_star(d.view(1, 1, 1)), d.view(1, 1, 1)))
             for d in dgrid], dim=0)
        feas = h <= 1e-6
        n_lead = (feas.cumsum(dim=0) == 0).sum(dim=0)
        lo_d = dgrid[(n_lead - 1).clamp(min=0)]
        hi_d = dgrid[n_lead.clamp(max=grid - 1)]
        for _ in range(bisect_iters):
            mid = (lo_d + hi_d) / 2
            hm = g_outer(x_of(mu_star(mid), mid))
            reached = hm <= 1e-6
            lo_d = torch.where(reached, lo_d, mid)
            hi_d = torch.where(reached, mid, hi_d)
        mu = mu_star(hi_d)
        x_r = x_of(mu, hi_d)
        x_r, _ = x_r.sort(dim=2, descending=True)
        total = x_r.sum(dim=2)
        fl = (total + 1e-4).floor().long()
        r_star = torch.where(fl % 2 == parity, fl, fl - 1)
        ok = (r_star == r) & (g_of(x_r, r) <= 1e-4)
        d2 = d2_of(x_r)
        if best_x is None:
            best_x, best_d2, best_ok = x_r, d2, ok
        else:
            take = ok & (~best_ok | (d2 < best_d2))
            best_x = torch.where(take.unsqueeze(2), x_r, best_x)
            best_d2 = torch.where(take, d2, best_d2)
            best_ok = ok | best_ok

    # lower-bound candidate for odd parity: x = clip(vals + mu), sum = 1
    lo = vals.new_zeros(batch, m)
    hi = vals.new_full((batch, m), 1.0)
    for _ in range(bisect_iters):
        mid = (lo + hi) / 2
        sm = (vals + mid.unsqueeze(2)).clamp(0.0, 1.0).sum(dim=2)
        below = sm < 1.0
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    x_low = (vals + hi.unsqueeze(2)).clamp(0.0, 1.0)
    x_low, _ = x_low.sort(dim=2, descending=True)
    ok_low = (parity == 1) & (x_low.sum(dim=2) >= 1.0 - 1e-4) \
        & (g_of(x_low, 1) <= 1e-4)
    d2 = d2_of(x_low)
    take = ok_low & (~best_ok | (d2 < best_d2))
    best_x = torch.where(take.unsqueeze(2), x_low, best_x)
    best_ok = ok_low | best_ok
    best_d2 = torch.where(take, d2, best_d2)

    # box-only fallback for entries with no valid candidate
    x0 = vals.clamp(0.0, 1.0)
    take = ~best_ok
    best_x = torch.where(take.unsqueeze(2), x0, best_x)

    out = torch.empty_like(best_x)
    out.scatter_(2, sort_idx, best_x)
    return torch.where(mask.unsqueeze(0), out, torch.zeros_like(out))


def admm_solve_batch(h, s, rho=2.0, alpha=1.0, max_iter=500,
                     tol_pri=1e-5, tol_dual=1e-5, device=None):
    """Solve the parity-polytope relaxation for syndromes s (batch, m).

    Return qubit error indicators x (batch, n) in [0, 1]. With rho large
    enough that rho * d_v > 1, the x-update is the unique minimizer of a
    separable quadratic, clipped to the unit cube. Tensors run on cuda
    when a GPU is available and device is not given.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m, n = h.shape
    if m == 0:
        return torch.zeros(s.shape[0], n, device=dev)
    idx, mask_np = build_tanner(h)
    mask = torch.as_tensor(mask_np, device=dev)
    flat_idx = torch.as_tensor(idx.reshape(-1), device=dev)
    batch = s.shape[0]
    parity = torch.as_tensor(s, dtype=torch.long, device=dev)
    d_v = torch.as_tensor((h != 0).sum(axis=0), dtype=torch.float32,
                          device=dev)
    z = parity_polytope_projection(
        torch.zeros(batch, m, idx.shape[1], device=dev), parity, mask)
    y = -rho * z
    x = torch.zeros(batch, n, device=dev)
    pad = torch.zeros(batch, n + 1, device=dev)
    for _ in range(max_iter):
        pad[:, :n] = x
        x_exp = pad[:, flat_idx].view(batch, m, -1)
        sums = torch.zeros(batch, n + 1, device=dev)
        sums.index_add_(1, flat_idx, (rho * z - y).reshape(batch, -1))
        x_new = ((sums[:, :n] - 1.0) / (rho * d_v)).clamp(0.0, 1.0)
        pad[:, :n] = x_new
        x_exp = pad[:, flat_idx].view(batch, m, -1)
        y_half = y + rho * (alpha - 1.0) * (x_exp - z)
        z_new = parity_polytope_projection(x_exp + y_half / rho, parity, mask)
        y = y_half + rho * (x_exp - z_new)
        if (torch.norm(x_exp - z_new) <= tol_pri
                and torch.norm(z_new - z) <= tol_dual):
            x = x_new
            break
        x, z = x_new, z_new
    return x


class AdmmOsdDecoder:
    """ADMM relaxation decoding with OSD post-processing for CSS codes.

    Mirrors LpOsdDecoder so benchmark_iid and the sinter wiring apply
    unchanged. osd_order: 0 for OSD-0, 1 for OSD-0 plus the combination
    sweep (OSD-CS).
    """

    def __init__(self, h_x, h_z, l_x, l_z, p_x=0.05, p_z=0.05, osd_order=1,
                 rho=2.0, max_iter=500):
        self.h_x = h_x.astype(np.int8)
        self.h_z = h_z.astype(np.int8)
        self.l_x = l_x.astype(np.int8)
        self.l_z = l_z.astype(np.int8)
        self.p_x = float(p_x)
        self.p_z = float(p_z)
        self.osd_order = osd_order
        self.rho = rho
        self.max_iter = max_iter

    def _osd(self, h, s, order):
        """Run OSD post-processing on h with syndrome s.

        order: column permutation, most likely error first. Return the
        per-qubit correction (n,) int8.
        """
        hp = h[:, order]
        aug = np.concatenate([hp, s.reshape(-1, 1)], axis=1)
        rref, pivots = gf2_rref(aug)
        r = len(pivots)
        e = np.zeros(h.shape[1], dtype=np.int8)
        e[pivots] = rref[:r, -1]
        if self.osd_order >= 1:
            pivot_set = set(pivots.tolist())
            for j in range(h.shape[1]):
                if j in pivot_set:
                    continue
                cand = e[pivots] ^ rref[:r, j]
                if cand.sum() + (e[j] ^ 1) < e[pivots].sum() + e[j]:
                    e[pivots] = cand
                    e[j] ^= 1
        corr = np.zeros(h.shape[1], dtype=np.int8)
        corr[order] = e
        return corr

    def _decode_osd(self, h, s, x):
        """OSD on one shot with ADMM indicators x as reliabilities."""
        dist = syndrome_distance(h, s)
        order = np.lexsort((dist, -x))
        return self._osd(h, s, order)

    def decode_corrections(self, sx, sz):
        """Decode syndromes sx (batch, m_z) and sz (batch, m_x).

        Return (corr_x, corr_z), each (batch, n) int8 per-qubit corrections.
        """
        x_x = admm_solve_batch(self.h_z, sx, rho=self.rho,
                               max_iter=self.max_iter).cpu().numpy()
        x_z = admm_solve_batch(self.h_x, sz, rho=self.rho,
                               max_iter=self.max_iter).cpu().numpy()
        c_x = np.stack(
            [self._decode_osd(self.h_z, sx[i], x_x[i])
             for i in range(sx.shape[0])], axis=0)
        c_z = np.stack(
            [self._decode_osd(self.h_x, sz[i], x_z[i])
             for i in range(sz.shape[0])], axis=0)
        return c_x, c_z

    def decode(self, sigma_x, sigma_z):
        """Decode one shot; return (corr_x, corr_z) as (n,) int8 arrays."""
        c_x, c_z = self.decode_corrections(
            sigma_x.reshape(1, -1).astype(np.int8),
            sigma_z.reshape(1, -1).astype(np.int8))
        return c_x[0], c_z[0]

    def decode_batch(self, shots):
        """Decode in the sinter protocol.

        shots: (batch, m_z + m_x) syndromes, concatenated [sigma_x | sigma_z].
        Return (batch, 2k) predicted observable flips
        [z-logical | x-logical], as uint8.
        """
        m = self.h_z.shape[0]
        sx = shots[:, :m].astype(np.int8)
        sz = shots[:, m:].astype(np.int8)
        c_x, c_z = self.decode_corrections(sx, sz)
        obs_x = (self.l_z @ c_x.T) % 2
        obs_z = (self.l_x @ c_z.T) % 2
        return np.concatenate([obs_x.T, obs_z.T], axis=1).astype(np.uint8)
