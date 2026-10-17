"""Batched ADMM decoding following Gu and Soleimanifar, ISIT 2026.

Plain ADMM solves the parity-polytope relaxation (their QP (8) with g = 0
and c = 1): per check, the neighborhood indicator z_j is projected onto the
parity polytope of the syndrome bit, and the qubit indicators minimize a
separable quadratic. The relaxation equals the exact LP (2) of the TIT
paper, so this solver replaces HiGHS where the subset formulation is too
large, such as detector error models. OSD post-processing reuses the LP
decoder's ordering, with decreasing error indicators and Tanner-distance
tie breaking.
"""

import numpy as np
import torch

from qudec.lp import syndrome_distance
from qudec.osd import osd_decode


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


def parity_polytope_projection(z, parity, mask, grid=9, bisect_iters=32,
                             max_r=None):
    """Euclidean projection onto masked even/odd parity polytopes.

    Cut search finds the unique forbidden-set inequality violated by
    the box projection, if any. The desired projection lies on that
    face; solve ``theta @ clip(z - mu * theta) = |V| - 1`` for mu >= 0.
    This is the cut-search projection of Zhang and Siegel, extended to
    odd parity (Gensheimer et al., arXiv:1901.03240, Section IV).
    Bisection resolves the scalar root to floating-point precision.

    ``z`` has shape (batch, checks, slots), ``parity`` (batch, checks),
    and ``mask`` (checks, slots). Padded slots do not enter the cut.
    Legacy ``grid`` and ``max_r`` arguments are accepted but no longer
    restrict the projection: truncating parity slices is not valid.
    """
    if bisect_iters < 1:
        raise ValueError("bisect_iters must be positive")
    mask = torch.as_tensor(mask, dtype=torch.bool, device=z.device)
    parity = torch.as_tensor(parity, device=z.device)
    if z.ndim != 3 or mask.shape != z.shape[1:] or parity.shape != z.shape[:2]:
        raise ValueError("incompatible projection tensor shapes")
    if not torch.isfinite(z).all() or torch.any((parity != 0) & (parity != 1)):
        raise ValueError("projection needs finite values and binary parities")
    parity = parity.long()
    degrees = mask.sum(dim=-1)
    if torch.any((degrees == 0).unsqueeze(0) & (parity == 1)):
        raise ValueError("an empty check cannot have odd parity")
    if z.shape[-1] == 0:
        return torch.zeros_like(z)
    active = mask.unsqueeze(0)
    box = torch.where(active, z.clamp(0.0, 1.0), 0.0)
    plus = (z > 0.5) & active
    # Forbidden sets have parity opposite to the requested syndrome.
    flip = (plus.sum(dim=-1) % 2 == parity) & (degrees > 0).unsqueeze(0)
    distance = torch.where(active, (z - 0.5).abs(), float("inf"))
    nearest = distance.argmin(dim=-1, keepdim=True)
    toggle = torch.zeros_like(plus).scatter_(-1, nearest, flip.unsqueeze(-1))
    plus = plus ^ toggle
    theta = torch.where(active, torch.where(plus, 1.0, -1.0), 0.0)
    bound = plus.sum(dim=-1) - 1
    violated = (theta * box).sum(dim=-1) > bound
    violated &= (degrees > 0).unsqueeze(0)
    # At this upper bound all positive cut coordinates are 0 and all
    # negative cut coordinates are 1, so the halfspace is satisfied.
    lo = torch.zeros_like(bound, dtype=z.dtype)
    hi = torch.where(active, z.abs(), 0.0).amax(dim=-1) + 1.0
    for _ in range(bisect_iters):
        mid = (lo + hi) / 2
        candidate = (z - mid.unsqueeze(-1) * theta).clamp(0.0, 1.0)
        gap = (theta * candidate).sum(dim=-1) - bound
        lo = torch.where(gap > 0, mid, lo)
        hi = torch.where(gap > 0, hi, mid)
    projected = (z - hi.unsqueeze(-1) * theta).clamp(0.0, 1.0)
    out = torch.where(violated.unsqueeze(-1), projected, box)
    return torch.where(active, out, 0.0)


def admm_solve_batch(h, s, rho=2.0, alpha=1.0, max_iter=500,
                     tol_pri=1e-5, tol_dual=1e-5, device=None, max_r=None,
                     c_vec=None, g_vec=None, x_init=None, z_init=None,
                     y_init=None, return_state=False):
    """Solve the parity-polytope relaxation for syndromes s (batch, m).

    Return qubit error indicators x (batch, n) in [0, 1]. c_vec are the
    per-column objective coefficients (log-likelihood weights per the
    paper's Appendix C; uniform by default), normalized by their maximum
    absolute value for a linear objective. This preserves the LP optimum
    and keeps the x-update well conditioned. g_vec is the
    diagonal of the quadratic term (zero for the plain relaxation,
    -2 lambda for the LDR variant). With rho large enough that
    g_i + rho * d_v > 0 the x-update is the unique minimizer of a
    separable quadratic, clipped to the unit cube. Tensors run on cuda
    when a GPU is available.
    """
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be positive and finite")
    if not np.isfinite(alpha) or not 0 < alpha < 2:
        raise ValueError("alpha must be between 0 and 2")
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h = np.asarray(h)
    s = np.asarray(s)
    if h.ndim != 2 or s.ndim != 2 or s.shape[1] != h.shape[0]:
        raise ValueError("h and s must be compatible two-dimensional matrices")
    if np.any((h != 0) & (h != 1)) or np.any((s != 0) & (s != 1)):
        raise ValueError("h and s must be binary")
    m, n = h.shape
    batch = s.shape[0]
    idx, mask_np = build_tanner(h)
    mask = torch.as_tensor(mask_np, device=dev)
    flat_idx = torch.as_tensor(idx.reshape(-1), device=dev)
    parity = torch.as_tensor(s, dtype=torch.long, device=dev)
    d_v = torch.as_tensor((h != 0).sum(axis=0), dtype=torch.float32,
                          device=dev)
    c = torch.ones(n, device=dev) if c_vec is None else torch.as_tensor(
        c_vec, dtype=torch.float32, device=dev)
    g = torch.zeros(n, device=dev) if g_vec is None else torch.as_tensor(
        g_vec, dtype=torch.float32, device=dev)
    for name, value in (("c_vec", c), ("g_vec", g)):
        if value.shape not in ((n,), (batch, n)) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape (n,) or (batch, n)")
    # Positive rescaling preserves a linear objective, including signed
    # LLRs. LDR supplies a quadratic term and already normalized c0;
    # rescaling only c there would change the intended objective.
    if c_vec is not None and g_vec is None and c.numel():
        scale = c.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        c = c / scale
    connected = d_v > 0
    denom = rho * d_v + g
    if torch.any(connected & (denom <= 0)):
        raise ValueError("rho is too small for the quadratic penalty")
    # Unchecked DEM mechanisms are independent one-variable problems.
    # They must not divide by degree zero or be forced to the wrong MAP bit.
    safe_g = torch.where(g > 0, g, 1.0)
    isolated = torch.where(g > 0, (-c / safe_g).clamp(0.0, 1.0),
                           (c + 0.5 * g < 0).to(c.dtype))
    if m == 0 or batch == 0:
        x = isolated.expand(batch, n).clone()
        z = torch.zeros(batch, m, idx.shape[1], device=dev)
        return (x, z, z.clone()) if return_state else x
    if z_init is not None:
        z = z_init.clone()
        x = x_init.clone() if x_init is not None else torch.zeros(batch, n, device=dev)
        y = y_init.clone() if y_init is not None else torch.zeros_like(z)
    else:
        z = parity_polytope_projection(
            torch.zeros(batch, m, idx.shape[1], device=dev), parity, mask,
            max_r=max_r)
        y = torch.zeros_like(z)
        x = torch.zeros(batch, n, device=dev)
    pad = torch.zeros(batch, n + 1, device=dev)
    done = torch.zeros(batch, dtype=torch.bool, device=dev)
    safe_denom = torch.where(connected, denom, 1.0)
    for _ in range(max_iter):
        sums = torch.zeros(batch, n + 1, device=dev)
        sums.index_add_(1, flat_idx, (rho * z - y).reshape(batch, -1))
        x_new = ((sums[:, :n] - c) / safe_denom).clamp(0.0, 1.0)
        x_new = torch.where(connected, x_new, isolated)
        pad[:, :n] = x_new
        x_exp = pad[:, flat_idx].view(batch, m, -1)
        y_half = y + rho * (alpha - 1.0) * (x_exp - z)
        z_new = parity_polytope_projection(x_exp + y_half / rho, parity,
                                           mask, max_r=max_r)
        y_new = y_half + rho * (x_exp - z_new)
        primal = torch.linalg.vector_norm((x_exp - z_new).flatten(1), dim=1)
        dual = rho * torch.linalg.vector_norm((z_new - z).flatten(1), dim=1)
        # Freeze converged shots so decoding does not depend on which
        # other shots happen to share a benchmark batch.
        x = torch.where(done[:, None], x, x_new)
        z = torch.where(done[:, None, None], z, z_new)
        y = torch.where(done[:, None, None], y, y_new)
        done |= (primal <= tol_pri) & (dual <= tol_dual)
        if bool(done.all()):
            break
    return (x, z, y) if return_state else x


def admm_ldr_solve(h, s, rho=2.0, alpha=1.0, max_iter=500, device=None,
                   max_r=None, outer=5, beta=1.0, c_vec=None, tol=1e-5):
    """LDR-ADMM: Lagrangian dual relaxation with the adaptive penalty.

    Outer loop updates lambda by the subgradient x - x**2 with a
    diminishing step; the inner ADMM uses c = c0 + lambda and
    g = -2 lambda, warm-started from the previous solution. Stops when
    x is integral and syndrome-feasible, or the outer budget is spent.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if outer < 1 or not np.isfinite(beta) or beta < 0:
        raise ValueError("outer must be positive and beta finite and nonnegative")
    batch = s.shape[0]
    n = h.shape[1]
    if c_vec is None:
        c0 = torch.ones(n, device=dev)
    else:
        c0 = torch.as_tensor(c_vec, dtype=torch.float32, device=dev)
        if c0.shape not in ((n,), (batch, n)) or not torch.isfinite(c0).all():
            raise ValueError("c_vec must be finite with shape (n,) or (batch, n)")
        if c0.numel():
            c0 = c0 / c0.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    lam = torch.zeros(batch, n, device=dev)
    x = None
    z = None
    y = None
    done = torch.zeros(batch, dtype=torch.bool, device=dev)
    h_tensor = torch.as_tensor(h, dtype=torch.float32, device=dev)
    syndrome = torch.as_tensor(s, dtype=torch.float32, device=dev)
    for k in range(outer):
        x_new, z_new, y_new = admm_solve_batch(
            h, s, rho=rho, alpha=alpha, max_iter=max_iter, device=dev,
            max_r=max_r, c_vec=(c0 + lam), g_vec=(-2.0 * lam),
            x_init=x, z_init=z, y_init=y, return_state=True)
        if x is None:
            x, z, y = x_new, z_new, y_new
        else:
            x = torch.where(done[:, None], x, x_new)
            z = torch.where(done[:, None, None], z, z_new)
            y = torch.where(done[:, None, None], y, y_new)
        if x.numel() == 0:
            break
        integral = (x - x * x).abs().amax(dim=1) <= tol
        feasible = (torch.remainder(x.round() @ h_tensor.T, 2) == syndrome).all(dim=1)
        done |= integral & feasible
        if bool(done.all()):
            break
        lam = lam + torch.where(done[:, None], 0.0,
                                 (beta / (k + 1) ** 0.5) * (x - x * x))
    return x


class AdmmOsdDecoder:
    """ADMM relaxation decoding with OSD post-processing for CSS codes.

    Mirrors LpOsdDecoder so benchmark_iid and the array decoding interface apply
    unchanged. osd_order: 0 for OSD-0, 1 for OSD-0 plus the combination
    sweep (OSD-CS).
    """

    def __init__(self, h_x, h_z, l_x, l_z, p_x=0.05, p_z=0.05, osd_order=1,
                 osd_lam=None, rho=2.0, alpha=1.0, max_iter=500,
                 max_r=None, weights_x=None, weights_z=None, ldr=False,
                 ldr_outer=5, ldr_beta=1.0):
        self.h_x = h_x.astype(np.int8)
        self.h_z = h_z.astype(np.int8)
        self.l_x = l_x.astype(np.int8)
        self.l_z = l_z.astype(np.int8)
        self.p_x = np.asarray(p_x, dtype=float)
        self.p_z = np.asarray(p_z, dtype=float)
        self.osd_order = osd_order
        self.osd_lam = osd_lam
        self.rho = rho
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_r = max_r
        self.weights_x = weights_x
        self.weights_z = weights_z
        self.ldr = ldr
        self.ldr_outer = ldr_outer
        self.ldr_beta = ldr_beta

    def _osd(self, h, s, order):
        """Run OSD post-processing on h with syndrome s.

        order: column permutation, most likely error first. Return the
        per-qubit correction (n,) int8.
        """
        return osd_decode(h, s, order, self.osd_order, self.osd_lam)

    def _decode_osd(self, h, s, x):
        """OSD on one shot with ADMM indicators x as reliabilities."""
        dist = syndrome_distance(h, s)
        order = np.lexsort((dist, -x))
        return self._osd(h, s, order)

    def decode_corrections(self, sx, sz):
        """Decode syndromes sx (batch, m_z) and sz (batch, m_x).

        Return (corr_x, corr_z), each (batch, n) int8 per-qubit corrections.
        """
        if sx.shape[0] == 0 and sz.shape[0] == 0:
            return (np.zeros((0, self.h_z.shape[1]), dtype=np.int8),
                    np.zeros((0, self.h_x.shape[1]), dtype=np.int8))
        kw = dict(rho=self.rho, alpha=self.alpha, max_iter=self.max_iter,
                  max_r=self.max_r)
        if self.ldr:
            x_x = admm_ldr_solve(self.h_z, sx, outer=self.ldr_outer,
                                 beta=self.ldr_beta, c_vec=self.weights_x,
                                 **kw).cpu().numpy()
            x_z = admm_ldr_solve(self.h_x, sz, outer=self.ldr_outer,
                                 beta=self.ldr_beta, c_vec=self.weights_z,
                                 **kw).cpu().numpy()
        else:
            x_x = admm_solve_batch(self.h_z, sx, c_vec=self.weights_x,
                                   **kw).cpu().numpy()
            x_z = admm_solve_batch(self.h_x, sz, c_vec=self.weights_z,
                                   **kw).cpu().numpy()
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
        """Predict observable flips from unpacked detector arrays.

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
