"""Decoders for stim detector error models (single non-CSS matrix form).

The dem check matrix has rows = detectors and columns = independent error
mechanisms, with observables as rows of the observable matrix. The two
decoders here run min-sum BP or the batched parity-polytope ADMM on the
raw matrix, followed by OSD post-processing from qudec.osd. Error
mechanisms are treated as independent with a uniform prior; this is the
standard first-order decoder for circuit-level dems.

torch is imported lazily so the module stays importable without it.
"""

import numpy as np

from qudec.osd import osd_decode


def min_sum_bp(h, s, llr, max_iter=30, alpha=0.9):
    """Run min-sum BP on one check matrix; return final LLRs (batch, n)."""
    import torch

    a = torch.as_tensor(h, dtype=torch.float32)
    s = torch.as_tensor(s, dtype=torch.float32)
    l_ch = torch.as_tensor(llr, dtype=torch.float32)
    batch, n = s.shape[0], a.shape[1]
    big = 1e4
    l_vc = a.unsqueeze(0) * l_ch.view(1, 1, n)
    for t in range(max_iter):
        a_t = alpha if alpha is not None else 1.0 - 2.0 ** (-(t + 1))
        abs_l = torch.where(a > 0, l_vc.abs(), big)
        amin, _ = abs_l.min(dim=2)
        asecond = torch.where(
            abs_l == amin.unsqueeze(2), big, abs_l).min(dim=2).values
        mag = torch.where(
            abs_l == amin.unsqueeze(2), asecond.unsqueeze(2),
            amin.unsqueeze(2)) * a.unsqueeze(0)
        neg = (l_vc < 0) & (a > 0)
        parity = (neg.sum(dim=2) + s) % 2
        sign = (1 - 2 * parity).unsqueeze(2) * (
            1 - 2 * (l_vc < 0).float())
        m_cv = sign * mag * a_t
        sum_in = m_cv.sum(dim=1)
        l_vc = a.unsqueeze(0) * (
            l_ch.view(1, 1, n) + sum_in.unsqueeze(1) - m_cv)
    return l_ch.view(1, n) + m_cv.sum(dim=1)


class BpOsdDemDecoder:
    """Min-sum BP with OSD post-processing on a dem check matrix."""

    def __init__(self, h, l, p=0.01, max_iter=30, alpha=0.9,
                 osd_order=1, osd_lam=None):
        self.h = h.astype(np.int8)
        self.l = l.astype(np.int8)
        self.p = float(p)
        self.max_iter = max_iter
        self.alpha = alpha
        self.osd_order = osd_order
        self.osd_lam = osd_lam

    def decode_batch(self, shots):
        """Decode detector shots (batch, m); return obs flips (batch, k)."""
        n = self.h.shape[1]
        llr = np.full(n, np.log((1 - self.p) / self.p))
        s = shots.astype(np.int8)
        l_fin = min_sum_bp(self.h, s, llr, self.max_iter,
                           self.alpha).numpy()
        obs = []
        for i in range(s.shape[0]):
            order = np.argsort(l_fin[i], kind="stable")
            corr = osd_decode(self.h, s[i], order, self.osd_order,
                              self.osd_lam)
            obs.append(((self.l @ corr) % 2).astype(np.uint8))
        return np.stack(obs, axis=0)


class AdmmOsdDemDecoder:
    """Parity-polytope ADMM with OSD post-processing on a dem matrix."""

    def __init__(self, h, l, rho=2.0, alpha=1.0, max_iter=500,
                 max_r=None, osd_order=1, osd_lam=None):
        self.h = h.astype(np.int8)
        self.l = l.astype(np.int8)
        self.rho = rho
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_r = max_r
        self.osd_order = osd_order
        self.osd_lam = osd_lam

    def decode_batch(self, shots):
        """Decode detector shots (batch, m); return obs flips (batch, k)."""
        from qudec.admm import admm_solve_batch
        from qudec.lp import syndrome_distance

        s = shots.astype(np.int8)
        x = admm_solve_batch(self.h, s, rho=self.rho, alpha=self.alpha,
                             max_iter=self.max_iter,
                             max_r=self.max_r).cpu().numpy()
        obs = []
        for i in range(s.shape[0]):
            dist = syndrome_distance(self.h, s[i])
            order = np.lexsort((dist, -x[i]))
            corr = osd_decode(self.h, s[i], order, self.osd_order,
                              self.osd_lam)
            obs.append(((self.l @ corr) % 2).astype(np.uint8))
        return np.stack(obs, axis=0)
