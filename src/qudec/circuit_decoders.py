"""Decoders for stim detector error models (single non-CSS matrix form).

The dem check matrix has rows = detectors and columns = independent error
mechanisms, with observables as rows of the observable matrix. The two
decoders here run min-sum BP or the batched parity-polytope ADMM on the
raw matrix, followed by OSD post-processing from qudec.osd. Error
mechanisms are treated as independent, using their individual DEM
probabilities when supplied.

torch is imported lazily so the module stays importable without it.
"""

import numpy as np

from qudec.bp import channel_llrs, min_sum_bp, probability_vector
from qudec.osd import osd_decode


def _detector_shots(shots, h):
    shots = np.asarray(shots)
    if (shots.ndim != 2 or shots.shape[1] != h.shape[0]
            or not np.all((shots == 0) | (shots == 1))):
        raise ValueError(f"shots must be binary with {h.shape[0]} detector columns")
    if np.any(shots[:, ~np.any(h, axis=1)]):
        raise ValueError("an empty check cannot have nonzero syndrome")
    return shots.astype(np.int8)


class BpOsdDemDecoder:
    """Min-sum BP with OSD post-processing on a dem check matrix."""

    def __init__(self, h, l, p=0.01, max_iter=30, alpha=0.9,
                 osd_order=1, osd_lam=None, probabilities=None):
        self.h = h.astype(np.int8)
        self.l = l.astype(np.int8)
        prior = p if probabilities is None else probabilities
        self.probabilities = probability_vector(prior, self.h.shape[1])
        self.p = float(prior) if np.ndim(prior) == 0 else self.probabilities.copy()
        self.max_iter = max_iter
        self.alpha = alpha
        self.osd_order = osd_order
        self.osd_lam = osd_lam

    def decode_batch(self, shots):
        """Decode detector shots (batch, m); return obs flips (batch, k)."""
        n = self.h.shape[1]
        llr = channel_llrs(self.probabilities, n)
        s = _detector_shots(shots, self.h)
        if s.shape[0] == 0 or n == 0:
            return np.zeros((s.shape[0], self.l.shape[0]), dtype=np.uint8)
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
                 max_r=None, osd_order=1, osd_lam=None, probabilities=None,
                 weighted=None, ldr=False, ldr_outer=4, ldr_beta=1.0):
        self.h = h.astype(np.int8)
        self.l = l.astype(np.int8)
        self.rho = rho
        self.alpha = alpha
        self.max_iter = max_iter
        self.max_r = max_r
        self.osd_order = osd_order
        self.osd_lam = osd_lam
        self.ldr = bool(ldr)
        self.ldr_outer = int(ldr_outer)
        self.ldr_beta = float(ldr_beta)
        self.probabilities = (None if probabilities is None else
                              probability_vector(probabilities, self.h.shape[1]))
        if weighted is True and self.probabilities is None:
            raise ValueError("weighted DEM ADMM requires mechanism probabilities")
        use_weights = (self.probabilities is not None
                       if weighted is None else bool(weighted))
        self.weighted = use_weights
        self.weights = (channel_llrs(self.probabilities, self.h.shape[1])
                        if use_weights else None)

    def decode_batch(self, shots):
        """Decode detector shots (batch, m); return obs flips (batch, k)."""
        from qudec.admm import admm_ldr_solve, admm_solve_batch
        from qudec.lp import syndrome_distance

        s = _detector_shots(shots, self.h)
        if s.shape[0] == 0 or self.h.shape[1] == 0:
            return np.zeros((s.shape[0], self.l.shape[0]), dtype=np.uint8)
        if self.ldr:
            x = admm_ldr_solve(self.h, s, rho=self.rho, alpha=self.alpha,
                               max_iter=self.max_iter, max_r=self.max_r,
                               outer=self.ldr_outer, beta=self.ldr_beta,
                               c_vec=self.weights).cpu().numpy()
        else:
            x = admm_solve_batch(self.h, s, rho=self.rho, alpha=self.alpha,
                                 max_iter=self.max_iter, max_r=self.max_r,
                                 c_vec=self.weights).cpu().numpy()
        obs = []
        for i in range(s.shape[0]):
            dist = syndrome_distance(self.h, s[i])
            order = np.lexsort((dist, -x[i]))
            corr = osd_decode(self.h, s[i], order, self.osd_order,
                              self.osd_lam)
            obs.append(((self.l @ corr) % 2).astype(np.uint8))
        return np.stack(obs, axis=0)
