"""Belief propagation with ordered statistics decoding for CSS codes."""

import numpy as np

from qudec.bp import channel_llrs, min_sum_bp, probability_vector
from qudec.osd import osd_decode


class BpOsdDecoder:
    """Min-sum BP followed by OSD post-processing on a CSS code.

    h_x: X-stabilizer matrix, detects Z errors
    h_z: Z-stabilizer matrix, detects X errors
    l_x, l_z: logical operator matrices, each k x n
    p_x, p_z: scalar or per-column marginal error probabilities in (0, 1)
    osd_order: 0 for OSD-0, 1 for OSD-0 plus the combination sweep (OSD-CS)
    osd_lam: set to run the reference exhaustive OSD-CS(lambda) sweep
    alpha: min-sum scaling; None selects the reference schedule 1 - 2**(-t)
    """

    def __init__(self, h_x, h_z, l_x, l_z, p_x, p_z,
                 max_iter=30, alpha=0.9, osd_order=1, osd_lam=None):
        self.h_x = h_x.astype(np.int8)
        self.h_z = h_z.astype(np.int8)
        self.l_x = l_x.astype(np.int8)
        self.l_z = l_z.astype(np.int8)
        self._probabilities_x = probability_vector(p_x, self.h_z.shape[1], "p_x")
        self._probabilities_z = probability_vector(p_z, self.h_x.shape[1], "p_z")
        self.p_x = float(p_x) if np.ndim(p_x) == 0 else self._probabilities_x.copy()
        self.p_z = float(p_z) if np.ndim(p_z) == 0 else self._probabilities_z.copy()
        self.max_iter = max_iter
        self.alpha = alpha
        self.osd_order = osd_order
        self.osd_lam = osd_lam

    def _bp(self, h, s, llr):
        """Run min-sum BP on check matrix h for syndromes s.

        h: (m, n) binary, s: (batch, m) syndromes, llr: (n,) channel LLR.
        Return final per-variable LLRs, shape (batch, n).
        """
        return min_sum_bp(h, s, llr, self.max_iter, self.alpha)

    def _osd(self, h, s, order):
        """Run OSD post-processing on h with syndrome s.

        order: column permutation, ascending LLR (most likely error first).
        Return the per-qubit correction (n,) int8.
        """
        return osd_decode(h, s, order, self.osd_order, self.osd_lam)

    def decode_corrections(self, sx, sz):
        """Decode syndromes sx (batch, m_z) and sz (batch, m_x).

        Return (corr_x, corr_z), each (batch, n) int8 per-qubit corrections.
        """
        if sx.shape[0] == 0 and sz.shape[0] == 0:
            return (np.zeros((0, self.h_z.shape[1]), dtype=np.int8),
                    np.zeros((0, self.h_x.shape[1]), dtype=np.int8))
        llr_x = channel_llrs(self._probabilities_x, self.h_z.shape[1], "p_x")
        l_x = self._bp(self.h_z, sx, llr_x).numpy()
        llr_z = channel_llrs(self._probabilities_z, self.h_x.shape[1], "p_z")
        l_z = self._bp(self.h_x, sz, llr_z).numpy()
        c_x = np.stack([
            self._osd(self.h_z, sx[i], np.argsort(l_x[i], kind="stable"))
            for i in range(sx.shape[0])], axis=0)
        c_z = np.stack([
            self._osd(self.h_x, sz[i], np.argsort(l_z[i], kind="stable"))
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
