"""Belief propagation with ordered statistics decoding for CSS codes."""

import numpy as np
import torch

from qudec.osd import osd_decode


class BpOsdDecoder:
    """Min-sum BP followed by OSD post-processing on a CSS code.

    h_x: X-stabilizer matrix, detects Z errors
    h_z: Z-stabilizer matrix, detects X errors
    l_x, l_z: logical operator matrices, each k x n
    p_x, p_z: marginal per-qubit error probabilities for the channel LLR
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
        self.p_x = float(p_x)
        self.p_z = float(p_z)
        self.max_iter = max_iter
        self.alpha = alpha
        self.osd_order = osd_order
        self.osd_lam = osd_lam

    def _bp(self, h, s, llr):
        """Run min-sum BP on check matrix h for syndromes s.

        h: (m, n) binary, s: (batch, m) syndromes, llr: (n,) channel LLR.
        Return final per-variable LLRs, shape (batch, n).
        """
        a = torch.as_tensor(h, dtype=torch.float32)
        s = torch.as_tensor(s, dtype=torch.float32)
        l_ch = torch.as_tensor(llr, dtype=torch.float32)
        batch, n = s.shape[0], a.shape[1]
        big = 1e4
        l_vc = a.unsqueeze(0) * l_ch.view(1, 1, n)
        for t in range(self.max_iter):
            a_t = self.alpha if self.alpha is not None else 1.0 - 2.0 ** (-(t + 1))
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
        n = self.h_z.shape[1]
        llr_x = np.full(n, np.log((1 - self.p_x) / self.p_x))
        l_x = self._bp(self.h_z, sx, llr_x).numpy()
        llr_z = np.full(n, np.log((1 - self.p_z) / self.p_z))
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
