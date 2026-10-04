"""Phenomenological noise: per-round data errors and measurement errors.

Detector syndromes delta_t = s_t xor s_{t-1} with
s_t = h (e_1 xor ... xor e_t) xor m_t give the time-expanded system

    delta_t = h e_t xor m_t xor m_{t-1}

for rounds t = 1..d. Decoding runs on the block-bidiagonal matrix with
d copies of h on the data columns and a bidiagonal identity on the
measurement columns; the final data correction is the xor of the
per-round corrections.
"""

import numpy as np

from qudec.noise import sample_iid_errors


def build_phenom_matrix(h, d):
    """Return the time-expanded check matrix for h (m, n) over d rounds.

    Shape (d*m, d*n + d*m): columns are d blocks of data errors followed
    by d blocks of measurement errors.
    """
    m, n = h.shape
    out = np.zeros((d * m, d * n + d * m), dtype=np.int8)
    eye = np.eye(m, dtype=np.int8)
    for t in range(d):
        out[t * m:(t + 1) * m, t * n:(t + 1) * n] = h
        out[t * m:(t + 1) * m, d * n + t * m:d * n + (t + 1) * m] = eye
        if t > 0:
            out[t * m:(t + 1) * m,
                d * n + (t - 1) * m:d * n + t * m] = eye
    return out


def sample_phenom(h_x, h_z, p, d, shots, model="depolarizing", seed=0,
                  p_meas=None):
    """Sample d rounds of data and measurement errors.

    Return (sx, sz, ex, ez): detector syndromes sx (shots, d*m_z) and
    sz (shots, d*m_x), plus the cumulative data errors ex, ez
    (shots, n) for the logical check. Measurement bits flip with
    probability p_meas, which defaults to p.
    """
    p_meas = p if p_meas is None else p_meas
    n = h_x.shape[1]
    m_z = h_z.shape[0]
    m_x = h_x.shape[0]
    rng = np.random.default_rng(seed)
    ex_tot = np.zeros((shots, n), dtype=np.int8)
    ez_tot = np.zeros((shots, n), dtype=np.int8)
    prev_sx = np.zeros((shots, m_z), dtype=np.int8)
    prev_sz = np.zeros((shots, m_x), dtype=np.int8)
    sx = np.zeros((shots, d * m_z), dtype=np.int8)
    sz = np.zeros((shots, d * m_x), dtype=np.int8)
    for t in range(d):
        e_x, e_z = sample_iid_errors(n, p, shots, model,
                                     seed=int(rng.integers(0, 2**32)))
        meas_z = (rng.random((shots, m_z)) < p_meas).astype(np.int8)
        meas_x = (rng.random((shots, m_x)) < p_meas).astype(np.int8)
        ex_tot ^= e_x
        ez_tot ^= e_z
        s_x = ((h_z @ ex_tot.T) % 2).T ^ meas_z
        s_z = ((h_x @ ez_tot.T) % 2).T ^ meas_x
        sx[:, t * m_z:(t + 1) * m_z] = s_x ^ prev_sx
        sz[:, t * m_x:(t + 1) * m_x] = s_z ^ prev_sz
        prev_sx = s_x
        prev_sz = s_z
    return sx, sz, ex_tot, ez_tot


class PhenomDecoder:
    """Wrap a CSS decoder for d rounds of phenomenological noise.

    The base decoder class is re-instantiated on the time-expanded check
    matrices, so decode_corrections and decode_batch operate on detector
    syndromes directly. The exposed logical matrices tile d copies, so
    benchmarks can check the accumulated correction.
    """

    def __init__(self, decoder_cls, h_x, h_z, l_x, l_z, d, **kwargs):
        self.d = d
        h_x_ph = build_phenom_matrix(h_x, d)
        h_z_ph = build_phenom_matrix(h_z, d)
        l_x_ph = np.tile(l_x, (1, d))
        l_z_ph = np.tile(l_z, (1, d))
        self.base = decoder_cls(h_x_ph, h_z_ph, l_x_ph, l_z_ph, **kwargs)
        self.h_x = self.base.h_x
        self.h_z = self.base.h_z
        self.l_x = l_x_ph
        self.l_z = l_z_ph

    def decode_corrections(self, sx, sz):
        """Decode detector syndromes; return per-round corrections.

        Return (corr_x, corr_z), each (shots, d*n + d*m) int8 with the
        d data blocks first and the d measurement blocks after; slice
        the data part for the accumulated correction.
        """
        return self.base.decode_corrections(sx, sz)


def benchmark_phenom(decoder, h_x, h_z, l_x, l_z, p, d, shots,
                     model="depolarizing", seed=0):
    """Benchmark a phenom decoder over d rounds of data and measurement
    errors, checking the accumulated correction against the logicals."""
    n = h_x.shape[1]
    sx, sz, ex, ez = sample_phenom(h_x, h_z, p, d, shots, model, seed)
    c_x, c_z = decoder.decode_corrections(sx, sz)
    c_x_tot = (c_x[:, :d * n].reshape(shots, d, n).sum(axis=1)
               % 2).astype(np.int8)
    c_z_tot = (c_z[:, :d * n].reshape(shots, d, n).sum(axis=1)
               % 2).astype(np.int8)
    ok_x = np.all(((l_z @ (ex.T ^ c_x_tot.T)) % 2) == 0, axis=0)
    ok_z = np.all(((l_x @ (ez.T ^ c_z_tot.T)) % 2) == 0, axis=0)
    invalid_x = int(
        np.any(((decoder.h_z @ c_x.T) % 2).T != sx, axis=1).sum())
    invalid_z = int(
        np.any(((decoder.h_x @ c_z.T) % 2).T != sz, axis=1).sum())
    return {
        "ler": float(1 - (ok_x & ok_z).mean()),
        "ler_x": float(1 - ok_x.mean()),
        "ler_z": float(1 - ok_z.mean()),
        "invalid_x": invalid_x,
        "invalid_z": invalid_z,
        "shots": shots,
        "p": p,
        "d": d,
    }
