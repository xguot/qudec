"""Phenomenological memory experiments with a perfect terminal readout.

There are d noisy data/measurement rounds, followed by a noiseless check
measurement. The last detector is m_d, so a valid correction also closes
the final data syndrome. terminal_perfect=False explicitly selects the
historical open boundary; its logical-only rate is not comparable to the
default closed-memory failure rate.
"""

import hashlib

import numpy as np

from qudec.noise import sample_iid_errors


def build_phenom_matrix(h, d, terminal_perfect=True):
    """Return ((d + terminal_perfect)*m, d*n + d*m) detector matrix.

    Columns are d data blocks then d measurement-error blocks. Closure
    adds the last measurement-error identity, without a noisy data round.
    """
    if d < 1:
        raise ValueError("d must be positive")
    m, n = h.shape
    out = np.zeros(((d + int(terminal_perfect)) * m, d * (n + m)), np.int8)
    eye = np.eye(m, dtype=np.int8)
    for t in range(d):
        out[t*m:(t+1)*m, t*n:(t+1)*n] = h
        out[t*m:(t+1)*m, d*n+t*m:d*n+(t+1)*m] = eye
        if t:
            out[t*m:(t+1)*m, d*n+(t-1)*m:d*n+t*m] = eye
    if terminal_perfect:
        out[d*m:(d+1)*m, d*n+(d-1)*m:d*n+d*m] = eye
    return out


def phenom_priors(h_x, h_z, d, p, p_meas=None):
    """BP priors: data marginal 2p/3, measurement probability p_meas.

    This corrects the historical scalar-prior BP baseline, which assigned
    the data error probability to measurement columns as well.
    """
    p_meas = p if p_meas is None else p_meas
    n = h_x.shape[1]
    return tuple(np.concatenate([np.full(d*n, 2*p/3),
                                 np.full(d*h.shape[0], p_meas)])
                 for h in (h_z, h_x))


def sample_phenom(h_x, h_z, p, d, shots, model="depolarizing", seed=0,
                  p_meas=None, terminal_perfect=True):
    """Return (sx, sz, cumulative_ex, cumulative_ez) for a memory run.

    Sampling is independent of decoding batch size. Reuse the tuple for
    paired variants. Closure adds a detector block without consuming any
    extra randomness; the historical first d blocks are unchanged.
    """
    if d < 1 or shots < 1:
        raise ValueError("d and shots must be positive")
    p_meas = p if p_meas is None else p_meas
    if not 0 <= p_meas <= 1:
        raise ValueError("p_meas must be in [0, 1]")
    n = h_x.shape[1]
    m_z, m_x = h_z.shape[0], h_x.shape[0]
    rng = np.random.default_rng(seed)
    ex_tot = np.zeros((shots, n), np.int8)
    ez_tot = np.zeros((shots, n), np.int8)
    prev_sx = np.zeros((shots, m_z), np.int8)
    prev_sz = np.zeros((shots, m_x), np.int8)
    blocks = d + int(terminal_perfect)
    sx = np.zeros((shots, blocks*m_z), np.int8)
    sz = np.zeros((shots, blocks*m_x), np.int8)
    for t in range(d):
        e_x, e_z = sample_iid_errors(n, p, shots, model,
                                     seed=int(rng.integers(0, 2**32)))
        meas_z = (rng.random((shots, m_z)) < p_meas).astype(np.int8)
        meas_x = (rng.random((shots, m_x)) < p_meas).astype(np.int8)
        ex_tot ^= e_x
        ez_tot ^= e_z
        s_x = ((h_z @ ex_tot.T) % 2).T ^ meas_z
        s_z = ((h_x @ ez_tot.T) % 2).T ^ meas_x
        sx[:, t*m_z:(t+1)*m_z] = s_x ^ prev_sx
        sz[:, t*m_x:(t+1)*m_x] = s_z ^ prev_sz
        prev_sx, prev_sz = s_x, s_z
    if terminal_perfect:
        sx[:, d*m_z:] = meas_z
        sz[:, d*m_x:] = meas_x
    return sx, sz, ex_tot, ez_tot


class PhenomDecoder:
    """Wrap a CSS decoder on the memory detector matrices.

    Logical matrices include zero columns for measurement errors. The
    previous wrapper omitted these columns, breaking decode_batch.
    """

    def __init__(self, decoder_cls, h_x, h_z, l_x, l_z, d,
                 terminal_perfect=True, **kwargs):
        self.d = d
        self.terminal_perfect = terminal_perfect
        h_x_ph = build_phenom_matrix(h_x, d, terminal_perfect)
        h_z_ph = build_phenom_matrix(h_z, d, terminal_perfect)
        # l_z acts on X corrections (h_z columns); l_x on Z corrections.
        self.l_x = np.pad(np.tile(l_x, (1, d)), ((0, 0), (0, d*h_x.shape[0])))
        self.l_z = np.pad(np.tile(l_z, (1, d)), ((0, 0), (0, d*h_z.shape[0])))
        self.base = decoder_cls(h_x_ph, h_z_ph, self.l_x, self.l_z, **kwargs)
        self.h_x, self.h_z = self.base.h_x, self.base.h_z

    def decode_corrections(self, sx, sz):
        """Return data-round and measurement corrections in detector space."""
        return self.base.decode_corrections(sx, sz)

    def decode_batch(self, shots):
        """Return [Z-logical | X-logical] flips for concatenated [sx | sz]."""
        m = self.h_z.shape[0]
        if shots.ndim != 2 or shots.shape[1] != m + self.h_x.shape[0]:
            raise ValueError("shots must contain concatenated sx and sz detectors")
        c_x, c_z = self.decode_corrections(shots[:, :m], shots[:, m:])
        return np.concatenate([(self.l_z @ c_x.T).T % 2,
                               (self.l_x @ c_z.T).T % 2], axis=1).astype(np.uint8)


def benchmark_phenom(decoder, h_x, h_z, l_x, l_z, p, d, shots,
                     model="depolarizing", seed=0, batch_size=64,
                     p_meas=None, samples=None, return_failures=False):
    """Decode bounded batches of one shared sample set.

    Closed-memory failures include residual data syndrome, logical flips,
    and invalid detector corrections. The explicit legacy open protocol
    retains its logical-only rate and reports residual/invalid counts
    separately. Optional per-shot outcomes enable paired McNemar tests.
    """
    if shots < 1 or batch_size < 1:
        raise ValueError("shots and batch_size must be positive")
    closed = getattr(decoder, "terminal_perfect", True)
    samples = samples if samples is not None else sample_phenom(
        h_x, h_z, p, d, shots, model, seed, p_meas, closed)
    sx, sz, ex, ez = samples
    n = h_x.shape[1]
    expected = ((shots, decoder.h_z.shape[0]), (shots, decoder.h_x.shape[0]),
                (shots, n), (shots, n))
    if any(a.shape != shape for a, shape in zip(samples, expected)):
        raise ValueError("sample shapes do not match decoder protocol and shot count")
    digest = hashlib.sha256()
    for a in samples:
        digest.update(str(a.shape).encode())
        digest.update(np.asarray(a, np.int8).tobytes())
    flags = {key: np.zeros(shots, dtype=bool) for key in
             ("logical_x", "logical_z", "invalid_x", "invalid_z",
              "residual_x", "residual_z", "failure_x", "failure_z", "failure")}
    for start in range(0, shots, batch_size):
        stop = min(shots, start + batch_size)
        sl = slice(start, stop)
        c_x, c_z = decoder.decode_corrections(sx[sl], sz[sl])
        cx = c_x[:, :d*n].reshape(stop-start, d, n).sum(axis=1) % 2
        cz = c_z[:, :d*n].reshape(stop-start, d, n).sum(axis=1) % 2
        rx, rz = ex[sl] ^ cx.astype(np.int8), ez[sl] ^ cz.astype(np.int8)
        flags["logical_x"][sl] = np.any((l_z @ rx.T) % 2, axis=0)
        flags["logical_z"][sl] = np.any((l_x @ rz.T) % 2, axis=0)
        flags["residual_x"][sl] = np.any((h_z @ rx.T) % 2, axis=0)
        flags["residual_z"][sl] = np.any((h_x @ rz.T) % 2, axis=0)
        flags["invalid_x"][sl] = np.any(((decoder.h_z @ c_x.T) % 2).T != sx[sl], axis=1)
        flags["invalid_z"][sl] = np.any(((decoder.h_x @ c_z.T) % 2).T != sz[sl], axis=1)
    for sector in ("x", "z"):
        flags[f"failure_{sector}"] = flags[f"logical_{sector}"].copy()
        if closed:
            flags[f"failure_{sector}"] |= (flags[f"residual_{sector}"]
                                             | flags[f"invalid_{sector}"])
    flags["failure"] = flags["failure_x"] | flags["failure_z"]
    counts = {key: int(value.sum()) for key, value in flags.items()}
    result = dict(counts, ler=counts["failure"]/shots,
                  ler_x=counts["failure_x"]/shots, ler_z=counts["failure_z"]/shots,
                  shots=shots, p=p, p_meas=p if p_meas is None else p_meas,
                  d=d, seed=seed, model=model, batch_size=batch_size,
                  protocol="closed-v1" if closed else "legacy-open-v1",
                  sample_id=digest.hexdigest())
    if return_failures:
        result["outcomes"] = {key: "".join(value.astype(np.uint8).astype(str))
                              for key, value in flags.items()}
    return result
