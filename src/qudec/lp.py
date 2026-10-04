"""LP decoding for CSS codes, following Gu and Soleimanifar, IEEE TIT 72(10) 2026.

The decoder relaxes the minimum-weight decoding problem to the LP (2) of the
paper: for each check, parity-consistent subsets of its neighborhood carry
weights that sum to one and couple to per-qubit error indicators. The qubit
indicators feed OSD post-processing as reliabilities, with ties broken by
Tanner-graph distance to a nontrivial syndrome (their Sec. V). The exact LP
is solved with HiGHS via scipy; for large detector error models use the
batched ADMM solver instead.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from qudec.codes import gf2_rref


def parity_subsets(neighbors, synd_bit):
    """Return parity-consistent subsets of check neighbors as index arrays.

    For a check with syndrome bit b, enumerate all subsets S of its
    neighborhood with |S| mod 2 == b, including the empty set when b == 0.
    """
    subsets = []
    for mask in range(1 << len(neighbors)):
        if (mask.bit_count() & 1) != synd_bit:
            continue
        subsets.append([neighbors[k] for k in range(len(neighbors))
                        if (mask >> k) & 1])
    return subsets


def lp_formulation(h, s):
    """Build the LP (2) for check matrix h (m, n) and syndrome s (m,).

    Variables are x_i for qubits followed by w_{j,S} for each check j and
    each parity-consistent subset S of its neighborhood. Return
    (c, a_eq, b_eq) for scipy.optimize.linprog with all variables >= 0;
    the sum-to-one constraints make explicit upper bounds redundant.
    """
    m, n = h.shape
    neighbors = [np.nonzero(h[j])[0].tolist() for j in range(m)]
    subsets = [parity_subsets(neighbors[j], int(s[j])) for j in range(m)]
    n_w = sum(len(v) for v in subsets)
    c = np.concatenate([np.ones(n), np.zeros(n_w)])
    rows, cols, data = [], [], []
    subset_vars = []
    offset = n
    for j in range(m):
        pairs = []
        for subset in subsets[j]:
            rows.append(j)
            cols.append(offset)
            data.append(1.0)
            pairs.append((offset, subset))
            offset += 1
        subset_vars.append(pairs)
    row = m
    for j in range(m):
        for i in neighbors[j]:
            rows.append(row)
            cols.append(i)
            data.append(-1.0)
            for var_idx, subset in subset_vars[j]:
                if i in subset:
                    rows.append(row)
                    cols.append(var_idx)
                    data.append(1.0)
            row += 1
    a_eq = sparse.coo_matrix((data, (rows, cols)),
                             shape=(row, n + n_w)).tocsr()
    b_eq = np.concatenate([np.ones(m), np.zeros(row - m)])
    return c, a_eq, b_eq


def solve_lp(h, s):
    """Solve the LP (2) for one syndrome; return the qubit indicators x."""
    c, a_eq, b_eq = lp_formulation(h, s)
    res = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"LP solver failed: {res.message}")
    return res.x[:h.shape[1]]


def syndrome_distance(h, s):
    """Return each qubit's Tanner-graph distance to a nontrivial syndrome.

    Multi-source BFS from checks with s == 1 over the bipartite Tanner
    graph. Qubits disconnected from every nontrivial check get a large
    distance, so the sort key degrades gracefully to zero syndromes.
    """
    m, n = h.shape
    nz = np.nonzero(s)[0]
    if len(nz) == 0:
        return np.zeros(n, dtype=np.int64)
    big = m + n + 1
    dist = np.full(n + m, big, dtype=np.int64)
    queue = list(int(n + j) for j in nz)
    for j in nz:
        dist[n + j] = 0
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        if u < n:
            nbrs = (np.nonzero(h[:, u])[0] + n).tolist()
        else:
            nbrs = np.nonzero(h[u - n])[0].tolist()
        for v in nbrs:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                queue.append(int(v))
    return dist[:n]


class LpOsdDecoder:
    """LP decoding with OSD post-processing for CSS codes.

    Mirrors BpOsdDecoder so benchmark_iid and the sinter wiring apply
    unchanged. osd_order: 0 for OSD-0, 1 for OSD-0 plus the combination
    sweep (OSD-CS).
    """

    def __init__(self, h_x, h_z, l_x, l_z, p_x=0.05, p_z=0.05, osd_order=1):
        self.h_x = h_x.astype(np.int8)
        self.h_z = h_z.astype(np.int8)
        self.l_x = l_x.astype(np.int8)
        self.l_z = l_z.astype(np.int8)
        self.p_x = float(p_x)
        self.p_z = float(p_z)
        self.osd_order = osd_order

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

    def _decode_side(self, h, s):
        """Decode one side of the CSS code: solve the LP, then OSD.

        Order columns by descending error indicator, breaking ties by
        distance to a nontrivial syndrome (Gu-Soleimanifar Sec. V).
        """
        x = solve_lp(h, s)
        dist = syndrome_distance(h, s)
        order = np.lexsort((dist, -x))
        return self._osd(h, s, order)

    def decode_corrections(self, sx, sz):
        """Decode syndromes sx (batch, m_z) and sz (batch, m_x).

        Return (corr_x, corr_z), each (batch, n) int8 per-qubit corrections.
        """
        c_x = np.stack(
            [self._decode_side(self.h_z, sx[i]) for i in range(sx.shape[0])],
            axis=0)
        c_z = np.stack(
            [self._decode_side(self.h_x, sz[i]) for i in range(sz.shape[0])],
            axis=0)
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
