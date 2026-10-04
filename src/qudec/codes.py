"""CSS code constructions and GF(2) linear algebra for QEC decoding."""

import numpy as np


def build_circulant(l, m, monomials):
    """Return the lm x lm binary double-circulant of the given monomials.

    Monomial (a, b) sets H[r, c] = 1 where r - c = (a, b) mod (l, m).
    """
    out = np.zeros((l * m, l * m), dtype=np.int8)
    rows = np.arange(l * m)
    for a, b in monomials:
        src = rows
        dst = ((rows // m + a) % l) * m + (rows % m + b) % m
        out[dst, src] ^= 1
    return out


def bivariate_bicycle(l, m, a_monomials, b_monomials):
    """Return (h_x, h_z) check matrices of a bivariate bicycle code."""
    a = build_circulant(l, m, a_monomials)
    b = build_circulant(l, m, b_monomials)
    h_x = np.concatenate([a, b], axis=1).astype(np.int8)
    h_z = np.concatenate([b.T, a.T], axis=1).astype(np.int8)
    return h_x, h_z


def gross_code():
    """Return (h_x, h_z) for the [[144, 12, 12]] bivariate bicycle code."""
    return bivariate_bicycle(
        12, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])


def medium_code():
    """Return (h_x, h_z) for the [[72, 12, 6]] bivariate bicycle code."""
    return bivariate_bicycle(
        6, 6, [(3, 0), (0, 1), (0, 2)], [(0, 3), (1, 0), (2, 0)])


def steane_code():
    """Return (h_x, h_z) for the [[7, 1, 3]] Steane code."""
    h = np.array([
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ], dtype=np.int8)
    return h.copy(), h.copy()


def repetition_code(d):
    """Return (h_x, h_z) for the [[d, 1, d]] repetition code."""
    h_z = np.zeros((d - 1, d), dtype=np.int8)
    for i in range(d - 1):
        h_z[i, i] = 1
        h_z[i, i + 1] = 1
    h_x = np.zeros((0, d), dtype=np.int8)
    return h_x, h_z


def gf2_rref(mat):
    """Return (rref, pivots) of a binary matrix in reduced row echelon form."""
    a = mat.astype(np.int8).copy()
    rows, cols = a.shape
    pivots = []
    r = 0
    for c in range(cols):
        if r == rows:
            break
        below = np.nonzero(a[r:, c])[0]
        if len(below) == 0:
            continue
        pivot = r + below[0]
        if pivot != r:
            a[[r, pivot]] = a[[pivot, r]]
        mask = (a[:, c] == 1)
        mask[r] = False
        a[mask] ^= a[r]
        pivots.append(c)
        r += 1
    return a, np.array(pivots, dtype=np.int64)


def gf2_rank(mat):
    """Return the GF(2) rank of a binary matrix."""
    _, pivots = gf2_rref(mat)
    return len(pivots)


def gf2_nullspace(mat):
    """Return a basis of {v : mat @ v = 0} as rows of a binary matrix."""
    rref, pivots = gf2_rref(mat)
    rows, cols = rref.shape
    pivot_set = set(pivots.tolist())
    basis = []
    for f in range(cols):
        if f in pivot_set:
            continue
        v = np.zeros(cols, dtype=np.int8)
        v[f] = 1
        for i, c in enumerate(pivots):
            if rref[i, f]:
                v[c] = 1
        basis.append(v)
    return np.array(basis, dtype=np.int8).reshape(-1, cols)


def quotient_basis(base, sub):
    """Return a basis of rowspace(base) modulo rowspace(sub) as rows."""
    mat = sub.astype(np.int8).copy()
    reps = []
    for w in base:
        if gf2_rank(np.vstack([mat, w])) > gf2_rank(mat):
            mat = np.vstack([mat, w])
            reps.append(w)
    return np.array(reps, dtype=np.int8).reshape(-1, base.shape[1])


def logicals(h_x, h_z):
    """Return (l_x, l_z) logical operator matrices, each k x n binary."""
    l_x = quotient_basis(gf2_nullspace(h_z), h_x)
    l_z = quotient_basis(gf2_nullspace(h_x), h_z)
    return l_x, l_z


def code_info(h_x, h_z):
    """Return the number of encoded logical qubits k of the CSS code.

    X-type and Z-type stabilizer subgroups intersect only at the
    identity, so k = n - rank(h_x) - rank(h_z) regardless of overlap
    between the two row spaces.
    """
    return h_x.shape[1] - gf2_rank(h_x) - gf2_rank(h_z)
