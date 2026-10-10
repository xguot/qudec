"""Ordered statistics decoding post-processing for CSS codes.

Implements OSD-0 and two combination-sweep variants on a permuted check
matrix: the one-pass greedy sweep used in the committed benchmarks, and
the reference OSD-CS(lambda) of Roffe et al. (PRR 2020) as described by
Gu and Soleimanifar (IEEE Trans. Inf. Theory 2026): all weight-one
configurations of the non-pivot support T plus weight-two configurations
restricted to the first lambda columns of T, where lambda = 60 is the
reference setting.

The module is numpy-only so the post-processing logic runs and tests
without torch on the cluster or a laptop.
"""

import numpy as np

from qudec.codes import gf2_rref


def _rref_aug(h, s, order):
    """Return (hp, rref, pivots) of the augmented matrix [h[:, order] | s]."""
    hp = h[:, order]
    aug = np.concatenate([hp, s.reshape(-1, 1)], axis=1)
    rref, pivots = gf2_rref(aug)
    return hp, rref, pivots


def osd0(h, s, order):
    """Run OSD-0: zero the non-pivot support, solve the pivots."""
    _, rref, pivots = _rref_aug(h, s, order)
    r = len(pivots)
    e = np.zeros(h.shape[1], dtype=np.int8)
    e[pivots] = rref[:r, -1]
    corr = np.zeros(h.shape[1], dtype=np.int8)
    corr[order] = e
    return corr


def osd_cs_greedy(h, s, order):
    """Run the one-pass greedy combination sweep over non-pivot columns.

    Kept verbatim from the decoder-local implementation that produced
    the committed benchmark numbers, so those results stay reproducible.
    """
    _, rref, pivots = _rref_aug(h, s, order)
    r = len(pivots)
    e = np.zeros(h.shape[1], dtype=np.int8)
    e[pivots] = rref[:r, -1]
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


def osd_cs_exhaustive(h, s, order, lam):
    """Run the reference OSD-CS(lambda) combination sweep.

    Consider e_T = 0 (OSD-0), every weight-one e_T, and every weight-two
    e_T among the first lambda non-pivot columns in the sorted order;
    solve the pivot support by Gaussian elimination for each candidate
    and keep the lowest-weight correction, ties resolved by first found.
    """
    _, rref, pivots = _rref_aug(h, s, order)
    r = len(pivots)
    n = h.shape[1]
    pivot_set = set(pivots.tolist())
    t_cols = np.array([j for j in range(n) if j not in pivot_set],
                      dtype=np.int64)
    base = rref[:r, -1].copy()
    best_w = int(base.sum())
    best_t = []
    for j in t_cols:
        w = int((base ^ rref[:r, j]).sum()) + 1
        if w < best_w:
            best_w = w
            best_t = [int(j)]
    lam_eff = min(lam, len(t_cols))
    for i1 in range(lam_eff):
        t1 = t_cols[i1]
        for i2 in range(i1 + 1, lam_eff):
            t2 = t_cols[i2]
            w = int((base ^ rref[:r, t1] ^ rref[:r, t2]).sum()) + 2
            if w < best_w:
                best_w = w
                best_t = [int(t1), int(t2)]
    e = np.zeros(n, dtype=np.int8)
    for t in best_t:
        e[t] = 1
        base = base ^ rref[:r, t]
    e[pivots] = base
    corr = np.zeros(n, dtype=np.int8)
    corr[order] = e
    return corr


def osd_decode(h, s, order, order_kind=1, lam=None):
    """Dispatch OSD post-processing.

    lam set runs the exhaustive reference sweep; otherwise order_kind 0
    runs OSD-0 and order_kind 1 runs the one-pass greedy sweep.
    """
    if lam is not None:
        return osd_cs_exhaustive(h, s, order, lam)
    if order_kind >= 1:
        return osd_cs_greedy(h, s, order)
    return osd0(h, s, order)
