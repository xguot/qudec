"""Shared min-sum belief propagation and channel-prior validation."""

import numbers

import numpy as np


def probability_vector(probabilities, n, name="probabilities"):
    """Validate a scalar or length-n vector of probabilities in (0, 1)."""
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.all(np.isfinite(values) & (values > 0) & (values < 1)):
        raise ValueError(f"{name} must contain finite probabilities in (0, 1)")
    if values.ndim == 0:
        values = np.full(n, float(values), dtype=np.float64)
    elif values.shape == (n,):
        values = values.copy()
    else:
        raise ValueError(f"{name} must be a scalar or a vector of length {n}")
    return values


def channel_llrs(probabilities, n, name="probabilities"):
    """Return per-column log likelihood ratios for valid channel priors."""
    p = probability_vector(probabilities, n, name)
    return np.log1p(-p) - np.log(p)


def min_sum_bp(h, s, llr, max_iter=30, alpha=0.9):
    """Return min-sum posterior LLRs for syndromes s, shape (batch, n).

    The check-to-variable minimum excludes exactly the receiving edge,
    including when several incoming magnitudes are equal. Degree-one
    checks impose their syndrome with a finite message cap of 1e4.
    Empty checks are permitted only with zero syndrome. With no checks,
    no variables, or zero iterations, the channel LLRs are returned.
    """
    import torch

    h = np.asarray(h)
    s = np.asarray(s)
    if h.ndim != 2 or not np.all((h == 0) | (h == 1)):
        raise ValueError("h must be a two-dimensional binary check matrix")
    m, n = h.shape
    if s.ndim != 2 or s.shape[1] != m or not np.all((s == 0) | (s == 1)):
        raise ValueError(f"s must be a binary syndrome array with {m} columns")
    if (not isinstance(max_iter, numbers.Integral)
            or isinstance(max_iter, bool) or max_iter < 0):
        raise ValueError("max_iter must be a nonnegative integer")
    if alpha is not None and (not np.isscalar(alpha)
                              or not np.isfinite(alpha) or alpha < 0):
        raise ValueError("alpha must be finite and nonnegative, or None")
    llr = np.asarray(llr, dtype=np.float64)
    if llr.shape != (n,) or not np.all(np.isfinite(llr)):
        raise ValueError(f"llr must be a finite vector of length {n}")
    empty = ~np.any(h, axis=1)
    if np.any(s[:, empty]):
        raise ValueError("an empty check cannot have nonzero syndrome")

    batch = s.shape[0]
    l_ch = torch.as_tensor(llr, dtype=torch.float32)
    if max_iter == 0 or m == 0 or n == 0 or batch == 0:
        return l_ch.unsqueeze(0).expand(batch, n).clone()

    edges = torch.as_tensor(h != 0)
    syndromes = torch.as_tensor(s, dtype=torch.float32)
    cap = 1e4
    l_vc = torch.where(edges.unsqueeze(0), l_ch.view(1, 1, n), 0.0)
    for t in range(max_iter):
        scale = alpha if alpha is not None else 1.0 - 2.0 ** (-(t + 1))
        abs_l = torch.where(edges.unsqueeze(0), l_vc.abs(), float("inf"))
        minima, indices = abs_l.topk(min(2, n), dim=2, largest=False)
        first = minima[:, :, 0]
        second = minima[:, :, 1] if n > 1 else torch.full_like(first, cap)
        mag = first.unsqueeze(2).expand_as(l_vc).clone()
        mag.scatter_(2, indices[:, :, :1], second.unsqueeze(2))
        mag = torch.where(edges.unsqueeze(0), mag.clamp(max=cap), 0.0)
        neg = (l_vc < 0) & edges.unsqueeze(0)
        parity = (neg.sum(dim=2) + syndromes) % 2
        sign = (1 - 2 * parity).unsqueeze(2) * (1 - 2 * neg.float())
        m_cv = sign * mag * scale
        sum_in = m_cv.sum(dim=1)
        l_vc = torch.where(
            edges.unsqueeze(0), l_ch.view(1, 1, n) + sum_in.unsqueeze(1) - m_cv,
            0.0)
    return l_ch.view(1, n) + m_cv.sum(dim=1)
