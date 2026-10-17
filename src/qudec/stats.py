"""Reproducible comparisons from exact failure counts and paired outcomes."""

import math

import numpy as np
from scipy.stats import binomtest


def two_proportion(failures_a, shots_a, failures_b, shots_b):
    """Pooled two-sided z test; positive delta/z means B fails more.

    This is an independent-sample descriptive comparison. Use McNemar
    when the decoders saw the same shots; it accounts for pairing.
    """
    for failures, shots in ((failures_a, shots_a), (failures_b, shots_b)):
        if shots <= 0 or not 0 <= failures <= shots:
            raise ValueError("require positive shot counts and 0 <= failures <= shots")
    delta = failures_b / shots_b - failures_a / shots_a
    pooled = (failures_a + failures_b) / (shots_a + shots_b)
    se = math.sqrt(pooled * (1 - pooled) * (1/shots_a + 1/shots_b))
    z = delta / se if se else 0.0
    return {"delta": delta, "z": z, "p_value": math.erfc(abs(z)/math.sqrt(2)),
            "failures_a": int(failures_a), "shots_a": int(shots_a),
            "failures_b": int(failures_b), "shots_b": int(shots_b)}


def mcnemar(failed_a, failed_b):
    """Exact two-sided paired test with discordance counts and signed z.

    z is the uncorrected normal approximation; p_value uses the exact
    binomial test on discordant shots. No multiple-testing correction is
    applied here. A_only means A failed and B succeeded, and vice versa.
    """
    a, b = np.asarray(failed_a, bool), np.asarray(failed_b, bool)
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("paired outcomes must be nonempty equal-length vectors")
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    discordant = a_only + b_only
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant,
            "shots": int(a.size), "delta": float(b.mean() - a.mean()),
            "z": (b_only-a_only)/math.sqrt(discordant) if discordant else 0.0,
            "p_value": float(binomtest(b_only, discordant, 0.5).pvalue)
            if discordant else 1.0}


def decode_outcomes(record, field="failure"):
    """Validate and decode a stored 0/1 outcome string."""
    value = record["outcomes"][field]
    if not isinstance(value, str) or len(value) != record["shots"] or set(value) - {"0", "1"}:
        raise ValueError("invalid per-shot outcome record")
    result = np.frombuffer(value.encode("ascii"), dtype=np.uint8) == ord("1")
    if int(result.sum()) != record[field]:
        raise ValueError("outcome count does not match saved failure count")
    return result
