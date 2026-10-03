"""I.i.d. error sampling for code-capacity benchmarks."""

import numpy as np


def sample_iid_errors(n, p, shots, model="depolarizing", seed=0):
    """Return (e_x, e_z) binary error arrays of shape (shots, n).

    model 'x' applies bit flips with probability p, 'z' phase flips,
    'depolarizing' applies X, Y, Z each with probability p / 3.
    """
    rng = np.random.default_rng(seed)
    if model in ("x", "bit_flip"):
        e_x = (rng.random((shots, n)) < p).astype(np.int8)
        e_z = np.zeros((shots, n), dtype=np.int8)
    elif model in ("z", "phase_flip"):
        e_x = np.zeros((shots, n), dtype=np.int8)
        e_z = (rng.random((shots, n)) < p).astype(np.int8)
    elif model == "depolarizing":
        r = rng.random((shots, n))
        e_x = (r < 2 * p / 3).astype(np.int8)
        e_z = ((r >= p / 3) & (r < p)).astype(np.int8)
    else:
        raise ValueError(f"unknown noise model {model}")
    return e_x, e_z
