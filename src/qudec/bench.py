"""Code-capacity logical error rate benchmark for decoders."""

import numpy as np

from qudec.noise import sample_iid_errors


def benchmark_iid(decoder, p, shots, model="depolarizing", seed=0):
    """Benchmark a decoder under i.i.d. noise on the raw code.

    Decoding succeeds when the correction differs from the true error
    only by a stabilizer, checked against the logical operator matrices.
    Return a dict with logical error rates and invalid-correction counts.
    """
    n = decoder.h_x.shape[1]
    e_x, e_z = sample_iid_errors(n, p, shots, model, seed)
    sx = (decoder.h_z @ e_x.T) % 2
    sz = (decoder.h_x @ e_z.T) % 2
    c_x, c_z = decoder.decode_corrections(
        sx.T.astype(np.int8), sz.T.astype(np.int8))
    ok_x = np.all(((decoder.l_z @ (e_x.T ^ c_x.T)) % 2) == 0, axis=0)
    ok_z = np.all(((decoder.l_x @ (e_z.T ^ c_z.T)) % 2) == 0, axis=0)
    invalid_x = int(np.any(((decoder.h_z @ c_x.T) % 2) != sx, axis=0).sum())
    invalid_z = int(np.any(((decoder.h_x @ c_z.T) % 2) != sz, axis=0).sum())
    return {
        "ler": float(1 - (ok_x & ok_z).mean()),
        "ler_x": float(1 - ok_x.mean()),
        "ler_z": float(1 - ok_z.mean()),
        "invalid_x": invalid_x,
        "invalid_z": invalid_z,
        "shots": shots,
        "p": p,
    }
