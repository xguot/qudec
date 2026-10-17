"""stim circuits and detector-error-model utilities for circuit-level noise.

The builder produces a d-round memory experiment under circuit-level
depolarizing noise: the data qubits hold the code block, each check gets
a fresh ancilla per round, and every reset, gate, idle, and measurement
is noisy at rate p. Two bases are supported: Z (data in |0>, Z logicals
tracked through the final data measurement) and X (data in |+>, X
logicals tracked).

Detector structure, basis Z: round-0 Z-check outcomes are detectors
against the |0> initial state; Z- and X-check outcomes pair across
consecutive rounds; and the final data measurement combines with the
last round of Z-check outcomes. X-checks carry no round-0 detector
because the |0> state is not an X-parity eigenstate. Basis X mirrors
this with X and Z interchanged.

stim is imported lazily so the numpy-only helpers run without it.
"""

import numpy as np


def dem_check_matrix(dem, return_probabilities=False):
    """Return (h, l) binary matrices from a detector error model.

    h: (num_detectors, num_errors), one column per independent error
    mechanism; l: (num_observables, num_errors). Accepts a stim dem or
    any iterable of instructions exposing .type and .targets_copy(), so
    the extraction tests with stubs. When return_probabilities is true,
    also return a probability vector aligned with the error columns.
    """
    m = dem.num_detectors
    k = dem.num_observables
    cols = []
    obs_rows = []
    probabilities = []
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        col = np.zeros(m, dtype=np.int8)
        obs_col = np.zeros(k, dtype=np.int8)
        for t in instr.targets_copy():
            if t.is_relative_detector_id():
                col[t.val] ^= 1
            elif t.is_logical_observable_id():
                obs_col[t.val] ^= 1
        cols.append(col)
        obs_rows.append(obs_col)
        if return_probabilities:
            probabilities.append(instr.args_copy()[0])
    if not cols:
        h = np.zeros((m, 0), dtype=np.int8)
        l = np.zeros((k, 0), dtype=np.int8)
    else:
        h = np.stack(cols, axis=1).astype(np.int8)
        l = np.stack(obs_rows, axis=1).astype(np.int8)
    if return_probabilities:
        return h, l, np.asarray(probabilities, dtype=float)
    return h, l


def detector_count(h_x, h_z, d, basis):
    """Return the expected detector count of the memory circuit."""
    m_z = h_z.shape[0]
    m_x = h_x.shape[0]
    if basis == "Z":
        return d * m_z + (d - 1) * m_x + m_z
    return d * m_x + (d - 1) * m_z + m_x


def build_memory_circuit(h_x, h_z, l_x, l_z, d, p, basis="Z"):
    """Return the stim circuit for a d-round noisy memory experiment."""
    import stim

    if basis not in ("X", "Z"):
        raise ValueError("basis must be X or Z")
    if not isinstance(d, (int, np.integer)) or d < 1:
        raise ValueError("d must be a positive integer")
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be a probability between 0 and 1")
    h_x = np.asarray(h_x)
    h_z = np.asarray(h_z)
    l_x = np.asarray(l_x)
    l_z = np.asarray(l_z)
    n = h_x.shape[1]
    m_z = h_z.shape[0]
    m_x = h_x.shape[0]
    c = stim.Circuit()

    def rec(idx):
        """Convert an absolute measurement index to a stim rec lookback."""
        return stim.target_rec(idx - c.num_measurements)
    for q in range(n):
        c.append("R", [q])
        c.append("DEPOLARIZE1", [q], p)
    if basis == "X":
        for q in range(n):
            c.append("H", [q])
            c.append("DEPOLARIZE1", [q], p)
    meas_z = []
    meas_x = []
    for r in range(d):
        for q in range(n):
            c.append("DEPOLARIZE1", [q], p)
        rec0 = c.num_measurements
        for i in range(m_z):
            a = n + r * (m_z + m_x) + i
            c.append("R", [a])
            c.append("DEPOLARIZE1", [a], p)
            for q in np.nonzero(h_z[i])[0]:
                c.append("CNOT", [int(q), a])
                c.append("DEPOLARIZE2", [int(q), a], p)
            c.append("M", [a], p)
        for i in range(m_x):
            a = n + r * (m_z + m_x) + m_z + i
            c.append("RX", [a])
            c.append("DEPOLARIZE1", [a], p)
            for q in np.nonzero(h_x[i])[0]:
                c.append("CNOT", [a, int(q)])
                c.append("DEPOLARIZE2", [a, int(q)], p)
            c.append("H", [a])
            c.append("DEPOLARIZE1", [a], p)
            c.append("M", [a], p)
        for i in range(m_z):
            meas_z.append(rec0 + i)
        for i in range(m_x):
            meas_x.append(rec0 + m_z + i)
        if r == 0:
            if basis == "Z":
                for i in range(m_z):
                    c.append("DETECTOR", [rec(meas_z[i])])
            else:
                for i in range(m_x):
                    c.append("DETECTOR", [rec(meas_x[i])])
        else:
            for i in range(m_z):
                c.append("DETECTOR",
                         [rec(meas_z[-m_z + i]),
                          rec(meas_z[-2 * m_z + i])])
            for i in range(m_x):
                c.append("DETECTOR",
                         [rec(meas_x[-m_x + i]),
                          rec(meas_x[-2 * m_x + i])])
    for q in range(n):
        c.append("DEPOLARIZE1", [q], p)
        if basis == "X":
            c.append("H", [q])
            c.append("DEPOLARIZE1", [q], p)
        c.append("M", [q], p)
    rec_data = c.num_measurements - n
    if basis == "Z":
        for i in range(m_z):
            det = [rec(meas_z[-m_z + i])]
            for q in np.nonzero(h_z[i])[0]:
                det.append(rec(rec_data + int(q)))
            c.append("DETECTOR", det)
        for logical_id, row in enumerate(l_z):
            obs = [rec(rec_data + int(q))
                   for q in np.nonzero(row)[0]]
            c.append("OBSERVABLE_INCLUDE", obs, logical_id)
    else:
        for i in range(m_x):
            det = [rec(meas_x[-m_x + i])]
            for q in np.nonzero(h_x[i])[0]:
                det.append(rec(rec_data + int(q)))
            c.append("DETECTOR", det)
        for logical_id, row in enumerate(l_x):
            obs = [rec(rec_data + int(q))
                   for q in np.nonzero(row)[0]]
            c.append("OBSERVABLE_INCLUDE", obs, logical_id)
    return c


def sample_memory(circuit, shots, seed=0):
    """Return (dets, obs) sampled from the compiled detector sampler."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    return sampler.sample(shots, separate_observables=True)
