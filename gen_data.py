"""Generate synthetic Pauli-measurement data for quantum process tomography.

Construct a noisy N-qubit channel (depolarizing unitary), compute its Choi
matrix, then simulate the standard informationally complete QPT protocol:
4^N input states x 3^N measurement settings x 2^N outcomes per setting.

Pure numpy, no external quantum libraries.
"""

import numpy as np

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

PLUS_STATES = {
    "Z": {"0": np.array([1, 0], dtype=complex),
          "1": np.array([0, 1], dtype=complex)},
    "X": {"+": np.array([1, 1], dtype=complex) / np.sqrt(2),
          "-": np.array([1, -1], dtype=complex) / np.sqrt(2)},
    "Y": {"+": np.array([1, 1j], dtype=complex) / np.sqrt(2),
          "-": np.array([1, -1j], dtype=complex) / np.sqrt(2)},
}

INPUT_LABELS = [("Z", "0"), ("Z", "1"), ("X", "+"), ("Y", "+")]


def cnot_unitary():
    """Return the 4x4 CNOT unitary in the |00,01,10,11> basis."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)


def depolarized_choi(unitary, p):
    """Return the Choi matrix of (1-p) unitary channel + p depolarizing."""
    d = unitary.shape[0]
    vec_u = unitary.reshape(-1, order="F")
    projector = np.outer(vec_u, vec_u.conj())
    return (1 - p) * projector + (p / d) * np.eye(d * d, dtype=complex)


def input_states(n_qubits):
    """Return the 4^N product input states, one per qubit from each eigenbasis."""
    states = []
    for combo in np.ndindex(*([4] * n_qubits)):
        ket = np.array([1], dtype=complex)
        for basis_idx in combo:
            basis, label = INPUT_LABELS[basis_idx]
            ket = np.kron(ket, PLUS_STATES[basis][label])
        states.append(ket)
    return states


def measurement_effects(n_qubits):
    """Return all (setting, outcome) projector pairs for 3^N Pauli settings.

    Each setting is a tensor product of X, Y, Z per qubit; each outcome is a
    string of +/- eigenstate labels giving a rank-1 projector.
    """
    effects = []
    for basis_combo in np.ndindex(*([3] * n_qubits)):
        basis_names = ["XYZ"[i] for i in basis_combo]
        for labels in np.ndindex(*([2] * n_qubits)):
            ket = np.array([1], dtype=complex)
            for bname, lbl in zip(basis_names, labels):
                sign = "+" if lbl == 0 else "-"
                ket = np.kron(ket, PLUS_STATES[bname][sign])
            effects.append((basis_names, labels, np.outer(ket, ket.conj())))
    return effects


def build_measurement_matrix(n_qubits, use_outcomes):
    """Return the linear map A such that y = A @ vec(Choi).

    With use_outcomes=True each row is vec(rho_j^T (x) Pi_{k,b}) for a full
    outcome record (24^N rows, informationally complete). With False, rows
    are vec(rho_j^T (x) P_k) for Pauli expectation values only (12^N rows,
    not informationally complete for N >= 2).
    """
    d = 2 ** n_qubits
    rho_in = [np.outer(k, k.conj()) for k in input_states(n_qubits)]
    rows = []
    for rho in rho_in:
        if use_outcomes:
            for _, _, proj in measurement_effects(n_qubits):
                rows.append(np.kron(rho.T, proj))
        else:
            for basis_combo in np.ndindex(*([3] * n_qubits)):
                op = np.array([1], dtype=complex)
                for i in basis_combo:
                    op = np.kron(op, PAULI["XYZ"[i]])
                rows.append(np.kron(rho.T, op))
    return np.array([r.conj().reshape(-1, order="F") for r in rows])


def simulate_counts(choi, n_qubits, shots):
    """Sample multinomial outcome counts for every input state and setting."""
    d = 2 ** n_qubits
    choi_t = choi.reshape(d, d, d, d)
    counts = {}
    for idx_j, rho in enumerate(input_states(n_qubits)):
        for basis_combo in np.ndindex(*([3] * n_qubits)):
            basis_names = ["XYZ"[i] for i in basis_combo]
            probs = []
            for labels in np.ndindex(*([2] * n_qubits)):
                ket = np.array([1], dtype=complex)
                for bname, lbl in zip(basis_names, labels):
                    sign = "+" if lbl == 0 else "-"
                    ket = np.kron(ket, PLUS_STATES[bname][sign])
                proj = np.outer(ket, ket.conj())
                m = np.kron(rho.T, proj).reshape(d, d, d, d, order="F")
                probs.append(np.real(np.einsum("ijkl,ijkl->", choi_t, m)))
            probs = np.array(probs)
            probs = np.clip(probs / probs.sum(), 0, None)
            counts[(idx_j, basis_combo)] = np.random.multinomial(shots, probs)
    return counts


def fidelity(choi_true, choi_est):
    """Return the Choi-state process fidelity between two CP maps."""
    sq = np.linalg.eigvalsh(choi_true)
    if sq.min() < -1e-10:
        raise ValueError("true Choi not PSD")
    root = np.linalg.eigvalsh(choi_true)
    root[root < 0] = 0
    w, v = np.linalg.eigh(choi_true)
    w[w < 0] = 0
    root = (v * np.sqrt(w)) @ v.conj().T
    inner = root @ choi_est @ root
    val = np.real(np.trace(np.linalg.eigvalsh(inner) ** 0.5))
    return val


def main():
    n_qubits = 2
    p_noise = 0.05
    shots = 1000
    rng = np.random.default_rng(7)

    unitary = cnot_unitary()
    choi = depolarized_choi(unitary, p_noise)
    d = 2 ** n_qubits

    tp_error = np.linalg.norm(
        choi.reshape(d, d, d, d).trace(axis1=2, axis2=3) - np.eye(d))
    evals = np.linalg.eigvalsh(choi)
    print(f"N={n_qubits}  Choi {d*d}x{d*d}  depolarizing p={p_noise}")
    print(f"TP error ||Tr_out(Lambda) - I|| = {tp_error:.3e}")
    print(f"min eigenvalue of Lambda      = {evals.min():.3e}  (CP ok if >= 0)")

    a_out = build_measurement_matrix(n_qubits, use_outcomes=True)
    a_exp = build_measurement_matrix(n_qubits, use_outcomes=False)
    print(f"A (outcome record): {a_out.shape}  rank {np.linalg.matrix_rank(a_out)}"
          f"  (IC needs {d*d})")
    print(f"A (exp values):     {a_exp.shape}  rank {np.linalg.matrix_rank(a_exp)}"
          f"  (not IC for N>=2)")

    counts = simulate_counts(choi, n_qubits, shots)
    y = []
    for idx_j in range(4 ** n_qubits):
        for basis_combo in np.ndindex(*([3] * n_qubits)):
            y.extend(counts[(idx_j, basis_combo)] / shots)
    y = np.array(y)
    print(f"measurement record: {y.size} frequencies from {shots} shots/setting")

    np.savez("data/qpt_n2_p005.npz",
             y=y, a=a_out, choi=choi, n_qubits=n_qubits, shots=shots)
    print("saved data/qpt_n2_p005.npz")


if __name__ == "__main__":
    main()
