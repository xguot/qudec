"""Small deterministic validation of both circuit memory bases and decoders.

Run on the cluster before benchmarking. This checks exact noiseless
invariants, then samples a fixed noisy Steane circuit and validates decoder
shapes and output bits. A short smoke run cannot establish a useful LER.
"""

import numpy as np

from qudec.circuit import (
    build_memory_circuit, dem_check_matrix, detector_count, sample_memory,
)
from qudec.circuit_decoders import AdmmOsdDemDecoder, BpOsdDemDecoder
from qudec.codes import logicals, steane_code


def check(cond, msg):
    if not cond:
        raise AssertionError(f"SMOKE FAIL: {msg}")
    print("smoke ok:", msg)


def main():
    h_x, h_z = steane_code()
    l_x, l_z = logicals(h_x, h_z)
    d = 2
    for basis in ("Z", "X"):
        circ0 = build_memory_circuit(h_x, h_z, l_x, l_z, d, 0, basis)
        check(circ0.num_detectors == detector_count(h_x, h_z, d, basis),
              f"detector count {basis}")
        check(circ0.num_observables == len(l_x), f"logical count {basis}")
        dets0, obs0 = sample_memory(circ0, 32, seed=0)
        check(not dets0.any() and not obs0.any(), f"noiseless {basis} samples")
        h0, l0, p0 = dem_check_matrix(circ0.detector_error_model(), True)
        check(h0.shape == (circ0.num_detectors, 0) and
              l0.shape == (circ0.num_observables, 0) and p0.shape == (0,),
              f"noiseless {basis} DEM has no error mechanisms")

        circ = build_memory_circuit(h_x, h_z, l_x, l_z, d, 0.01, basis)
        h, l, probabilities = dem_check_matrix(circ.detector_error_model(), True)
        check(h.shape[0] == circ.num_detectors and
              l.shape[0] == circ.num_observables and h.shape[1] > 0 and
              len(probabilities) == h.shape[1], f"noisy {basis} DEM dimensions")
        dets, obs = sample_memory(circ, 8, seed=1)
        check(dets.any(), f"noisy {basis} samples contain detection events")
        for name, decoder in [
                ("bp", BpOsdDemDecoder(h, l, max_iter=10,
                                        probabilities=probabilities)),
                ("admm", AdmmOsdDemDecoder(h, l, max_iter=20,
                                            probabilities=probabilities))]:
            # The benchmark bounds batches too; keep smoke solver memory small.
            pred = np.concatenate([decoder.decode_batch(dets[i:i + 2])
                                   for i in range(0, len(dets), 2)])
            check(pred.shape == obs.shape and np.isin(pred, [0, 1]).all(),
                  f"{name} {basis} prediction shape and bits")
            print(f"smoke diagnostic: {name} {basis} "
                  f"failures={int(np.any(pred != obs, axis=1).sum())}/{len(obs)}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
