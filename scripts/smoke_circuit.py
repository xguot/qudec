"""Validation smoke tests for the circuit-level pipeline.

Runs on the cluster where stim is installed, before any benchmark:
1. Noiseless circuit: every detector and observable is zero.
2. Detector-count sanity against the structural formula.
3. Decoder sanity at low noise: logical error rate and detector rates
   are nonzero but far from one, and the decoders return valid shapes.

Exit nonzero on any failure so the slurm job aborts before benching.
"""

import sys

import numpy as np

from qudec.circuit import (
    build_memory_circuit,
    dem_check_matrix,
    detector_count,
    sample_memory,
)
from qudec.circuit_decoders import AdmmOsdDemDecoder, BpOsdDemDecoder
from qudec.codes import gross_code, logicals, medium_code, steane_code


def check(cond, msg):
    if not cond:
        print("SMOKE FAIL:", msg)
        sys.exit(1)
    print("smoke ok:", msg)


def main():
    h_x, h_z = steane_code()
    l_x, l_z = logicals(h_x, h_z)
    d = 3

    circ0 = build_memory_circuit(h_x, h_z, l_x, l_z, d, 0.0, basis="Z")
    check(circ0.num_detectors == detector_count(h_x, h_z, d, "Z"),
          f"detector count Z ({circ0.num_detectors} "
          f"vs {detector_count(h_x, h_z, d, 'Z')})")
    dets0, obs0 = sample_memory(circ0, 200, seed=0)
    check(int(dets0.sum()) == 0, "noiseless detectors all zero")
    check(int(obs0.sum()) == 0, "noiseless observables all zero")

    circx = build_memory_circuit(h_x, h_z, l_x, l_z, d, 0.0, basis="X")
    check(circx.num_detectors == detector_count(h_x, h_z, d, "X"),
          "detector count X")
    detsx, obsx = sample_memory(circx, 200, seed=0)
    check(int(detsx.sum()) == 0, "noiseless X-basis detectors all zero")
    check(int(obsx.sum()) == 0, "noiseless X-basis observables all zero")

    h_x, h_z = medium_code()
    l_x, l_z = logicals(h_x, h_z)
    circ = build_memory_circuit(h_x, h_z, l_x, l_z, d, 0.02, basis="Z")
    dem = circ.detector_error_model(decompose_errors=True)
    h, l = dem_check_matrix(dem)
    check(h.shape[0] == circ.num_detectors, "dem detector rows match")
    check(l.shape[0] == circ.num_observables, "dem observable rows match")
    check(h.shape[1] > 0, "dem has error mechanisms")
    dets, obs = sample_memory(circ, 200, seed=1)
    check(0 < int(dets.sum()) < dets.size, "detectors fire at p=0.02")
    for name, dec in [
            ("bp", BpOsdDemDecoder(h, l, p=0.02)),
            ("admm", AdmmOsdDemDecoder(h, l))]:
        pred = dec.decode_batch(dets)
        check(pred.shape == obs.shape, f"{name} prediction shape")
        ler = float((pred != obs).any(axis=1).mean())
        check(0.0 < ler < 0.5, f"{name} LER sane at p=0.02 ({ler:.4f})")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
