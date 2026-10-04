# qudec

GPU-accelerated quantum error correction decoders in PyTorch.

Phase one delivers the offline-verifiable core: CSS code constructions
(bivariate bicycle, Steane, repetition), GF(2) linear algebra, a batched
min-sum belief propagation pass with ordered statistics decoding
(BP + OSD-0/OSD-CS) post-processing, an i.i.d. code-capacity benchmark,
and a `sinter`-protocol `decode_batch` interface.

## Layout

- `src/qudec/codes.py` — code constructions, GF(2) rank/RREF/nullspace,
  logical operator bases
- `src/qudec/bposd.py` — batched min-sum BP (torch) + OSD post-processing
- `src/qudec/lp.py` — LP decoding (Gu-Soleimanifar TIT 2026) with OSD
  post-processing
- `src/qudec/noise.py` — i.i.d. X/Z/depolarizing error sampling
- `src/qudec/bench.py` — code-capacity logical error rate benchmark
- `tests/` — exhaustive checks against brute-force MLD on small codes

## Install and test

    pip install -e .
    PYTHONPATH=src python3 -m unittest discover -s tests -v

Reference sweep for paper parity (adds `ldpc` + `bposd`):

    pip install -e ".[reference]"
    python -u scripts/bench_gross_ref.py

## Quickstart

    import numpy as np
    from qudec.bposd import BpOsdDecoder
    from qudec.bench import benchmark_iid
    from qudec.codes import gross_code, logicals

    h_x, h_z = gross_code()          # [[144, 12, 12]] bivariate bicycle
    l_x, l_z = logicals(h_x, h_z)
    dec = BpOsdDecoder(h_x, h_z, l_x, l_z, p_x=0.02, p_z=0.02)
    res = benchmark_iid(dec, p=0.03, shots=200, seed=0)
    print(res)

## Status and roadmap

- Phase one (this commit): BP + OSD core, verified offline against
  brute-force decoding on small codes and the repetition-code MLD rate.
- Phase two (Rivanna): reproduce `ldpc`/`bposd` accuracy on bivariate
  bicycle codes, batch OSD elimination on CUDA, wall-clock benchmarks.
- Phase three (Rivanna): `stim` + `sinter` circuit-level noise
  reproduction of published logical error rates.
- Phase four: ADMM decoder on the decoding LP as an OSD replacement,
  exploiting code degeneracy.

`gen_data.py` at the repo root is a leftover from the earlier QPT
exploration and is unrelated to the decoder work.
