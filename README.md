# qudec

GPU-accelerated quantum error correction decoders in PyTorch.

The implementation includes CSS code constructions
(bivariate bicycle, Steane, repetition), GF(2) linear algebra, a batched
min-sum belief propagation pass with ordered statistics decoding
(BP + OSD-0/OSD-CS) post-processing, an i.i.d. code-capacity benchmark,
and an unpacked-array `decode_batch` observable-prediction interface.
Circuit experiments sample directly through Stim; a native Sinter adapter
is not provided.

## Layout

- `src/qudec/codes.py` — code constructions, GF(2) rank/RREF/nullspace,
  logical operator bases
- `src/qudec/bposd.py` — batched min-sum BP (torch) + OSD post-processing
- `src/qudec/lp.py` — LP decoding (Gu-Soleimanifar TIT 2026) with OSD
  post-processing
- `src/qudec/admm.py` — batched ADMM solver for the parity-polytope
  relaxation (Gu-Soleimanifar ISIT 2026)
- `src/qudec/noise.py` — i.i.d. X/Z/depolarizing error sampling
- `src/qudec/phenom.py` — phenomenological noise: per-round data and
  measurement errors on the time-expanded detector system
- `src/qudec/bench.py` — code-capacity logical error rate benchmark
- `tests/` — exhaustive checks against brute-force MLD on small codes

## Install and test

Requires Python 3.10 or newer. Install the circuit extra to run the
Stim integration tests and smoke validation.

    pip install -e ".[circuit]"
    PYTHONPATH=src python3 -m unittest discover -s tests -v

Reference sweep for paper parity (adds `ldpc` + `bposd`):

    pip install -e ".[reference]"
    python -u scripts/bench_gross_ref.py

## HPC (Rivanna)

Sync code with `bash hpc/sync.sh push`, set up the environment once with
`bash hpc/setup.sh`, then submit `hpc/run_test_bench.slurm` (tests plus
code-capacity sweeps) or `hpc/run_phenom.slurm` (phenomenological noise).
Pull results back with `bash hpc/sync.sh pull`.

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

## Status and reproducibility

The code includes BP+OSD, exact LP+OSD, batched ADMM variants,
phenomenological benchmarks, tuning/confirmation jobs, and a preliminary
Stim memory circuit. ADMM selects CUDA when available; BP and OSD run on
CPU. Timing comparisons must include the device and software environment.

A correctness audit found tied-minimum BP messages, truncated ADMM
projections, unmatched BP measurement priors, and circuit-pipeline bugs.
These are repaired in the current implementation. **Previously recorded
in-tree decoder comparisons require rerunning; they do not establish an
ADMM advantage or deficit.** Historical numbers remain in
research_summary.md and docs/paper_draft.md for provenance.

New phenomenological runs default to a perfect terminal syndrome after
noisy rounds. The explicit legacy open boundary is a different protocol;
results must not be pooled across boundaries. Decoding uses bounded
batches, matched BP data/measurement priors, and per-shot failure output
for paired analysis. See `python scripts/bench_phenom.py --help` and
`python scripts/analyze_phenom.py --help` for output and comparison options.
Changing the batch size preserves sampled shots within a fixed run.

Circuit validation and a small benchmark:

    PYTHONPATH=src python scripts/smoke_circuit.py
    PYTHONPATH=src python scripts/bench_circuit.py 72 3 100 0.001 --basis Z

Remaining research work: rerun the corrected baseline and tuning grid,
replicate the promising historical setting with independent seeds,
complete larger-code/high-statistics comparisons, and integrate the
validated results and full bibliography into the manuscript. Circuit
results remain preliminary and optional in the submission plan.

`gen_data.py` at the repo root is a leftover from the earlier QPT
exploration and is unrelated to the decoder work.
