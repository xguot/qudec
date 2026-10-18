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

Optional external-reference sweep (adds `ldpc` + `bposd`):

    pip install -e ".[reference]"
    python -u scripts/bench_gross_ref.py

## HPC (Rivanna)

Campaign **21501046 completed all 75 tasks** on October 17, 2026.
Its [audited report](docs/results/21501046/report.json) and
[research summary](research_summary.md) contain the corrected results.
There is no need to resubmit this campaign for the reporting and cleanup
changes in the repository.

On Rivanna, update an existing Git checkout after the desired commits
have been published to its remote:

    git pull --ff-only

For a new campaign, run `bash hpc/setup.sh` once in that checkout, then
submit with the single Slurm entry point:

    sbatch hpc/rerun_all.slurm

The initial job freezes a source snapshot, runs the tests and pilot,
and submits dependent tuning, setting selection, benchmark workers,
and a final report. Confirmation waits for successful tuning and uses
independent seeds. Results, frozen settings, source provenance, and worker
logs live under `results/reruns/<initial_job_id>/`. The default worker
limit is four simultaneous jobs. The optional
`bash hpc/sync.sh pull` command retrieves results to the local checkout.

The default campaign includes both codes at six phenomenological rounds,
three error rates (0.005, 0.010, 0.020), and both greedy OSD and OSD-CS(60).
Each confirmation cell contains four independent 2500-shot shards,
for 10,000 shots per decoder, with paired BP, plain ADMM, and weighted
ADMM outcomes. Separate tasks cover a held-out replication, exploratory
LDR comparisons, code-capacity checks, and a preliminary circuit sweep.
The completed run used these defaults: 10,000 shots per decoder in each
of the 12 primary cells. The optional 100,000-shot escalation and
external-reference comparison were not run. See
[the campaign plan](docs/plan.md) for scope, outcomes, and follow-ups.

Common overrides, set before `sbatch`:

| Variable | Default | Effect |
| --- | --- | --- |
| `QUDEC_CONFIRM_SEEDS` | `4` | Set to `40` for 100,000 shots per confirmation cell. |
| `QUDEC_CONFIRM_SHOTS` | `2500` | Shots in each confirmation shard. |
| `QUDEC_BATCH_SIZE` | `32` | Bound decoder memory independently of total shots. |
| `QUDEC_MAX_PARALLEL` | `4` | Maximum simultaneous benchmark workers. |
| `QUDEC_INCLUDE_CIRCUIT` | `1` | Set to `0` to omit the preliminary circuit sweep. |
| `QUDEC_INCLUDE_REFERENCE` | `0` | Set to `1` to include the external `ldpc`/`bposd` comparison. |
| `QUDEC_TASK_TIME` | `12:00:00` | Time limit requested for benchmark worker jobs. |
| `QUDEC_PYTHON` | auto-detected | Python executable for the campaign. |

Inspect the plan locally, or resume an existing campaign on Rivanna:

    .venv/bin/python scripts/rerun_all.py plan
    sbatch hpc/rerun_all.slurm resume /absolute/path/to/results/reruns/<initial_job_id>

Resume after the previous jobs have stopped. Completed cells are checked
and skipped; partial cells are replaced. `QUDEC_TASK_TIME` and
`QUDEC_MAX_PARALLEL` may change for a retry, while samples, batch size,
source, and decoder settings remain frozen.

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
These are repaired in the current implementation and were exercised by
campaign 21501046. Its source snapshot, artifact hashes, paired outcomes,
and complete task coverage were audited. Historical comparisons that
predate the fixes are superseded by the
[corrected results](research_summary.md).

At [[72,12,6]], p = 0.020, with greedy OSD, weighted ADMM recorded
294 failures versus BP's 339 in 10,000 paired shots; the primary
24-comparison Holm-adjusted p value is 0.03746. No other primary
comparison passed that correction. The separate held-out cohort uses
OSD-CS(60), so it does not independently replicate the greedy result.
The preliminary circuit sweep favors BP in all four tested cells.

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

Research follow-ups include an independent replication of the greedy
[[72,12,6]], p = 0.020 result, optional higher-statistics [[144,12,12]]
comparisons, and circuit-specific tuning. The bibliography still needs
completion. These are separate from the completed default campaign;
circuit results remain preliminary.
