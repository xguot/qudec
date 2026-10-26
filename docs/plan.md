# Corrected campaign: results and follow-ups

The correctness audit found tied-minimum BP messages, truncated ADMM
projections, unmatched BP measurement priors, and circuit-pipeline bugs.
These are repaired, with regression tests. Campaign **21501046 completed
all 75 tasks** on October 17, 2026. Its source snapshot, manifest,
settings, artifact hashes, and paired outcomes passed the audit. The
[tracked report](results/21501046/report.json) and
[research summary](../research_summary.md) supersede the historical
decoder comparisons.

All new phenomenological runs use full parity projection, matched
per-column BP priors, bounded decoder batches, and a perfect terminal
syndrome. Per-shot failures support paired inference. The historical
open-boundary metric is a separate protocol and is excluded from this
campaign.

Campaign 21501046 completed four tuning tasks and 71 benchmark tasks:
seven code capacity, eight exploration, 48 confirmation, four held-out
replication, and four preliminary circuit cells. Each of the 12 primary
confirmation cells has 10,000 shots per decoder. Cluster validation
passed 116 tests and the circuit smoke checks; non-circuit records have
zero invalid corrections and residual data syndromes. No resubmission is
needed for that completed baseline campaign.

## One submission for a new or higher-statistics campaign

On Rivanna, update the Git checkout after the desired commits have been
published to its remote:

    git pull --ff-only

Run `bash hpc/setup.sh` once in that checkout. To launch a campaign,
submit via the single Slurm entry point (`hpc/rerun_all.slurm`):

    # Default 10,000-shot campaign (now includes both greedy and CS(60) replication)
    sbatch hpc/rerun_all.slurm

    # High-statistics 100,000-shot confirmation + 100,000-shot replication + reference
    sbatch hpc/rerun_all.slurm high_stat

    # Intermediate 25,000-shot confirmation + 10,000-shot replication + reference
    sbatch hpc/rerun_all.slurm followup

This is the only Slurm entry point. It creates an immutable source
snapshot and a campaign directory under
`results/reruns/<initial_job_id>/`, with the manifest, frozen settings,
outputs, provenance, and worker logs. The initial GPU job runs validation
and a pilot before submitting dependent stages. A failed gate prevents
the subsequent research stage from running.

## Stages and default scope

1. **Validation and pilot.** Run the unit suite and circuit smoke checks,
   verify the GPU environment, and exercise representative bounded
   benchmark tasks. Historical runtimes predate the fixes and cannot
   predict the corrected campaign's cost.
2. **Tuning.** Use two independent tuning seeds on each of `[[72,12,6]]`
   and `[[144,12,12]]`. Sweep plain and weighted ADMM rho/alpha settings
   under both greedy OSD and OSD-CS(60). Select and freeze the settings
   before confirmation; tuning samples are excluded from confirmation.
3. **Confirmation.** For each code at d = 6, run p = 0.005, 0.010,
   0.020 under both OSD choices. Each of the 12 cells has four
   independent 2500-shot shards (48 tasks in `default`; 40 shards / 480
   tasks for 100,000 shots per cell in `high_stat`), each comparing BP,
   plain ADMM, and weighted ADMM on the same samples.
4. **Held-out replication.** Repeat `[[72,12,6]]`, d = 6, p = 0.020 under
   both greedy OSD and OSD-CS(60) (`QUDEC_REPLICATE_MODES=greedy,lam60`)
   on independent 2500-shot seeds (`4..7` in `default`, `40..79` in
   `high_stat`). Report these predeclared replication cohorts separately
   from primary confirmation.
5. **Exploration.** Run both codes at d = 6 over p = 0.005, 0.010,
   0.020, 0.030 with 300 shots (1,000 in `high_stat`), including LDR.
   These results characterize the corrected settings and remain separate
   from confirmation. LDR uses fixed rho = 2, alpha = 1 and four outer
   iterations; it does not inherit a low rho selected for the plain
   convex relaxation.
6. **Code capacity.** Run `[[144,12,12]]` at p = 0.005, 0.010, 0.020,
   0.030, 0.050, 0.080, 0.100 with 500 shots (2,500 in `high_stat`),
   comparing BP, exact LP, and ADMM (plus the optional `ldpc`/`bposd`
   reference when `QUDEC_INCLUDE_REFERENCE=1`).
7. **Preliminary circuits.** By default, run `[[72,12,6]]` at three rounds,
   in both X and Z memory bases, at p = 0.001 and 0.002 with 300 shots
   (2,500 in `high_stat`). Standalone circuit tuning and paired JSONL
   analysis are also supported via `scripts/bench_circuit.py`.
8. **Final report.** Validate expected task coverage and provenance,
   then summarize the stages separately. Report counts, uncertainty,
   and paired decoder comparisons. A nonsignificant difference does
   not establish equivalence.

## Configuration and recovery

Set environment variables before submitting, or select a preset
(`QUDEC_PRESET=default|high_stat|followup`). The default
`QUDEC_CONFIRM_SEEDS=4` and `QUDEC_CONFIRM_SHOTS=2500` give 10,000 shots
per confirmation cell. `sbatch hpc/rerun_all.slurm high_stat` (or
`QUDEC_PRESET=high_stat`) configures 40 confirmation seeds and 40
replication seeds (100,000 shots per cell), 1,000 tuning shots, 2,500
capacity and circuit shots, `QUDEC_INCLUDE_REFERENCE=1`,
`QUDEC_MAX_PARALLEL=8`, and `QUDEC_TASK_TIME=24:00:00`.

`QUDEC_BATCH_SIZE=32` bounds decoder memory. `QUDEC_MAX_PARALLEL`
limits simultaneous benchmark workers. `QUDEC_TASK_TIME` sets the
benchmark worker time limit, and `QUDEC_PYTHON` overrides the Python
executable. Set `QUDEC_INCLUDE_CIRCUIT=0` to omit the circuit stage, or
`QUDEC_REPLICATE_MODES=lam60` to restrict replication to the legacy
campaign 21501046 task list.

Inspect the manifest plan without submitting:

    .venv/bin/python scripts/rerun_all.py plan
    .venv/bin/python scripts/rerun_all.py plan --preset high_stat

Resume a campaign from its original snapshot and settings:

    sbatch hpc/rerun_all.slurm resume /absolute/path/to/results/reruns/<initial_job_id>

Resume after the earlier jobs have stopped. Valid completed cells are
skipped and partial cells are replaced. Scheduling time and concurrency
may be overridden on retry; statistical inputs and decoder settings are
unchanged.

Optionally pull results to the local checkout with
`bash hpc/sync.sh pull`. Code updates can use Git independently of
result transfers. Keep completed campaign snapshots and raw artifacts
unchanged so their provenance remains verifiable.

## Findings and follow-up readiness

Weighted ADMM improves on BP in the `[[72,12,6]]`, p = 0.020, greedy-OSD
primary cell: 294 versus 339 failures in 10,000 paired shots. Its exact
paired p value is 0.001561 and its Holm-adjusted p value is 0.03746
across the 24 primary comparisons. No other primary comparison passes
that correction. Zero-failure cells provide upper bounds; a failure
count of zero in 10,000 shots has a 95% upper rate bound of 0.00036882.

The predeclared OSD-CS(60) held-out cohort in campaign 21501046 records
274 weighted-ADMM failures versus BP's 310, with paired p = 0.007434.
It remains separate from primary confirmation and does not replicate the
greedy-OSD cell. The four preliminary circuit comparisons favor BP.

The repository now includes:
- Complete, verified bibliographic citations and equation attributions in
  [docs/paper_draft.md](paper_draft.md).
- Default held-out replication for both `greedy` and `lam60` modes in
  `scripts/rerun_all.py`.
- Ready-to-submit `high_stat` (`100k`) and `followup` Slurm presets in
  `hpc/rerun_all.slurm` and `scripts/rerun_all.py`.
- Plain (`admm-u`), weighted (`admm-w`), and `admm-ldr` DEM circuit
  decoders, $(\rho, \alpha)$ tuning grids, and paired JSONL outcome
  output in `scripts/bench_circuit.py`.

## Retired files

The old text-only capacity sweeps (`bench_gross.py`, `bench_lp_gross.py`)
are replaced by `scripts/bench_capacity.py`. The unrelated root QPT
generator is removed. Their source and the withdrawn numerical tables
remain available in Git history. The optional reference wrapper remains
because the current capacity runner imports it.

Superseded standalone `results/tune_72.csv` and pre-correction logs were
removed from the local checkout after the corrected results were committed.
Campaign 21501046, its frozen source, all raw outputs, and its launcher
logs are retained. `hpc/rerun_all.slurm` remains the only Slurm entry point.
