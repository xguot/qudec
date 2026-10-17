# Corrected rerun campaign

The correctness audit found tied-minimum BP messages, truncated ADMM
projections, unmatched BP measurement priors, and circuit-pipeline bugs.
These are repaired, with regression tests. Historical decoder comparisons
remain available for provenance; their advantage and deficit conclusions
are withdrawn pending corrected runs.

All new phenomenological runs use full parity projection, matched
per-column BP priors, bounded decoder batches, and a perfect terminal
syndrome. Per-shot failures support paired inference. The historical
open-boundary metric is a separate protocol and is excluded from this
campaign.

## One submission

From the local repository, run `bash hpc/sync.sh push`. On Rivanna, enter
the synced repository and run `bash hpc/setup.sh` once. Submit:

    sbatch hpc/rerun_all.slurm

This is the only Slurm entry point. It creates an immutable source
snapshot and a campaign directory under
`results/reruns/<initial_job_id>/`, with the manifest, frozen settings,
outputs, provenance, and worker logs. The initial GPU job runs validation
and a pilot before submitting dependent stages. A failed gate prevents
the subsequent research stage from running. The workflow has been
prepared locally; a corrected cluster campaign has not been submitted
as part of this work.

## Stages and default scope

1. **Validation and pilot.** Run the unit suite and circuit smoke checks,
   verify the GPU environment, and exercise representative bounded
   benchmark tasks. Historical runtimes predate the fixes and cannot
   predict the corrected campaign's cost.
2. **Tuning.** Use two independent tuning seeds on each of [[72,12,6]]
   and [[144,12,12]]. Sweep plain and weighted ADMM rho/alpha settings
   under both greedy OSD and OSD-CS(60). Select and freeze the settings
   before confirmation; tuning samples are excluded from confirmation.
3. **Confirmation.** For each code at d = 6, run p = 0.005, 0.010,
   0.020 under both OSD choices. Each of the 12 cells has four
   independent 2500-shot shards (48 tasks), each comparing BP, plain
   ADMM, and weighted ADMM on the same samples. This gives 10,000 shots
   per decoder per cell.
4. **Held-out replication.** Repeat [[72,12,6]], d = 6, p = 0.020,
   OSD-CS(60), on four further independent 2500-shot seeds. Report this
   predeclared replication separately from the primary confirmation.
5. **Exploration.** Run both codes at d = 6 over p = 0.005, 0.010,
   0.020, 0.030 with 300 shots, including LDR. These results characterize
   the corrected settings and remain separate from confirmation.
   LDR uses fixed rho = 2, alpha = 1 and four outer iterations; it does
   not inherit a low rho selected for the plain convex relaxation.
6. **Code capacity.** Run [[144,12,12]] at p = 0.005, 0.010, 0.020,
   0.030, 0.050, 0.080, 0.100 with 500 shots, comparing BP, exact LP,
   and ADMM. This is a modest baseline check; low-rate zero-failure
   cells provide upper bounds rather than evidence of zero error.
7. **Preliminary circuits.** By default, run [[72,12,6]] at three rounds,
   in both X and Z memory bases, at p = 0.001 and 0.002 with 300 shots.
   These exploratory Stim results do not establish high-statistics
   circuit-level performance. A native Sinter adapter is not provided.
8. **Final report.** Validate expected task coverage and provenance,
   then summarize the stages separately. Report counts, uncertainty,
   and paired decoder comparisons. A nonsignificant difference does
   not establish equivalence.

The external `ldpc`/`bposd` code-capacity comparison is optional and off
by default. Enable it when a fresh external-reference column is needed;
historical reference numbers must remain clearly labeled.

## Configuration and recovery

Set environment variables before submitting. The default
`QUDEC_CONFIRM_SEEDS=4` and `QUDEC_CONFIRM_SHOTS=2500` give 10,000 shots
per confirmation cell. Set `QUDEC_CONFIRM_SEEDS=40` for 100,000 shots
per cell. This larger campaign is an explicit escalation in cost.

`QUDEC_BATCH_SIZE=32` bounds decoder memory. `QUDEC_MAX_PARALLEL=4`
limits simultaneous benchmark workers. `QUDEC_TASK_TIME=12:00:00`
sets the benchmark worker time limit, and `QUDEC_PYTHON` overrides the
Python executable. Set `QUDEC_INCLUDE_CIRCUIT=0` to omit the preliminary
circuit stage, or `QUDEC_INCLUDE_REFERENCE=1` to include the external
reference stack installed by setup.

Inspect the manifest plan without submitting:

    .venv/bin/python scripts/rerun_all.py plan

Resume a campaign from its original snapshot and settings:

    sbatch hpc/rerun_all.slurm resume /absolute/path/to/results/reruns/<initial_job_id>

Resume after the earlier jobs have stopped. Valid completed cells are
skipped and partial cells are replaced. Scheduling time and concurrency
may be overridden on retry; statistical inputs and decoder settings are
unchanged.

Pull the results with `bash hpc/sync.sh pull`. Sync push retires the
eight replaced Slurm files by exact filename on the remote checkout and
preserves historical logs and campaign results. Its `--dry-run` option
does not create or remove remote files.

## After the campaign

Check the final report and any failed or missing tasks before updating
research claims. Integrate corrected tables and their uncertainty into
the manuscript, retain the replication distinction, and finish the
bibliography. Historical tables are not corrected evidence merely
because the implementation now passes its tests.
