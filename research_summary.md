# Research summary: corrected decoder campaign

Campaign **21501046** completed on Rivanna on October 17, 2026. All
75 expected artifacts validate against the frozen source, settings,
sample fingerprints, and output checksums. The cluster preflight passed
116 tests and the circuit smoke checks. Independent reconstruction from
saved outcomes reproduced every aggregate count and paired test.
Non-circuit benchmark records have zero invalid or residual-syndrome flags.

The tracked [report](docs/results/21501046/report.json),
[manifest](docs/results/21501046/manifest.json),
[selected settings](docs/results/21501046/selected.json), and
[audit record](docs/results/21501046/audit.json) retain the results and
provenance. Raw outcomes, immutable source, and worker logs remain in
`results/reruns/21501046/`. Launcher validation logs remain in
`hpc/logs/qudec_rerun_21501046.{out,err}`. These raw directories are
ignored by Git and must be retained separately.

## Interpretation

One primary phenomenological comparison supports a modest weighted-ADMM
improvement: [[72,12,6]], six noisy rounds, p = 0.020, greedy OSD.
BP failed on 339/10,000 samples and weighted ADMM on 294/10,000
(3.39% versus 2.94%, a 0.45 percentage-point reduction). There were
120 BP-only and 75 weighted-only failures. The two-sided exact McNemar
p-value is 0.00156071; Holm adjustment over all 24 primary comparisons
gives **0.0374571**. This is the only primary comparison below the
0.05 familywise threshold.

The separate CS(60) cohort favors ADMM but uses a different OSD setting;
it does **not** independently replicate the significant greedy comparison.
No [[144,12,12]] primary comparison survives Holm correction. The
preliminary circuit sweep favors BP in every cell. The evidence supports
a limited phenomenological finding, not a general decoder ranking or a
demonstrated circuit-level advantage.

## Protocol and frozen settings

Both codes use six depolarizing data-noise rounds, measurement flips at p,
and a perfect terminal syndrome (`closed-v1`). BP uses matched per-column
priors: 2p/3 for data and p for measurement columns. Variants decode the
same samples in batches of at most 32 with shared OSD post-processing.
Failure includes a logical flip, residual data syndrome, or invalid
detector correction.

Tuning used seeds 100000 and 100001, 300 shots each, at p = 0.020 for
each code. Plain and normalized-LLR-weighted ADMM were selected separately
for each OSD mode from rho = {0.5, 1, 2, 4, 8} and
alpha = {1, 1.5, 1.8}, minimizing pooled training failures, with ties
broken by lower rho then alpha. Settings were frozen before confirmation:

| Code | OSD | Plain rho / alpha | Weighted rho / alpha |
| --- | --- | --- | --- |
| 72 | greedy | 2 / 1.8 | 2 / 1.8 |
| 72 | CS(60) | 2 / 1.5 | 2 / 1.8 |
| 144 | greedy | 0.5 / 1 | 0.5 / 1 |
| 144 | CS(60) | 0.5 / 1 | 0.5 / 1 |

Primary cells use seeds 0–3, four 2500-shot shards each; training samples
are excluded. The predeclared CS(60) replication uses seeds 4–7 and is
reported separately. Exact binomial 95% intervals are marginal, without
simultaneous coverage adjustment. Primary inference uses paired exact
McNemar tests and Holm correction across 2 codes × 3 rates × 2 OSD
settings × 2 ADMM comparisons. Other phases' p-values are unadjusted.
Independent two-proportion statistics in the report are descriptive only.

## Primary confirmation

Entries are **failure counts out of 10,000**, paired within each row.
Full marginal intervals and both ADMM families' raw and adjusted tests
are in the tracked report.

| Code | p | OSD | BP | Plain ADMM | Weighted ADMM | Weighted Holm p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 72 | 0.005 | greedy | 4 | 3 | 4 | 1 |
| 72 | 0.005 | CS(60) | 3 | 3 | 3 | 1 |
| 72 | 0.010 | greedy | 35 | 26 | 26 | 1 |
| 72 | 0.010 | CS(60) | 31 | 29 | 30 | 1 |
| 72 | 0.020 | greedy | 339 | 323 | 294 | **0.037457** |
| 72 | 0.020 | CS(60) | 321 | 310 | 294 | 0.975753 |
| 144 | 0.005 | greedy | 0 | 0 | 0 | 1 |
| 144 | 0.005 | CS(60) | 0 | 0 | 0 | 1 |
| 144 | 0.010 | greedy | 1 | 0 | 0 | 1 |
| 144 | 0.010 | CS(60) | 0 | 0 | 0 | 1 |
| 144 | 0.020 | greedy | 19 | 7 | 6 | 0.165540 |
| 144 | 0.020 | CS(60) | 6 | 3 | 3 | 1 |

For the significant 72-qubit cell, BP's marginal 95% interval is
[0.0304393, 0.0376355], and weighted ADMM's is [0.0261768, 0.0329010].
The paired test accounts for shared samples. Zero failures in 10,000 shots
give a two-sided exact 95% upper bound of 0.000368820 rather than evidence
of a zero underlying failure rate. Nonsignificance does not establish
equivalence.

## Separate replication and exploratory phases

The held-out [[72,12,6]], p = 0.020, CS(60) cohort has BP 310,
plain ADMM 270, and weighted ADMM 274 failures per 10,000 shots.
Paired exact p-values against BP are 0.00411824 and 0.00743387,
respectively. These nominal p-values are separate from the 24-test
primary family; cohorts are not pooled.

At code capacity, [[144,12,12]] has these failures per 500 shots:

| p | BP | Exact LP | ADMM |
| --- | ---: | ---: | ---: |
| 0.005 | 0 | 0 | 0 |
| 0.010 | 0 | 0 | 0 |
| 0.020 | 0 | 0 | 0 |
| 0.030 | 2 | 0 | 0 |
| 0.050 | 7 | 2 | 2 |
| 0.080 | 57 | 48 | 41 |
| 0.100 | 159 | 147 | 147 |

This is a modest baseline check. The optional external `ldpc`/`bposd`
reference was not enabled. All eight 300-shot phenomenological exploration
cells, including LDR, also completed and remain in the report. They do
not add confirmatory evidence. LDR used fixed rho = 2, alpha = 1 and four
outer iterations rather than inheriting convex ADMM tuning.

The preliminary [[72,12,6]] circuit sweep uses three rounds, 300 shots
per cell, unweighted ADMM at rho = 2 / alpha = 1, and DEM mechanism priors:

| Memory basis | p | BP failures | ADMM failures |
| --- | --- | ---: | ---: |
| Z | 0.001 | 1 | 7 |
| Z | 0.002 | 10 | 50 |
| X | 0.001 | 1 | 8 |
| X | 0.002 | 11 | 41 |

Circuit CSVs retain marginal counts; the report adds exact binomial
intervals. Per-shot paired outcomes were not saved for a paired significance
test. These results provide no evidence of a
circuit-level ADMM advantage at the tested settings.

## Limitations and next work

The completed default campaign needs no resubmission for these reporting
edits. Further research should independently replicate the greedy
72-qubit p = 0.020 setting with frozen parameters, increase larger-code
statistics where rare failures limit inference, and tune circuit settings
on separate training samples before another circuit comparison.
A 100,000-shot confirmation and external reference are optional follow-ups,
not completed stages.

BP/LP and OSD ran on CPU; ADMM used CUDA on A40/RTX A6000 allocations
(Python 3.10.21, PyTorch 2.5.1+cu121, NumPy 2.2.6, SciPy 1.15.3).
These runs do not establish a speedup against matched hardware. Finite
iteration budgets, GPU variation, modest training samples, and exploratory
LDR choices remain limitations. The manuscript bibliography and current
literature review still need verification before submission.

Historical comparisons were withdrawn because of BP tie handling, ADMM
projection, measurement-prior, and circuit-pipeline defects. Their open
terminal boundary also differs from `closed-v1`. Superseded tables remain
in Git history and must not be pooled with this campaign. See the
[campaign plan](docs/plan.md) and [manuscript draft](docs/paper_draft.md).
