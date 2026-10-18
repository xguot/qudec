# LP-family decoding under phenomenological noise for bivariate bicycle codes

## Abstract

We compare belief propagation with ordered statistics decoding (BP+OSD)
against plain and LLR-weighted parity-polytope ADMM under phenomenological
noise on [[72,12,6]] and [[144,12,12]] bivariate bicycle codes. Following
implementation repairs, independent tuning and confirmation use matched
channel priors, a perfect terminal syndrome, bounded decoding batches,
and paired outcomes. Twelve confirmation cells contain 10,000 shots each.
On the 72-qubit code at p = 0.020 with greedy OSD, weighted ADMM reduces
failures from 339 to 294, with exact paired p = 0.00156071 and
Holm-adjusted p = 0.0374571 across 24 primary tests. This is the only
primary comparison below 0.05 after adjustment. A separate CS(60) cohort
favors ADMM but does not independently replicate the greedy result.
Larger-code differences remain unresolved at these shot counts, and the
preliminary circuit sweep favors BP. The finding is limited to the tested
phenomenological setting. This remains a draft pending bibliography
verification and further replication.

## 1. Scope

The project asks whether optimization-based quantum LDPC decoding can
improve on BP+OSD once syndrome measurements are noisy. The implementation
contains exact LP decoding at code capacity, batched ADMM variants,
phenomenological memory experiments, and a preliminary Stim circuit.
Numerical claims here are limited to corrected campaign **21501046**,
completed on Rivanna on October 17, 2026.

The [report](results/21501046/report.json),
[manifest](results/21501046/manifest.json),
[frozen settings](results/21501046/selected.json), and
[audit record](results/21501046/audit.json) provide counts, marginal
intervals, paired tests, and provenance. Raw results and immutable source
remain under `results/reruns/21501046/` outside Git. Historical comparisons
were withdrawn after BP message, ADMM projection, measurement-prior, and
circuit-pipeline defects were found. They remain in Git history and are
excluded from the present analysis.

## 2. Methods

### 2.1 Codes, noise, and scoring

Both bivariate bicycle codes undergo six noisy rounds. Each round applies
depolarizing data noise at rate p (marginal X/Z error rate 2p/3) and
syndrome measurement flips at rate p. Detector syndromes obey
`delta_t = h e_t xor m_t xor m_(t-1)`. A perfect terminal syndrome adds
one detector block equal to the final measurement error (`closed-v1`).
Historical open-boundary experiments are a different protocol and are not
pooled with these data.

Failure includes a logical flip, residual data syndrome, or invalid
detector correction. All variants within a task decode identical samples.
Non-circuit outputs retain per-shot logical, invalid, residual, and failure
indicators, integer totals, sample fingerprints, parameters, environment,
and source hashes. Decoder batches contain at most 32 shots.

### 2.2 Decoders and selection

BP uses tie-safe min-sum messages, 30 iterations, scaling alpha = 0.9,
and matched priors 2p/3 for data and p for measurement columns. Plain ADMM
uses a uniform linear objective; weighted ADMM uses channel LLRs normalized
by their maximum absolute magnitude. Both use full parity-polytope
projection with forbidden-set cut search and a scalar root solve. OSD
post-processing is shared, with either the implemented greedy order-one
combination sweep or OSD-CS(60).

ADMM uses 150 iterations in the phenomenological benchmark. Two 300-shot
training seeds per code (100000 and 100001) at p = 0.020 sweep
rho in {0.5, 1, 2, 4, 8} and alpha in {1, 1.5, 1.8}.
Plain and weighted settings are chosen separately under each OSD mode
by minimum pooled training failures, breaking ties by lower rho then
alpha. They are frozen before confirmation and used at all three rates:

| Code | OSD | Plain rho / alpha | Weighted rho / alpha |
| --- | --- | --- | --- |
| 72 | greedy | 2 / 1.8 | 2 / 1.8 |
| 72 | CS(60) | 2 / 1.5 | 2 / 1.8 |
| 144 | greedy | 0.5 / 1 | 0.5 / 1 |
| 144 | CS(60) | 0.5 / 1 | 0.5 / 1 |

Exact LP with HiGHS is included only at code capacity. Exploratory LDR
uses rho = 2, alpha = 1, four outer iterations, and a diminishing
beta/sqrt(k+1) step with beta = 1; it does not inherit convex ADMM tuning.

### 2.3 Confirmation and inference

Each code, p in {0.005, 0.010, 0.020}, and OSD mode receives four
independent 2500-shot shards (seeds 0–3), for 10,000 shots per decoder.
Training samples are excluded. Two-sided exact McNemar tests compare each
ADMM family against paired BP outcomes. Holm adjustment covers all 24
primary comparisons at familywise threshold 0.05. The report retains raw
`mcnemar.p_value` and adjusted `holm_p_value` separately. Independent
two-proportion statistics are descriptive only.

Marginal 95% intervals are exact binomial intervals without simultaneous
coverage adjustment. The predeclared CS(60) replication uses seeds 4–7
on the 72-qubit code at p = 0.020, separately from primary data.
Replication and exploratory p-values are unadjusted.

### 2.4 Validation and execution

The cluster gate passed 116 tests, circuit smoke validation, and a
larger-code CUDA pilot. All 75 artifacts validated against their completion
markers, frozen settings, sample pairing, and checksums. Independent
reconstruction from raw outcomes reproduced aggregate counts and paired
tests. Non-circuit records had zero invalid or residual flags.

BP, LP, and OSD ran on CPU; ADMM used CUDA on A40/RTX A6000 allocations.
The environment included Python 3.10.21, PyTorch 2.5.1+cu121,
NumPy 2.2.6, and SciPy 1.15.3. Timings are descriptive and do not establish
a speedup against a matched-hardware baseline.

## 3. Corrected results

### 3.1 Primary phenomenological comparison

All entries are failures out of 10,000 shots. Marginal intervals and
both ADMM comparisons' exact tests are available in the report.

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

In the significant greedy 72-qubit cell, BP's rate is 0.0339
(95% interval [0.0304393, 0.0376355]) and weighted ADMM's is 0.0294
([0.0261768, 0.0329010]). The absolute reduction is 0.0045. There are
120 BP-only and 75 weighted-only failures, giving exact paired
p = 0.00156071 and Holm-adjusted p = 0.0374571. No other primary comparison
survives adjustment. Zero failures in 10,000 shots give a 95% upper bound
of 0.000368820, so larger-code low-rate cells remain limited by rare events.
Nonsignificance does not establish equivalence.

### 3.2 Separate CS(60) replication

On independent seeds 4–7, the 72-qubit p = 0.020 CS(60) cell has 310 BP,
270 plain-ADMM, and 274 weighted-ADMM failures per 10,000 shots. Exact
paired p-values are 0.00411824 and 0.00743387, respectively, without
multiplicity adjustment. These data favor ADMM in that cohort, while the
corresponding primary weighted comparison has Holm p = 0.975753.
This cohort does not independently replicate the significant greedy
comparison, and is not pooled with primary data.

### 3.3 Code capacity and exploration

At code capacity on [[144,12,12]], failures per 500 shots are:

| p | BP | Exact LP | ADMM |
| --- | ---: | ---: | ---: |
| 0.005 | 0 | 0 | 0 |
| 0.010 | 0 | 0 | 0 |
| 0.020 | 0 | 0 | 0 |
| 0.030 | 2 | 0 | 0 |
| 0.050 | 7 | 2 | 2 |
| 0.080 | 57 | 48 | 41 |
| 0.100 | 159 | 147 | 147 |

These are small baseline checks, separate from the primary family. The
optional external-reference stack was not enabled. Eight 300-shot
phenomenological exploration cells, including LDR, also completed;
full results are in the report and provide no additional confirmatory claim.

### 3.4 Preliminary circuit sweep

The 72-qubit three-round memory experiment uses DEM mechanism priors,
300 shots per cell, and unweighted ADMM at rho = 2, alpha = 1,
500 iterations. Counts are:

| Memory basis | p | BP failures | ADMM failures |
| --- | --- | ---: | ---: |
| Z | 0.001 | 1 | 7 |
| Z | 0.002 | 10 | 50 |
| X | 0.001 | 1 | 8 |
| X | 0.002 | 11 | 41 |

BP has fewer failures in every tested cell. Circuit CSVs retain marginal
counts; the report adds exact binomial intervals. Per-shot paired outcomes
were not saved. These preliminary
settings were not tuned independently for the circuit and provide no
evidence of a circuit-level ADMM advantage.

## 4. Discussion and remaining work

The corrected campaign supports one modest weighted-ADMM improvement under
the specified phenomenological protocol. It does not establish a general
advantage across code sizes, OSD choices, or circuit noise. The priority
is independent replication of the fixed greedy 72-qubit p = 0.020
comparison. Higher larger-code statistics, separate circuit tuning, and
protocol-sensitivity experiments are further research, not missing tasks
in this completed campaign.

Finite iteration limits, small training samples, exploratory LDR choices,
GPU variation, and unequal CPU/GPU execution remain limitations.
A 100,000-shot campaign and fresh external-reference comparison were not
performed. Completed tasks need no rerun for the report-label correction;
raw samples, settings, and original source remain unchanged. Literature
coverage and the bibliography must be verified before submission.

## References to verify before submission

This draft retains the previous reference placeholders; publication details,
equation attribution, and current literature coverage are not yet checked:
Bravyi et al. (bivariate bicycle codes); Roffe et al. (BP+OSD);
Gu and Soleimanifar (LP/ADMM decoding); Barman et al. (ADMM LP decoding);
Li and Vontobel (quantum LP decoding); SymBreak; Gensheimer et al.
(parity-polytope projection, arXiv:1901.03240).
