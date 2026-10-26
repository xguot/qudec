# LP-family decoding under phenomenological noise for bivariate bicycle codes

## Abstract

We compare belief propagation with ordered statistics decoding (BP+OSD) [2, 3]
against plain and log-likelihood-ratio (LLR) weighted parity-polytope
alternating direction method of multipliers (ADMM) decoding [4, 5] under
phenomenological noise on `[[72,12,6]]` and `[[144,12,12]]` bivariate bicycle
(BB) codes [1]. Following implementation repairs, independent tuning and
confirmation use matched channel priors, a perfect terminal syndrome, bounded
decoding batches, and paired per-shot outcomes. Twelve primary confirmation
cells contain 10,000 shots per decoder each. On the 72-qubit code at
$p = 0.020$ with greedy OSD, weighted ADMM reduces failures from 339 to 294,
with two-sided exact paired McNemar $p = 0.00156071$ and Holm-adjusted
$p = 0.0374571$ across all 24 primary tests. This is the only primary
comparison below 0.05 after familywise adjustment. A separate `OSD-CS(60)`
held-out cohort favors ADMM ($p = 0.004118$ plain, $p = 0.007434$ weighted
unadjusted) but does not independently replicate the greedy setting.
Larger-code differences remain unresolved at 10,000 shots, and a preliminary
three-round Stim circuit sweep favors BP. The evidence supports a modest
phenomenological improvement in a single high-noise 72-qubit regime rather
than a general decoder ranking or circuit-level advantage.

## 1. Scope and background

Low-overhead quantum low-density parity-check (QLDPC) architectures rely on
sparse codes with nonlocal connectivity, notably bivariate bicycle (BB) codes
such as `[[72,12,6]]` and `[[144,12,12]]` [1]. Standard message-passing belief
propagation (BP) suffers from short cycles and quantum degeneracy on CSS
Tanner graphs, motivating ordered statistics decoding (BP+OSD) post-processing
[2, 3] as well as recent graph-modification, symmetry-breaking, ensemble, and
neural approaches [11–14].

An alternative optimization framework relaxes maximum-likelihood syndrome
decoding to a linear or quadratic program over parity polytopes [4–10].
Gu and Soleimanifar [4] analyzed the subset-based linear programming (LP)
relaxation (their Eq. (2)) for CSS codes, proving that constant-weight error
configurations supported on Tanner-graph cycles produce fractional LP optima
that fail under independent rounding, while combining LP reliabilities with
OSD post-processing and Tanner-graph syndrome-distance tie breaking (LP+OSD,
their Sec. IV, Algorithm 1) outperforms BP+OSD on several mid-sized codes at
code capacity. Independently, Lyu and He [5] formulated quantum LDPC decoding
as a continuous quadratic program over check parity polytopes (their Eq. (8))
solved via over-relaxed ADMM updates (their Eqs. (6a)–(6d) and (9)) and an
adaptive Lagrangian dual relaxation penalty (LDR-ADMM, their Sec. IV,
Eqs. (11)–(12)), building on classical ADMM LP decoding [6, 7] and fast
cut-search parity-polytope projection [8, 9].

This study evaluates whether parity-polytope ADMM+OSD improves on tie-safe
min-sum BP+OSD once syndrome measurements are noisy. The repository implements
exact HiGHS LP+OSD [4] at code capacity, batched GPU-accelerated ADMM+OSD
variants [4, 5, 9], closed-boundary phenomenological memory benchmarks, and
Stim detector-error-model (DEM) circuit benchmarks [15]. Numerical claims in
this manuscript are restricted to audited campaign **21501046**, completed on
Rivanna on October 17, 2026.

The tracked [report](results/21501046/report.json),
[manifest](results/21501046/manifest.json),
[frozen settings](results/21501046/selected.json), and
[audit record](results/21501046/audit.json) provide exact failure counts,
marginal binomial intervals, paired tests, and SHA-256 provenance. Raw
per-shot records and the immutable source snapshot remain under
`results/reruns/21501046/` outside Git. Historical comparisons were withdrawn
after auditing identified tied-minimum BP message defects, truncated ADMM
parity-slice projections, unmatched BP measurement priors, and circuit-pipeline
bugs.

## 2. Methods

### 2.1 Codes, noise, and scoring

Both bivariate bicycle codes (`[[72,12,6]]` and `[[144,12,12]]` [1]) undergo
$d = 6$ noisy rounds. Each round applies depolarizing data noise at rate $p$
(marginal $X$/$Z$ flip probability $2p/3$) and independent syndrome measurement
flips at rate $p_{\text{meas}} = p$. Difference detectors obey
$\delta_t = H e_t \oplus m_t \oplus m_{t-1}$ for $t \in \{1, \dots, d\}$,
followed by a noiseless terminal syndrome readout that appends the final
measurement-error block $\delta_{d+1} = m_d$ (`closed-v1`). Historical
open-boundary experiments omit this terminal closure and are not pooled with
`closed-v1` data.

A block failure occurs if either CSS sector exhibits a nontrivial logical
operator flip, a nonzero residual data syndrome, or an invalid detector
correction ($\hat{H} \hat{c} \neq \delta$). All decoder variants within a task
decode identical sampled error realizations in batches of at most 32 shots.
Non-circuit JSONL records store per-shot logical, invalid, residual, and
combined failure bitstrings, integer totals, SHA-256 sample fingerprints,
solver parameters, runtime environment metadata, and source hashes.

### 2.2 Decoders and hyperparameter selection

Min-sum BP uses tie-safe message exclusion, 30 flooding iterations, message
scaling $\alpha = 0.9$, and matched per-column priors ($2p/3$ for data columns
and $p$ for measurement columns) [3]. Plain ADMM minimizes the uniform linear
objective $c = \mathbf{1}$ over the parity-polytope relaxation (Lyu and He [5],
Eq. (8) with $g = 0$). Weighted ADMM (`admm-w`) uses per-column channel log-likelihood
ratios $c_i = \log((1 - p_i)/p_i)$ as specified for heterogeneous error
mechanisms in Gu and Soleimanifar [4, Appendix C], normalized by
$\max_i |c_i|$ so the linear objective scale remains invariant across error
rates. Both ADMM variants project check neighborhoods onto even/odd parity
polytopes via the Zhang–Siegel / Gensheimer et al. cut-search algorithm [8, 9]
with 32-step scalar bisection. OSD post-processing is shared across all
decoders, using either the greedy order-1 combination sweep (`greedy`,
$\lambda = \text{None}$) or `OSD-CS(60)` ($\lambda = 60$) [3, 4], with ADMM
and LP column reliabilities ordered by descending indicator $x_i$ and ties
broken by ascending Tanner-graph distance to a violated check [4, Sec. IV,
Algorithm 1].

Phenomenological ADMM runs use a maximum of 150 iterations with primal/dual
stopping tolerances $\epsilon^{\text{pri}} = \epsilon^{\text{dual}} = 10^{-5}$
[5, Eq. (10)]. Two independent 300-shot training seeds per code (100000 and
100001) at $p = 0.020$ sweep $\rho \in \{0.5, 1, 2, 4, 8\}$ and over-relaxation
$\alpha \in \{1.0, 1.5, 1.8\}$. Plain and weighted ADMM settings are selected
separately under each OSD mode by minimizing pooled training failures, breaking
ties by smaller $\rho$ then smaller $\alpha$. Selected parameters were frozen
prior to confirmation and applied across all three physical error rates:

| Code | OSD | Plain $\rho / \alpha$ | Weighted $\rho / \alpha$ |
| --- | --- | --- | --- |
| 72 | greedy | 2.0 / 1.8 | 2.0 / 1.8 |
| 72 | CS(60) | 2.0 / 1.5 | 2.0 / 1.8 |
| 144 | greedy | 0.5 / 1.0 | 0.5 / 1.0 |
| 144 | CS(60) | 0.5 / 1.0 | 0.5 / 1.0 |

Exact subset LP decoding with HiGHS [4, Eq. (2)] is evaluated at code capacity
($d = 0$, 500 shots per rate). Exploratory LDR-ADMM [5, Sec. IV-B] uses fixed
$\rho = 2.0$, $\alpha = 1.0$, four outer subgradient iterations, and diminishing
step size $\beta / \sqrt{k + 1}$ with $\beta = 1.0$ so the concave penalty
$-2\lambda_i$ remains coercive ($g_i + \rho d_v(i) > 0$).

### 2.3 Confirmation design and statistical inference

Each primary cell — 2 codes $\times$ 3 error rates ($p \in \{0.005, 0.010, 0.020\}$)
$\times$ 2 OSD modes — comprises four independent 2,500-shot shards (seeds 0–3),
yielding 10,000 paired shots per decoder per cell. Training seeds are strictly
disjoint from confirmation and replication seeds. Two-sided exact McNemar tests
compare plain ADMM and weighted ADMM against paired BP outcomes on the same
10,000 samples. Holm step-down adjustment controls the familywise error rate
at $\alpha_{\text{FWER}} = 0.05$ across all 24 predeclared primary comparisons.
The report records raw `mcnemar.p_value` and adjusted `holm_p_value` fields
explicitly; unpaired two-proportion $z$-statistics are retained as descriptive
summaries only.

Marginal 95% confidence intervals for logical error rates are exact
Clopper–Pearson binomial intervals without simultaneous coverage adjustment.
A predeclared held-out replication cohort uses seeds 4–7 (10,000 shots) on
`[[72,12,6]]` at $p = 0.020$ and is analyzed separately from the primary family.

### 2.4 Cluster validation and execution environment

The Rivanna preflight gate passed all 116 unit tests, deterministic Steane
circuit smoke checks, and a `[[144,12,12]]` CUDA pilot before dispatching worker
arrays. All 75 campaign artifacts passed completion-marker, frozen-setting,
sample-fingerprint, and SHA-256 checksum audits. Independent reconstruction
from raw bitstrings reproduced every aggregate count and paired McNemar test
statistic. Non-circuit records exhibited zero invalid detector corrections and
zero residual data syndromes across all decoded shots.

BP, exact LP, and OSD post-processing executed on CPU, while batched ADMM ran
on CUDA (`NVIDIA A40` and `NVIDIA RTX A6000` nodes; Python 3.10.21,
PyTorch 2.5.1+cu121, NumPy 2.2.6, SciPy 1.15.3). Wall-clock times are therefore
descriptive and do not represent a matched-hardware latency comparison.

## 3. Corrected results

### 3.1 Primary phenomenological confirmation

Table entries report **total block failures out of 10,000 paired shots** per
cell. Full marginal 95% intervals and exact paired statistics for both plain and
weighted ADMM are archived in [results/21501046/report.json](results/21501046/report.json).

| Code | $p$ | OSD | BP | Plain ADMM | Weighted ADMM | Plain Holm $p$ | Weighted Holm $p$ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 72 | 0.005 | greedy | 4 | 3 | 4 | 1.000000 | 1.000000 |
| 72 | 0.005 | CS(60) | 3 | 3 | 3 | 1.000000 | 1.000000 |
| 72 | 0.010 | greedy | 35 | 26 | 26 | 1.000000 | 1.000000 |
| 72 | 0.010 | CS(60) | 31 | 29 | 30 | 1.000000 | 1.000000 |
| 72 | 0.020 | greedy | 339 | 323 | 294 | 1.000000 | **0.037457** |
| 72 | 0.020 | CS(60) | 321 | 310 | 294 | 1.000000 | 0.975753 |
| 144 | 0.005 | greedy | 0 | 0 | 0 | 1.000000 | 1.000000 |
| 144 | 0.005 | CS(60) | 0 | 0 | 0 | 1.000000 | 1.000000 |
| 144 | 0.010 | greedy | 1 | 0 | 0 | 1.000000 | 1.000000 |
| 144 | 0.010 | CS(60) | 0 | 0 | 0 | 1.000000 | 1.000000 |
| 144 | 0.020 | greedy | 19 | 7 | 6 | 0.371812 | 0.165540 |
| 144 | 0.020 | CS(60) | 6 | 3 | 3 | 1.000000 | 1.000000 |

In the significant `[[72,12,6]]`, $p = 0.020$, greedy-OSD cell, BP fails on
339/10,000 shots ($\text{LER} = 0.0339$, 95% CI $[0.0304393, 0.0376355]$) and
weighted ADMM fails on 294/10,000 shots ($\text{LER} = 0.0294$, 95% CI
$[0.0261768, 0.0329010]$), an absolute reduction of $0.0045$ (13.3% relative
reduction). Discordant pairs consist of 120 BP-only failures and 75
weighted-ADMM-only failures (195 discordant shots), yielding two-sided exact
McNemar $p = 0.00156071$ and Holm-adjusted $p = 0.0374571$. On `[[144,12,12]]`
at $p = 0.020$ with greedy OSD, weighted ADMM records 6 failures versus BP's 19
(17 BP-only vs. 4 weighted-ADMM-only, raw exact McNemar $p = 0.00719738$),
which gives Holm-adjusted $p = 0.165540$ across the 24-test family. Zero
failures in 10,000 shots correspond to a two-sided exact 95% upper bound of
$3.6882 \times 10^{-4}$, so low-rate `[[144,12,12]]` comparisons are bounded by
rare-event statistics at 10,000 shots.

### 3.2 Separate held-out replication cohort

On independent seeds 4–7 (10,000 shots) for `[[72,12,6]]` at $p = 0.020$ with
`OSD-CS(60)`, BP records 310 failures, plain ADMM records 270 failures
(104 BP-only vs. 64 plain-only, unadjusted exact McNemar $p = 0.00411824$), and
weighted ADMM records 274 failures (99 BP-only vs. 63 weighted-only, unadjusted
exact McNemar $p = 0.00743387$). Because campaign 21501046 predeclared only the
`OSD-CS(60)` setting for seeds 4–7 while the primary comparison that passed
Holm correction used greedy OSD ($\lambda = \text{None}$), this cohort provides
supportive evidence for ADMM at $p = 0.020$ under `OSD-CS(60)` but does not
constitute an independent replication of the greedy-OSD primary finding. The
updated campaign runner (`scripts/rerun_all.py`) now schedules held-out
replication shards for both `greedy` and `lam60` modes by default.

### 3.3 Code capacity and phenomenological exploration

At code capacity ($d = 0$, 500 shots per rate) on `[[144,12,12]]`, failures are:

| $p$ | BP+OSD | Exact LP+OSD | ADMM+OSD |
| --- | ---: | ---: | ---: |
| 0.005 | 0 | 0 | 0 |
| 0.010 | 0 | 0 | 0 |
| 0.020 | 0 | 0 | 0 |
| 0.030 | 2 | 0 | 0 |
| 0.050 | 7 | 2 | 2 |
| 0.080 | 57 | 48 | 41 |
| 0.100 | 159 | 147 | 147 |

Consistent with Gu and Soleimanifar [4], LP+OSD and ADMM+OSD track closely and
show modest reductions relative to BP+OSD at intermediate physical error rates
($p = 0.050$–$0.080$, unadjusted McNemar $p = 0.00904$ for ADMM vs. BP at
$p = 0.080$). Eight 300-shot phenomenological exploration cells (including
4-outer-iteration LDR-ADMM [5]) also completed and are archived in the campaign
report as non-confirmatory characterization data.

### 3.4 Preliminary Stim circuit sweep

The preliminary `[[72,12,6]]` three-round Stim memory experiment [1, 15] used
undecomposed detector-error-model (DEM) mechanism priors, 300 shots per cell,
and untuned ADMM ($\rho = 2.0, \alpha = 1.0$, 500 iterations, greedy OSD):

| Memory basis | $p$ | BP+OSD failures | ADMM+OSD failures |
| --- | --- | ---: | ---: |
| Z | 0.001 | 1 | 7 |
| Z | 0.002 | 10 | 50 |
| X | 0.001 | 1 | 8 |
| X | 0.002 | 11 | 41 |

BP+OSD achieves fewer failures in all four preliminary cells. Circuit-level DEM
matrices for BB codes have much larger check/mechanism dimensions and extensive
short-cycle hyperedge overlap from correlated two-qubit gate faults [13, 14].
Without circuit-specific $(\rho, \alpha)$ tuning or graph-level correlation
rewiring [13], fixed-budget ADMM underperforms BP+OSD on these circuit
instances.

## 4. Discussion, limitations, and follow-up capability

Campaign 21501046 demonstrates a statistically significant weighted-ADMM
improvement in one primary phenomenological cell (`[[72,12,6]]`, $d = 6$,
$p = 0.020$, greedy OSD) after 24-test Holm correction, alongside nominal
improvements in the held-out `OSD-CS(60)` cohort and in the `[[144,12,12]]`,
$p = 0.020$ greedy cell (raw $p = 0.0072$, Holm $p = 0.1655$). However, it
does not establish a universal advantage across code sizes, OSD settings, or
circuit-level noise. Key limitations include:

1. **Held-out greedy replication**: Campaign 21501046 replicated `OSD-CS(60)`
   rather than `greedy` on seeds 4–7. The campaign planner in
   `scripts/rerun_all.py` now includes both `greedy` and `lam60` held-out
   replication cohorts (`QUDEC_REPLICATE_MODES=greedy,lam60`).
2. **Rare-event statistics on `[[144,12,12]]`**: Ten thousand shots per cell are
   insufficient to resolve sub-$10^{-3}$ failure rates at $p \le 0.010$ or to
   confirm the 19-vs.-6 failure gap at $p = 0.020$ after 24-test multiplicity
   correction. A ready-to-submit 100,000-shot preset (`sbatch hpc/rerun_all.slurm high_stat`,
   setting `QUDEC_CONFIRM_SEEDS=40` and `QUDEC_REPLICATE_SEEDS=40`) is provided
   for higher-statistics cluster execution.
3. **Circuit-level tuning and paired inference**: `scripts/bench_circuit.py` and
   `src/qudec/circuit_decoders.py` now support explicit plain (`admm-u`),
   LLR-weighted (`admm` / `admm-w`), and `admm-ldr` DEM decoding, $(\rho, \alpha)$
   tuning grids (`--rhos`, `--alphas`), and per-shot paired JSONL outcome
   records (`--output <path>.jsonl`) compatible with `scripts/analyze_phenom.py`.
4. **Hardware parity**: Because BP and OSD ran on CPU while ADMM ran on CUDA
   GPUs, wall-clock measurements reflect implementation placement rather than
   intrinsic algorithmic latency.

## References

1. S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall, and T. J. Yoder,
   "High-threshold and low-overhead fault-tolerant quantum memory," *Nature*,
   vol. 627, no. 8005, pp. 778–782, Mar. 2024,
   doi: [10.1038/s41586-024-07107-7](https://doi.org/10.1038/s41586-024-07107-7).
2. P. Panteleev and G. Kalachev, "Degenerate quantum LDPC codes with good
   finite length performance," *Quantum*, vol. 5, p. 585, Nov. 2021,
   doi: [10.22331/q-2021-11-22-585](https://doi.org/10.22331/q-2021-11-22-585).
3. J. Roffe, D. R. White, S. Burton, and E. Campbell, "Decoding across the
   quantum low-density parity-check code landscape," *Physical Review Research*,
   vol. 2, no. 4, Art. no. 043423, Dec. 2020,
   doi: [10.1103/PhysRevResearch.2.043423](https://doi.org/10.1103/PhysRevResearch.2.043423).
4. S. Gu and M. Soleimanifar, "Power and Limitations of Linear Programming
   Decoder for Quantum LDPC Codes," *IEEE Transactions on Information Theory*,
   vol. 72, no. 10, pp. 7604–7618, Oct. 2026,
   doi: [10.1109/TIT.2026.3715538](https://doi.org/10.1109/TIT.2026.3715538)
   (arXiv:2508.04769).
5. J. Lyu and G. He, "ADMM Decoding of Quantum LDPC Codes," in *Proc. 2026 IEEE
   International Symposium on Information Theory (ISIT)*, 2026, pp. 1–6,
   doi: [10.1109/ISIT62367.2026.11653953](https://doi.org/10.1109/ISIT62367.2026.11653953).
6. S. Barman, X. Liu, S. C. Draper, and B. Recht, "Decomposition Methods for
   Large Scale LP Decoding," *IEEE Transactions on Information Theory*, vol. 59,
   no. 12, pp. 7870–7886, Dec. 2013,
   doi: [10.1109/TIT.2013.2281372](https://doi.org/10.1109/TIT.2013.2281372).
7. X. Liu and S. C. Draper, "The ADMM Penalized Decoder for LDPC Codes," *IEEE
   Transactions on Information Theory*, vol. 62, no. 6, pp. 2966–2984, Jun. 2016,
   doi: [10.1109/TIT.2016.2549038](https://doi.org/10.1109/TIT.2016.2549038).
8. X. Zhang and P. H. Siegel, "Efficient Iterative LP Decoding of LDPC Codes
   with Alternating Direction Method of Multipliers," in *Proc. 2013 IEEE
   International Symposium on Information Theory (ISIT)*, Istanbul, Turkey,
   Jul. 2013, pp. 1501–1505,
   doi: [10.1109/ISIT.2013.6620477](https://doi.org/10.1109/ISIT.2013.6620477).
9. F. Gensheimer, T. Dietz, K. Kraft, S. Ruzika, and N. Wehn, "A
   Reduced-Complexity Projection Algorithm for ADMM-Based LP Decoding," *IEEE
   Transactions on Information Theory*, vol. 66, no. 8, pp. 4819–4833, Aug. 2020,
   doi: [10.1109/TIT.2020.2977074](https://doi.org/10.1109/TIT.2020.2977074)
   (arXiv:1901.03240).
10. J. X. Li and P. O. Vontobel, "LP Decoding of Quantum Stabilizer Codes," in
    *Proc. 2018 IEEE International Symposium on Information Theory (ISIT)*,
    Vail, CO, USA, Jun. 2018, pp. 1306–1310,
    doi: [10.1109/ISIT.2018.8437905](https://doi.org/10.1109/ISIT.2018.8437905);
    see also O. Fawzi, L. Grouès, and A. Leverrier, "Linear Programming Decoder
    for Hypergraph Product Quantum Codes," in *Proc. 2021 IEEE Information
    Theory Workshop (ITW)*, Apr. 2021, pp. 1–5,
    doi: [10.1109/ITW46852.2021.9457611](https://doi.org/10.1109/ITW46852.2021.9457611).
11. K. Yin, X. Fang, J. Ruan, H. Zhang, D. Tullsen, A. Sornborger, C. Liu,
    A. Li, T. Humble, and Y. Ding, "SymBreak: Mitigating Quantum Degeneracy
    Issues in QLDPC Code Decoders by Breaking Symmetry," arXiv:2412.02885,
    Dec. 2024, url: [https://arxiv.org/abs/2412.02885](https://arxiv.org/abs/2412.02885).
12. S. Koutsioumpas, H. Sayginel, M. Webster, and D. E. Browne, "Automorphism
    Ensemble Decoding of Quantum LDPC Codes," arXiv:2503.01738, Mar. 2025,
    url: [https://arxiv.org/abs/2503.01738](https://arxiv.org/abs/2503.01738).
13. A. S. Maan, F. M. Garcia-Herrero, A. Paler, and V. Savin, "Decoding
    correlated errors in quantum LDPC codes," *Nature Communications*, vol. 17,
    Art. no. 3965, Mar. 2026,
    doi: [10.1038/s41467-026-70556-3](https://doi.org/10.1038/s41467-026-70556-3).
14. J. Blue, H. Avlani, Z. He, L. Ziyin, and I. L. Chuang, "Machine Learning
    Decoding of Circuit-Level Noise for Bivariate Bicycle Codes," *Quantum*,
    vol. 10, p. 2149, Jun. 2026,
    doi: [10.22331/q-2026-06-30-2149](https://doi.org/10.22331/q-2026-06-30-2149)
    (arXiv:2504.13043).
15. C. Gidney, "Stim: a fast stabilizer circuit simulator," *Quantum*, vol. 5,
    p. 497, Jul. 2021,
    doi: [10.22331/q-2021-07-06-497](https://doi.org/10.22331/q-2021-07-06-497).
