# LP-family decoding under phenomenological noise for a bivariate bicycle code

## Abstract

DRAFT — CORRECTED RESULTS PENDING. We study whether optimization-based
quantum LDPC decoders retain an advantage over BP+OSD under measurement
noise. The implementation includes exact LP decoding at code capacity,
batched parity-polytope ADMM variants, and phenomenological memory
experiments on bivariate bicycle codes. An implementation audit found
incorrect BP messages at tied reliabilities and a capped ADMM projection
that could violate the intended relaxation. Both affected the historical
comparisons reported below. The corrected benchmark also matches BP data
and measurement priors, closes the experiment with a perfect terminal
syndrome, and retains paired per-shot outcomes. Corrected high-statistics
results are required before concluding that an advantage survives or
disappears. The numerical tables in this draft are historical provenance,
not validated evidence for a decoder ranking.

## 1. Introduction

Quantum LDPC codes promise fault tolerance with reduced qubit overhead
compared to surface codes. Their practical adoption hinges on the
classical decoder: belief propagation (BP) with ordered statistics
decoding (OSD) post-processing is the incumbent [Roffe, Bravyi]. BP
suffers on degenerate codes because short cycles create ambiguities
[SymBreak, Gu-Soleimanifar].

Optimization-based decoding is the alternative: the maximum-likelihood
decoding problem relaxes to a linear program [Feldman; Li-Vontobel], and
Gu and Soleimanifar show LP+OSD outperforms BP+OSD at code capacity for
bivariate bicycle codes up to [[288,12,18]]. Their conclusion motivates evaluating the decoder under more realistic
noise. This project investigates the phenomenological extension; the
literature review and exact citation details remain to be verified before
submission.

## 2. Methods

### 2.1 Codes and noise

Bivariate bicycle codes [[72,12,6]] and [[144,12,12]] [Bravyi et al.
2024]. Phenomenological noise over d = 6 rounds: each round applies
i.i.d. depolarizing data errors (marginal rate 2p/3 per Pauli) and
syndrome measurement flips at rate p. The detector syndrome is
delta_t = h e_t xor m_t xor m_{t-1}, giving a block-bidiagonal
time-expanded check matrix with d data blocks and d measurement blocks.
The corrected protocol adds a noiseless terminal measurement, giving an
extra detector block delta_(d+1) = m_d. The historical tables used an open
terminal boundary and logical-only scoring. These protocols are distinct
and their estimates must not be pooled.

### 2.2 Decoders

- **BP+OSD**: min-sum BP, 30 iterations, OSD with a one-pass combination
  greedy sweep, or the explicitly selected exhaustive
  OSD-CS(lambda) variant. Corrected BP uses per-column priors 2p/3 for data
  and p for measurement errors.
- **LP+OSD**: the exact LP relaxation (Eq. 2 of Gu-Soleimanifar),
  solved with HiGHS, qubit indicators fed to OSD with Tanner-distance
  tie breaking (their Sec. V). Code capacity only; the subset
  formulation grows exponentially at detector scale.
- **ADMM+OSD**: the parity-polytope relaxation (their QP (8) with g = 0,
  c = 1), plain ADMM with rho = 2.0 (tuned runs use rho = 4.0, chosen
  by the historical Section 3.2 sweep; the corrected solver needs fresh
  tuning). The corrected projection uses a forbidden-set cut search [Gensheimer et al.] and
  a scalar root solve to floating-point precision. It does not truncate
  the available parity slices or fall back to box-only clipping.
- **ADMM-w+OSD**: per-column LLR weights per their Appendix C (data
  columns weighted by 2p/3, measurement columns by p), normalized by
  the maximum absolute weight. Positive scaling preserves the LP optimum
  and changes the relative objective/penalty scale in finite-iteration
  ADMM. The historical claim that normalization is necessary for
  convergence must be reassessed after the projection repair.
- **LDR+OSD**: the adaptive-penalty variant of their companion ISIT
  2026 paper, with a diminishing step beta/sqrt(k+1) and four outer
  iterations, warm-started between inner solves.

All decoders share the same OSD post-processing and logical checks;
corrections are validated against the detector syndromes, and runs with
any invalid correction are flagged.

### 2.3 Benchmark protocol

Historical runs used 300-shot explorations, 2000-shot confirmations,
and selected 10000-shot four-seed cells. They reported no invalid final
OSD corrections; this does not validate the preceding BP or ADMM steps.

Corrected runs sample one error set for all variants and decode it in
bounded batches. They save integer failure counts, per-shot failure and
residual-syndrome indicators, a sample fingerprint, protocol version,
solver parameters, environment, and source revision. Closed-memory
failure includes logical flips, residual data syndrome, or invalid
detector corrections. Paired McNemar comparisons use the same saved
sample set; independent two-proportion statistics are descriptive only
when outcomes are paired. Reruns must disclose tuning/selection and
multiple comparisons. BP/OSD execute on CPU; ADMM can execute on CUDA.

## 3. Historical results — require corrected reruns

All in-tree decoder comparisons in this section predate the correctness
repairs. Point estimates are retained for traceability; the earlier
advantage/deficit and equivalence interpretations are withdrawn.

### 3.1 Code capacity

[[144,12,12]], 500 shots (bposd reference: 20000):

- p=0.03: LP 0.000 · BP 0.002 · bposd 0.00135
- p=0.05: LP 0.004 · BP 0.008 · bposd 0.00950
- p=0.08: LP 0.096 · BP 0.088 · bposd 0.08475
- p=0.10: LP 0.294 · BP 0.302 · bposd 0.23210

The external reference and in-tree point estimates used different shot
counts. The in-tree BP baseline must be rerun before interpretation.

### 3.2 Phenomenological noise

[[72,12,6]], d = 6, 300 shots (initial sweep):

- p=0.005: BP 0.00667 · ADMM 0.00333 · ADMM-w 0.00333 · LDR 0.00333
- p=0.010: BP 0.03333 · ADMM 0.04333 · ADMM-w 0.02333 · LDR 0.03000
- p=0.020: BP 0.10000 · ADMM 0.14667 · ADMM-w 0.10000 · LDR 0.13000
- p=0.030: BP 0.26333 · ADMM 0.30333 · ADMM-w 0.28333 · LDR 0.32667

[[72,12,6]], d = 6, 2000 shots (confirmation):

- p=0.005: BP 0.00300 · ADMM 0.00250 · ADMM-w 0.00350
- p=0.010: BP 0.02000 · ADMM 0.02150 · ADMM-w 0.01800
- p=0.020: BP 0.11800 · ADMM 0.13800 · ADMM-w 0.12450

The earlier 3.9-sigma claim for 0.138 versus 0.118 at 2000 shots per
decoder was incorrect under the stated independent two-proportion method:
the pooled value is approximately 1.89 sigma. For 0.1245 versus 0.118 it
is approximately 0.63 sigma, rather than 1.3. Because variants share
sampled errors, valid paired inference requires discordant outcome counts
that the old runs did not retain. These arithmetic corrections do not
validate the historical decoder implementations.

Complete fan-out at rho = 4, four independent 2500-shot seeds
(10^4 shots per cell):

- p=0.020, greedy: BP 0.11700 · ADMM 0.12170 · ADMM-w 0.11000
- p=0.020, OSD-CS(60): BP 0.11440 · ADMM 0.11040 · ADMM-w 0.10200
- p=0.010, greedy: BP 0.02230 · ADMM 0.02220 · ADMM-w 0.02110

The weighted OSD-CS(60) point is 0.0124 below the historical BP estimate.
A descriptive independent two-proportion calculation is about 2.8 sigma;
it is neither a valid paired test nor evidence of an advantage after the
implementation audit. The setting was selected among several explored
variants. It should be rerun with the corrected protocol and independently
replicated after settings are fixed.

### 3.3 Preliminary code-size data

[[144,12,12]], d = 6, 300 shots:

- p=0.005: BP 0.00667 · ADMM 0.01000 · ADMM-w 0.00333
- p=0.010: BP 0.03333 · ADMM 0.05667 · ADMM-w 0.04333

These are exploratory point estimates. Their ordering differs between
rates; corrected confirmation counts on this code are outstanding. A
previous high-shot attempt exhausted memory before producing results.

### 3.4 The weighting normalization

The pre-audit implementation produced poor behavior with unnormalized
LLR weights. This observation cannot be separated from its projection
and stopping defects. The corrected solver rescales a linear objective
by a positive factor for conditioning; it preserves the LP optimum.
When a quadratic LDR penalty is present, the linear term is not
independently renormalized, which would change the intended objective.
Convergence sensitivity and any normalization requirement need new tests
at the reported benchmark settings.

## 4. Discussion

The scientific question remains open within this project. The historical
comparisons do not establish a loss, retention, or recovery of an ADMM
advantage because both the BP baseline and ADMM relaxation path had
correctness defects. The corrected implementation has regression coverage
against independent message and convex-projection oracles, but those
checks are not a substitute for corrected high-statistics experiments.

Next steps are a fresh matched-prior tuning/baseline run, independent
confirmation of fixed settings, larger-code validation, and manuscript
integration. Protocol sensitivity should compare closed and explicitly
legacy open terminal boundaries without pooling their outcomes.
Remaining limitations include finite ADMM iteration budgets, exploratory
LDR settings, GPU nondeterminism, unequal CPU/GPU execution, and an
optional preliminary circuit experiment. A nonsignificant comparison
does not establish statistical equivalence.

## References

[DRAFT: Bravyi et al. Nature 2024; Roffe et al. PRR 2020; Gu and
Soleimanifar TIT 2026 and ISIT 2026; Barman et al. IEEE TIT 2013; Li and
Vontobel ISIT 2018; SymBreak arXiv:2412.02885.]

Projection reference: F. Gensheimer, T. Dietz, K. Kraft, S. Ruzika,
and N. Wehn, “A Reduced-Complexity Projection Algorithm for ADMM-based
LP Decoding,” arXiv:1901.03240 (2019), Section IV.
[arXiv:1901.03240](https://arxiv.org/abs/1901.03240)
