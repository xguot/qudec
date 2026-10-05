# LP-family decoding under phenomenological noise for a bivariate bicycle code

## Abstract

DRAFT. Belief propagation with ordered statistics decoding (BP+OSD) is the
standard decoder for quantum low-density parity-check (QLDPC) codes. Gu and
Soleimanifar (IEEE Trans. Inf. Theory 72(10), 2026) showed that linear
programming (LP) decoding with OSD post-processing slightly outperforms
BP+OSD for bivariate bicycle codes at code capacity, and flagged the
question of whether the advantage survives more realistic noise. We
implement the parity-polytope relaxation solved by exact LP and by the
alternating direction method of multipliers (ADMM), and benchmark them
against BP+OSD under phenomenological noise on the [[72,12,6]] bivariate
bicycle code. Plain ADMM+OSD loses the code-capacity edge and trails
BP+OSD at three of four rates. The LLR-weighted variant prescribed by the
paper's Appendix C, with a normalization the paper omits, recovers parity
and edges BP+OSD at the two lowest rates, [CONFIRM: 2000-shot verdict].
An LDR variant with an adaptive penalty shows no consistent gain at four
outer iterations. We also report an implementation finding: the naive
LLR weighting collapses ADMM to the trivial fixed point, and normalizing
the weights is required for convergence.

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
bivariate bicycle codes up to [[288,12,18]]. Their conclusion states,
verbatim, that determining whether the advantage survives circuit-level
or phenomenological noise is the important open question; their Appendix
C sketches the extension but reports no numerics. This paper runs that
experiment at the phenomenological level.

## 2. Methods

### 2.1 Codes and noise

Bivariate bicycle codes [[72,12,6]] and [[144,12,12]] [Bravyi et al.
2024]. Phenomenological noise over d = 6 rounds: each round applies
i.i.d. depolarizing data errors (marginal rate 2p/3 per Pauli) and
syndrome measurement flips at rate p. The detector syndrome is
delta_t = h e_t xor m_t xor m_{t-1}, giving a block-bidiagonal
time-expanded check matrix with d data blocks and d measurement blocks.

### 2.2 Decoders

- **BP+OSD**: min-sum BP, 30 iterations, OSD with a one-pass combination
  sweep (OSD-CS, lambda = 1).
- **LP+OSD**: the exact LP relaxation (Eq. 2 of Gu-Soleimanifar),
  solved with HiGHS, qubit indicators fed to OSD with Tanner-distance
  tie breaking (their Sec. V). Code capacity only; the subset
  formulation grows exponentially at detector scale.
- **ADMM+OSD**: the parity-polytope relaxation (their QP (8) with g = 0,
  c = 1), plain ADMM with rho = 2.0. The projection onto the parity
  polytope is exact, computed from the two-slice representation
  [Barman et al.] with a segment walk that resolves roots at clip and
  sort-switch breakpoints.
- **ADMM-w+OSD**: per-column LLR weights per their Appendix C (data
  columns weighted by 2p/3, measurement columns by p), normalized by
  the maximum weight. The normalization leaves the LP optimum invariant
  but is required for convergence: without it the first x-update clips
  every coordinate to zero and the iteration locks at the trivial fixed
  point (Section 4.2).
- **LDR+OSD**: the adaptive-penalty variant of their companion ISIT
  2026 paper, with a diminishing step beta/sqrt(k+1) and four outer
  iterations, warm-started between inner solves.

All decoders share the same OSD post-processing and logical checks;
corrections are validated against the detector syndromes, and runs with
any invalid correction are flagged.

### 2.3 Benchmark protocol

Fixed seeds, 300 shots per point for the comparison sweep, 2000 shots
for the confirmation points [CONFIRM], zero invalid corrections
throughout. Logical error rate estimated per X and Z logical sectors
jointly.

## 3. Results

### 3.1 Code capacity

[[144,12,12]], 500 shots (bposd reference: 20000):

- p=0.03: LP 0.000 · BP 0.002 · bposd 0.00135
- p=0.05: LP 0.004 · BP 0.008 · bposd 0.00950
- p=0.08: LP 0.096 · BP 0.088 · bposd 0.08475
- p=0.10: LP 0.294 · BP 0.302 · bposd 0.23210

LP+OSD sits at or below BP+OSD at three of four points, reproducing the
paper's qualitative claim at the settings used here.

### 3.2 Phenomenological noise

[[72,12,6]], d = 6, 300 shots:

- p=0.005: BP 0.00667 · ADMM 0.00333 · ADMM-w 0.00333 · LDR 0.00333
- p=0.010: BP 0.03333 · ADMM 0.04333 · ADMM-w 0.02333 · LDR 0.03000
- p=0.020: BP 0.10000 · ADMM 0.14667 · ADMM-w 0.10000 · LDR 0.13000
- p=0.030: BP 0.26333 · ADMM 0.30333 · ADMM-w 0.28333 · LDR 0.32667

Per-point gaps are sub-2-sigma at 300 shots. Plain ADMM trails BP at
three of four rates. The weighted variant recovers parity, ahead at the
two lowest rates and marginally behind at p = 0.030. [CONFIRM: replace
or reinforce with the 2000-shot results, including error bars.]

### 3.3 The weighting normalization

Applying the Appendix C weights verbatim produces all-zero corrections:
with c_i near 5 the first x-update clips every coordinate to zero and
the ADMM iterates stay at the trivial fixed point regardless of
iteration count. Normalizing the weights by their maximum (objective
scaling leaves the LP optimum invariant) restores the plain-ADMM
convergence profile. The paper does not mention this normalization.

## 4. Discussion

The measured answer to the open question, at these settings: the
LP-family code-capacity advantage over BP+OSD does not survive
phenomenological noise for plain ADMM; the properly weighted variant
recovers parity, with a modest low-rate edge that the 2000-shot
confirmation [CONFIRM] does or does not firm. Limitations: one code
size at phenom level [CONFIRM: add [[144,12,12]] results], 300-shot
comparison points, our choice of LDR step size and outer-iteration
count, and OSD-CS at lambda = 1 rather than the reference lambda = 60.
Circuit-level noise via stim/sinter is the natural next step.

## References

[DRAFT: Bravyi et al. Nature 2024; Roffe et al. PRR 2020; Gu and
Soleimanifar TIT 2026 and ISIT 2026; Barman et al. IEEE TIT 2013; Li and
Vontobel ISIT 2018; SymBreak arXiv:2412.02885.]
