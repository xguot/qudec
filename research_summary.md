# Research Summary: LP and ADMM Decoding of Quantum LDPC Codes

## Correctness audit and result status

The implementation audit invalidates conclusions drawn from the historical
in-tree comparisons below. BP removed every tied minimum when computing
extrinsic check messages; ADMM benchmark settings capped parity slices
and could fall back outside the parity polytope. BP also used a data-error
prior for measurement columns. Zero invalid OSD corrections do not test
these message or relaxation errors.

The corrected code shares a tie-safe BP implementation, uses full
forbidden-set parity projection, and matches per-column BP channel priors.
New runs close the noisy history with a perfect terminal syndrome by
default, decode in bounded batches, and retain per-shot failure outcomes
for paired comparisons. The explicit legacy open boundary remains
available for protocol sensitivity checks. These protocol versions must
not be pooled. Corrected high-statistics runs have not been completed.

The old 0.118 versus 0.138 result at 2000 shots per decoder is approximately
1.89 sigma under a pooled independent two-proportion calculation, not the
previously reported 3.9 sigma. Variants actually share sampled errors, so
paired inference requires discordant per-shot outcomes; those were not
saved by the old runs. The historical 2.8-sigma independent calculation
also cannot establish an advantage, given the implementation defects,
shared samples, and selection across multiple settings.

## What this project does

This project builds GPU-accelerated decoders for quantum low-density
parity-check (QLDPC) codes and asks one research question: do
optimization-based decoders — linear programming (LP) and the
alternating direction method of multipliers (ADMM) — retain their
code-capacity advantage over belief propagation with ordered statistics
decoding (BP+OSD) once measurement errors enter through the syndrome
extraction process?

The testbed is the [[144,12,12]] bivariate bicycle (BB) code introduced
by Bravyi et al. (Nature 2024), with the [[72,12,6]] code as the
smaller instance.

## Why it is open

Gu and Soleimanifar (IEEE Trans. Inf. Theory 72(10), 2026) showed that
LP decoding with OSD post-processing slightly outperforms BP+OSD for
bivariate bicycle codes up to [[288,12,18]] — but benchmarked only at
code capacity (perfect syndrome extraction). Their own conclusion
states, verbatim, that "it would be important to determine if, and for
what code sizes, our decoder shows an advantage in the more realistic
setting of circuit-level depolarizing noise," and their Appendix C
sketches the extension without running it. Their companion ISIT 2026
paper introduces the ADMM variant, likewise at code capacity.

The project investigates that extension. A current literature review is
still required before making an exhaustive novelty claim.

## Pipeline

- **Codes** (`src/qudec/codes.py`): bivariate bicycle, Steane, and
  repetition constructions with GF(2) linear algebra and logical bases.
- **Decoders**:
  - `src/qudec/bposd.py` — batched min-sum BP (PyTorch) with OSD
    post-processing; the baseline.
  - `src/qudec/lp.py` — the exact LP relaxation of Gu-Soleimanifar
    (their Eq. 2), solved with HiGHS, with OSD post-processing and
    Tanner-graph-distance tie breaking (their Sec. V).
  - `src/qudec/admm.py` — plain ADMM on the parity-polytope relaxation
    (their QP (8) with g = 0, c = 1), batched in PyTorch. The
    z-projection uses forbidden-set cut search followed by a bounded
    scalar root solve, without a parity-slice cap.
- **Noise models**: i.i.d. code capacity, and phenomenological noise
  (`src/qudec/phenom.py`) with per-round data and measurement errors on
  the time-expanded detector system
  `delta_t = h e_t xor m_t xor m_{t-1}`.
- **Verification** (`tests/`): brute-force MLD comparisons on small
  codes, the repetition-code MLD rate, and the ADMM-vs-exact-LP
  objective gate across all Steane syndromes.

## Historical results (require corrected reruns)

Code capacity, [[144,12,12]] gross code (Rivanna, 500 shots for the
in-tree decoders, 20000 for the bposd reference):

- p=0.03: LP 0.000 · BP 0.002 · bposd 0.00135
- p=0.05: LP 0.004 · BP 0.008 · bposd 0.00950
- p=0.08: LP 0.096 · BP 0.088 · bposd 0.08475
- p=0.10: LP 0.294 · BP 0.302 · bposd 0.23210

These are historical point estimates, with zero invalid corrections
reported. The in-tree BP baseline needs rerunning before a comparison
with LP or the independent reference can be interpreted.

Phenomenological noise, [[72,12,6]], d = 6 rounds, 300 shots
(Rivanna, A6000/A40). Logical error rates for BP+OSD, plain ADMM+OSD,
the LLR-weighted ADMM+OSD (Appendix C weights, normalized), and
LDR-ADMM+OSD with four outer iterations:

- p=0.005: BP 0.00667 · ADMM 0.00333 · ADMM-w 0.00333 · LDR 0.00333
- p=0.010: BP 0.03333 · ADMM 0.04333 · ADMM-w 0.02333 · LDR 0.03000
- p=0.020: BP 0.10000 · ADMM 0.14667 · ADMM-w 0.10000 · LDR 0.13000
- p=0.030: BP 0.26333 · ADMM 0.30333 · ADMM-w 0.28333 · LDR 0.32667

These exploratory estimates came from the pre-audit implementation.
The earlier weighting-normalization observation is also implementation-
dependent: objective scaling preserves the LP optimum, but its effect on
finite-iteration ADMM convergence must be reassessed after the projection
repair. A large LLR is not evidence of an inevitable all-zero fixed point.

2000-shot confirmation, [[72,12,6]], d = 6:

- p=0.005: BP 0.00300 · ADMM 0.00250 · ADMM-w 0.00350
- p=0.010: BP 0.02000 · ADMM 0.02150 · ADMM-w 0.01800
- p=0.020: BP 0.11800 · ADMM 0.13800 · ADMM-w 0.12450

Preliminary [[144,12,12]], d = 6, 300 shots:

- p=0.005: BP 0.00667 · ADMM 0.01000 · ADMM-w 0.00333
- p=0.010: BP 0.03333 · ADMM 0.05667 · ADMM-w 0.04333

No corrected comparison is available yet. The earlier negative verdict
and its statistical significance are withdrawn pending reruns.

### Tuning sweep ([[72,12,6]], d = 6, p = 0.020, 300 shots)

Plain ADMM over rho x over-relaxation alpha, plus OSD-CS(lambda = 60)
on both BP and ADMM (raw table in results/tune_72.csv):

- bp 0.10000; bp-lam60 0.10000 (OSD-CS(60) does not move the BP
  baseline at this point)
- admm at rho = 2 (paper default) 0.14667; admm-lam60 0.13667
- rho = 1 0.16333; rho = 2 0.14667; rho = 4 0.11667
- alpha at rho = 4: 1.0 0.11667, 1.5 0.11667, 1.8 0.12000 (weak effect)

The historical sweep motivated rho = 4 for subsequent exploratory runs.
The corrected solver and baseline require a fresh tuning sweep before
choosing settings for independent confirmation.

### Complete fan-out ([[72,12,6]], d = 6, rho = 4, 10^4 shots per cell)

Four independent 2500-shot seeds per setting. The 2000-shot runs
share seed 0 with the first fan-out cell and are superseded. Zero
invalid corrections throughout.

Greedy OSD sweep:
- p=0.020: BP 0.11700 · ADMM 0.12170 · ADMM-w 0.11000
- p=0.010: BP 0.02230 · ADMM 0.02220 · ADMM-w 0.02110

OSD-CS(lambda = 60):
- p=0.020: BP 0.11440 · ADMM 0.11040 · ADMM-w 0.10200

Deltas against BP at the same OSD setting (two-sample, pooled):
- greedy p=0.020: ADMM +0.00470 (1.0 sigma), ADMM-w -0.00700
  (-1.6 sigma)
- OSD-CS(60) p=0.020: ADMM -0.00400 (-0.9 sigma), ADMM-w -0.01240
  (-2.8 sigma)
- greedy p=0.010: ADMM -0.00010, ADMM-w -0.00120 (-0.6 sigma)

The historical weighted rho = 4, OSD-CS(60), p = 0.020 cell is a
candidate for a corrected rerun and independent replication (seeds 4-7).
It does not establish an advantage. GPU run-to-run variation and multiple
settings must be reflected in the final analysis.

## Known limitations and remaining work

- All historical in-tree decoder comparisons need corrected reruns.
- The LDR diminishing step and outer iteration budget are implementation
  choices; finite-iteration sensitivity must be measured.
- HiGHS is restricted to small/code-capacity systems; ADMM covers larger
  detector systems. Its finite iteration budget is not a convergence proof.
- BP/OSD run on CPU and ADMM may run on CUDA; timing results need hardware
  disclosure and do not yet establish GPU speedups against a matched baseline.
- The old runs did not retain paired failure outcomes. New runs do; use the
  paired analysis instead of treating identical sampled shots as independent.
- Complete the corrected [[72,12,6]] grid (including p = 0.005), then
  10^4-10^5-shot confirmations and [[144,12,12]] comparisons. The previous
  2500-shot larger-code attempt exhausted memory; bounded decoding batches
  replace allocating the full decoder workspace for every shot at once.
- Validate preliminary circuit data in both bases, then decide whether the
  optional circuit experiment belongs in the submission.
- Update the manuscript with corrected results, intervals, paired statistics,
  setting-selection disclosure, and a verified bibliography.

See docs/plan.md for the ordered plan and Dec 15 timeline.
