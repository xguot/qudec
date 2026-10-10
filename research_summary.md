# Research Summary: LP and ADMM Decoding of Quantum LDPC Codes

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

No published work reports LP-family decoding results under
phenomenological or circuit-level noise for bivariate bicycle codes.
This project runs that experiment.

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
    z-projection onto the parity polytope is exact, via the two-slice
    representation of Barman et al. and a segment walk that resolves
    roots at clip and sort-switch breakpoints.
- **Noise models**: i.i.d. code capacity, and phenomenological noise
  (`src/qudec/phenom.py`) with per-round data and measurement errors on
  the time-expanded detector system
  `delta_t = h e_t xor m_t xor m_{t-1}`.
- **Verification** (`tests/`): brute-force MLD comparisons on small
  codes, the repetition-code MLD rate, and the ADMM-vs-exact-LP
  objective gate across all Steane syndromes.

## Results so far

Code capacity, [[144,12,12]] gross code (Rivanna, 500 shots for the
in-tree decoders, 20000 for the bposd reference):

- p=0.03: LP 0.000 · BP 0.002 · bposd 0.00135
- p=0.05: LP 0.004 · BP 0.008 · bposd 0.00950
- p=0.08: LP 0.096 · BP 0.088 · bposd 0.08475
- p=0.10: LP 0.294 · BP 0.302 · bposd 0.23210

LP+OSD sits at or below BP+OSD at three of four points and below the
20000-shot reference at low p, reproducing the paper's qualitative claim.
Zero invalid corrections throughout.

Phenomenological noise, [[72,12,6]], d = 6 rounds, 300 shots
(Rivanna, A6000/A40). Logical error rates for BP+OSD, plain ADMM+OSD,
the LLR-weighted ADMM+OSD (Appendix C weights, normalized), and
LDR-ADMM+OSD with four outer iterations:

- p=0.005: BP 0.00667 · ADMM 0.00333 · ADMM-w 0.00333 · LDR 0.00333
- p=0.010: BP 0.03333 · ADMM 0.04333 · ADMM-w 0.02333 · LDR 0.03000
- p=0.020: BP 0.10000 · ADMM 0.14667 · ADMM-w 0.10000 · LDR 0.13000
- p=0.030: BP 0.26333 · ADMM 0.30333 · ADMM-w 0.28333 · LDR 0.32667

Zero invalid corrections throughout; per-point gaps are sub-2-sigma at
300 shots. Plain ADMM loses the code-capacity edge and trails BP at
three of four rates. The LLR-weighted variant recovers parity, edging
BP at the two lowest rates and trailing marginally at p = 0.030. The
LDR variant shows no consistent gain at four outer iterations.

Implementation finding: applying the Appendix C weights naively
collapses the ADMM to the trivial all-zero fixed point on the first
update; normalizing the weights by their maximum (which leaves the LP
optimum invariant) restores proper convergence. The paper does not
mention this normalization.

2000-shot confirmation, [[72,12,6]], d = 6:

- p=0.005: BP 0.00300 · ADMM 0.00250 · ADMM-w 0.00350
- p=0.010: BP 0.02000 · ADMM 0.02150 · ADMM-w 0.01800
- p=0.020: BP 0.11800 · ADMM 0.13800 · ADMM-w 0.12450

Preliminary [[144,12,12]], d = 6, 300 shots:

- p=0.005: BP 0.00667 · ADMM 0.01000 · ADMM-w 0.00333
- p=0.010: BP 0.03333 · ADMM 0.05667 · ADMM-w 0.04333

Measured answer to the open question: the LP-family code-capacity
advantage over BP+OSD does not survive phenomenological noise. Plain
ADMM degrades significantly at p = 0.020 (0.138 versus 0.118, 3.9 sigma
at 2000 shots); the properly weighted variant repairs this to
statistical parity everywhere, with no significant advantage. The
300-shot hints of a low-rate edge wash out at 2000 shots.

### Tuning sweep ([[72,12,6]], d = 6, p = 0.020, 300 shots)

Plain ADMM over rho x over-relaxation alpha, plus OSD-CS(lambda = 60)
on both BP and ADMM (raw table in results/tune_72.csv):

- bp 0.10000; bp-lam60 0.10000 (OSD-CS(60) does not move the BP
  baseline at this point)
- admm at rho = 2 (paper default) 0.14667; admm-lam60 0.13667
- rho = 1 0.16333; rho = 2 0.14667; rho = 4 0.11667
- alpha at rho = 4: 1.0 0.11667, 1.5 0.11667, 1.8 0.12000 (weak effect)

Rho = 4 removes most of the plain-ADMM deficit seen at the paper's
default rho = 2: the gap to BP+OSD shrinks from 0.047 to 0.017. The
negative result survives but narrows: tuned plain ADMM still trails
BP+OSD at p = 0.020. These are 300-shot numbers; confirm rho = 4 at
2000 shots before changing the paper claims.

## Known limitations

- The LDR variant uses a diminishing beta/sqrt(k+1) step and four outer
  iterations; the paper does not pin beta_k, so this variant is our
  choice rather than their verbatim algorithm.
- Untuned ADMM parameters (rho = 2.0, no over-relaxation).
- OSD post-processing uses the lambda = 1 combination sweep rather than
  the lambda = 60 of the reference implementation.
- The LP solver (HiGHS) scales to code-capacity degrees only; the ADMM
  covers the time-expanded detector systems.
- 300 shots leave the phenom gaps sub-significant; a 2000-shot run
  would firm the low-rate weighted advantage.

## Roadmap

- Tune ADMM (rho, over-relaxation, OSD lambda) - swept at 300
  shots; rho = 4 removes most of the plain-ADMM deficit at
  rho = 2. Confirm rho = 4 at 2000 shots.
- Raise the [[72,12,6]] confirmation to 10^4-10^5 shots per point.
- Complete the [[144,12,12]] phenom comparison at confirmation shot
  counts.
- Preliminary circuit-level noise via stim/sinter on a small code.

See docs/plan.md for the ordered plan, per-item rationale, and the
Dec 15 timeline.
