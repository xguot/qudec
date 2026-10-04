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

Phenomenological noise, [[72,12,6]], d = 6 rounds, 300 shots (Rivanna A40):

- p=0.005: ADMM 0.00333 · BP 0.00667
- p=0.010: ADMM 0.04667 · BP 0.03333
- p=0.020: ADMM 0.14667 · BP 0.10000
- p=0.030: ADMM 0.30333 · BP 0.26333

Zero invalid corrections throughout. Per-point gaps are sub-2-sigma, but
the ADMM trails BP at the three highest rates, consistently on the same
side. The measured answer to the open question, at these settings, is
that the small code-capacity advantage of plain ADMM+OSD does not
survive phenomenological noise: it inverts into a modest, consistent
shortfall. The LLR-weighted and LDR variants remain untested and are the
natural follow-up.

## Known limitations

- The ADMM implementation is the paper's plain variant, not their
  LDR-ADMM with adaptive penalties, which is where their strongest
  claims live.
- Untuned ADMM parameters (rho = 2.0, no over-relaxation) and a uniform
  objective; the paper's Appendix C prescribes per-mechanism LLR
  weights, which matter when measurement and data error rates differ.
- OSD post-processing uses the lambda = 1 combination sweep rather than
  the lambda = 60 of the reference implementation.
- The LP solver (HiGHS) scales to code-capacity degrees only; the ADMM
  covers the time-expanded detector systems.

## Roadmap

- Sharpen the [[72,12,6]] comparison with more shots and test the
  LLR-weighted objective and the LDR-ADMM adaptive penalty, the natural
  strengthenings of the plain variant.
- Run the [[144,12,12]] code for the code-size dimension.
- Circuit-level noise through stim/sinter.
