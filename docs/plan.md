# Pre-submission plan

Ordered by cost; items 1-3 run on Rivanna (tongx gpu allocation,
A6000/A40), item 4 is a small-code stretch. Roughly nine weeks to the
Dec 15 deadline. Jobs sync through hpc/sync.sh push/pull and land in
hpc/logs; results are recorded in research_summary.md.

## 1. ADMM tuning sweep (do first)

The repo's own limitations note rho = 2.0, no over-relaxation, and
OSD lambda = 1 against the reference lambda = 60. A rho/lambda sweep
is low-compute and pre-empts the obvious criticism that the phenom
loss is an untuned-ADMM artifact.

- Sweep rho (roughly 0.5 to 8) and the over-relaxation alpha (1.0 to
  1.8) on the [[72,12,6]] phenom points where plain ADMM trails,
  especially p = 0.020 (0.138 versus 0.118 at 2000 shots).
- Sweep the OSD-CS lambda at the reference value 60 against the
  current 1.
- Re-run the 2000-shot confirmation with the tuned parameters and
  record the sensitivity table, so the essay reports the sweep rather
  than a single untuned point.

## 2. Higher-statistics confirmation of statistical parity

The "no significant advantage" verdict rests on sub-2-sigma gaps at
2000 shots. Raise the [[72,12,6]] confirmation to 10^4 and then 10^5
shots per point at p = 0.005, 0.010, 0.020 for bp, admm, admm-w.
Either outcome is recorded as-is: parity becomes a statement, or the
weighted-ADMM edge appears and the story shifts to "the reweighted
variant recovers part of the advantage."

- Size the job from the 2000-shot wall time in hpc/logs and extend
  hpc/run_phenom_confirm.slurm accordingly.

## 3. Complete the [[144,12,12]] phenom comparison

The first question a skeptical reader asks is whether the negative
result survives on the larger code. The essay currently claims only
the [[72,12,6]] result; the 144 data is 300 shots and sub-2-sigma.

- Run hpc/run_phenom_144.slurm at confirmation shot counts (10^4 to
  10^5) for bp, admm, admm-w over p = 0.005, 0.010, 0.020.
- Fold the result into docs/paper_draft.md section 3.3 and the
  discussion.

## 4. Stretch: preliminary circuit-level noise via stim/sinter

The essay's actual question is what survives under circuit-level
noise. Preliminary numbers on a small code upgrade the opening hook
from question to question-plus-evidence.

- Build a stim circuit for a small code with few syndrome-extraction
  rounds, sample through sinter, and decode with the existing
  sinter-protocol decode_batch interface (already wired in
  bposd/lp/admm).
- Keep it to one or two codes, one or two error rates, and report it
  explicitly as preliminary.

## Timeline

Roughly nine weeks, Oct 10 to Dec 15.

- Weeks 1-2: item 1 and the first 10^4-shot confirmation.
- Weeks 3-6: 10^5-shot confirmation and the [[144,12,12]] runs.
- Weeks 7-9: item 4 and essay integration, with a re-run buffer.

Item 4 is deliberately deferred; if weeks 1-6 slip, item 4 is the
cut.

## Implementation status

- Decoder fidelity: OSD-CS(lambda = 60) exhaustive sweep,
  ADMM over-relaxation alpha, and the BP reference
  schedule 1 - 2**(-t) implemented; osd_lam/alpha
  exposed on all decoders. Defaults reproduce the
  committed benchmark behavior.
- bench_tune.py + hpc/run_tune.slurm cover item 1.
- bench_phenom.py --seed and the QUDEC_SHOTS/QUDEC_SEED
  fan-out on run_phenom_confirm.slurm and
  run_phenom_144.slurm cover items 2 and 3.
- Item 4: stim memory circuit, dem-matrix decoders,
  smoke validation, bench_circuit.py, and
  hpc/run_circuit.slurm implemented; awaiting cluster
  validation and runs.
