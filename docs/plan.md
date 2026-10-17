# Pre-submission plan

Ordered by cost; items 1-3 run on Rivanna (tongx gpu allocation,
A6000/A40), item 4 is a small-code stretch. Roughly nine weeks to the
Dec 15 deadline. Jobs sync through hpc/sync.sh push/pull and land in
hpc/logs; results are recorded in research_summary.md.

## 0. Correctness gate before new research runs

The audit found tied-minimum BP messages, capped ADMM projections,
unmatched BP measurement priors, and circuit-pipeline bugs. The code
repairs and focused regression checks are implemented. Historical
comparisons are retained for provenance and require reruns; their old
advantage/deficit conclusions are withdrawn.

New phenomenological runs use full parity projection, matched per-column
BP priors, bounded decoder batches, a perfect terminal syndrome by
default, and saved per-shot failures for paired inference. The explicit
legacy open boundary is a separate protocol. Run local tests and circuit
smoke checks before cluster submission, then rerun the baseline and
retune with the corrected implementation.

## 1. ADMM tuning sweep (do first)

Historical benchmarks used rho = 2.0, no over-relaxation, and a greedy
OSD sweep. A sweep against the exhaustive reference lambda = 60
measures finite-iteration sensitivity before settings are fixed for
independent confirmation.

- Sweep rho (roughly 0.5 to 8) and the over-relaxation alpha (1.0 to
  1.8) on the [[72,12,6]] phenom points where plain ADMM trails,
  especially p = 0.020. The old 0.138 versus 0.118 estimates do not
  support a corrected decoder comparison.
- Sweep the OSD-CS lambda at the reference value 60 against the
  current greedy sweep.
- Re-run the 2000-shot confirmation with the tuned parameters and
  record the sensitivity table, so the essay reports the sweep rather
  than a single untuned point.

## 2. Higher-statistics decoder comparison

The historical verdict is withdrawn after the implementation audit.
Raise the [[72,12,6]] confirmation to 10^4 and then 10^5
shots per point at p = 0.005, 0.010, 0.020 for bp, admm, admm-w.
Report corrected effect sizes, uncertainty, and paired comparisons.
A nonsignificant result does not establish equivalence. Replicate any
selected promising setting on independent seeds before asserting an edge.

- Measure corrected wall time before sizing larger jobs; extend
  hpc/run_phenom_confirm.slurm accordingly.

## 3. Complete the [[144,12,12]] phenom comparison

The code-size comparison needs corrected evidence on both instances.
The historical larger-code data contains only 300 shots per point, and
the attempted 2500-shot confirmation exhausted memory.

- Run hpc/run_phenom_144.slurm at confirmation shot counts (10^4 to
  10^5) for bp, admm, admm-w over p = 0.005, 0.010, 0.020.
- Fold the result into docs/paper_draft.md section 3.3 and the
  discussion.

## 4. Stretch: preliminary circuit-level noise via Stim

The essay's actual question is what survives under circuit-level
noise. Preliminary numbers on a small code upgrade the opening hook
from question to question-plus-evidence.

- Build a stim circuit for a small code with few syndrome-extraction
  rounds, sample directly through Stim, and decode with the unpacked
  decode_batch array interface. A native Sinter adapter is not implemented.
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

- Shared tie-safe BP and scalar/per-column priors are implemented.
- Full forbidden-set parity projection replaces capped slice search;
  tests compare both parities and higher-degree checks to an independent QP.
- OSD-CS(lambda), ADMM rho/alpha, and the BP reference scaling schedule
  are configurable. Corrected settings need a fresh tuning sweep.
- Phenom benchmarks expose terminal boundary and bounded batch size,
  reuse samples across variants, and save failure records/metadata for
  reproducible paired analysis. Confirmation jobs expose these options.
- The circuit builder tracks separate logical observables in both bases,
  applies measurement flip noise, and extracts per-mechanism DEM priors.
  Local Stim tests and smoke checks validate the corrected pipeline;
  cluster-scale circuit results remain outstanding.
- Existing 10000-shot historical cells cover greedy p = 0.010/0.020
  and OSD-CS(60) p = 0.020 on [[72,12,6]]. They require reruns after the
  repairs. No corrected high-statistics research results are claimed.
- Higher-statistics [[144,12,12]] results, independent promising-cell
  replication, corrected manuscript tables, and a verified bibliography
  remain outstanding.
