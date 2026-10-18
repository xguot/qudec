# Audited campaign 21501046

This bundle records the completed October 17, 2026 Rivanna campaign.
Its benchmark source revision is `9cf40da36661542c696695bd95294f8d741f91b4`.
All 75 artifacts passed integrity and output validation. See the
[research summary](../../../research_summary.md) for interpretation and
the [manuscript draft](../../paper_draft.md) for methods and results.

- `report.json`: original cluster numerical results, with corrected
  reporting labels. Confirmation groups identify Holm adjustment;
  `primary_multiple_testing` names the raw `mcnemar.p_value` and adjusted
  `holm_p_value` fields. Every original numerical value and the original
  creation timestamp are retained.
- `manifest.json`: exact original task plan, statistical configuration,
  source hash, and full-snapshot hash.
- `selected.json`: exact frozen settings and training-output hashes.
- `audit.json`: original and published report checksums, artifact and
  completion-marker checksums, environment, validation outcomes, and
  launcher-log checksums.

Raw per-shot outputs, the original report, source snapshot, and worker
logs remain in `results/reruns/21501046/`. Launcher/test output remains
in `hpc/logs/qudec_rerun_21501046.{out,err}`. These directories are
ignored by Git; this bundle is an aggregate record, not a replacement
for the raw campaign backup. Keep the frozen snapshot unchanged even
when current source or obsolete scripts are edited or removed.

The updated report generator was also run against a temporary copy of
the full campaign. Counts, group structure, and configuration matched
exactly after allowing the intended label changes. Floating-point
statistics agreed within absolute tolerance 1e-10 (largest difference
below 5e-13 between local SciPy 1.18.0 and cluster SciPy 1.15.3).
The published copy retains the original cluster numbers exactly.
Independent reconstruction from raw outcomes reproduced failure totals,
paired discordances, and exact paired tests. The original campaign files
were verified unchanged after the audit.

This campaign has 10,000 shots per decoder in each of 12 primary cells.
The independent replication is CS(60), not the significant greedy OSD
setting. Neither 100,000-shot confirmation nor the optional external
reference was run. Reporting and cleanup changes require no resubmission.
