"""Aggregate JSONL memory benchmarks and compare variants with paired tests.

  python scripts/analyze_phenom.py results/phenom_72_*.jsonl --baseline bp

Different source/protocol/channel/OSD settings remain separate. Reused
seeds for the same variant are rejected, preventing overlapping reruns.
Paired tests require matching seeds and sample hashes for both variants.
P values are uncorrected for multiple comparisons.
For the tune script's lambda group, select --baseline bp-lam.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from qudec.stats import decode_outcomes, mcnemar, two_proportion


def analyze(records, baseline="bp"):
    groups = defaultdict(lambda: defaultdict(dict))
    for record in records:
        # Preserve algorithm-version and common postprocessing boundaries.
        group = (record["code"], record["d"], record["p"], record["p_meas"],
                 record["model"], record["protocol"], record["projection"],
                 record["environment"]["source_sha256"],
                 record["decoder_config"].get("osd_order", 1),
                 record["decoder_config"].get("osd_lam"))
        by_variant = groups[group]
        variant, sample_id, seed = record["variant"], record["sample_id"], record["seed"]
        if any(cell_seed == seed for cell_seed, _ in by_variant[variant]):
            raise ValueError(f"reused seed for {variant}: {seed}")
        # A variant name must denote identical settings across independent seeds.
        existing = next(iter(by_variant[variant].values()), None)
        if existing and record["decoder_config"] != existing["decoder_config"]:
            raise ValueError(f"inconsistent decoder settings for {variant}")
        decode_outcomes(record)
        by_variant[variant][(seed, sample_id)] = record
    summaries = []
    for group, variants in sorted(groups.items(), key=lambda pair: str(pair[0])):
        summary = dict(zip(("code", "d", "p", "p_meas", "model", "protocol",
                            "projection", "source_sha256", "osd_order", "osd_lam"), group))
        summary["variants"] = {}
        for name, cells in sorted(variants.items()):
            shots = sum(r["shots"] for r in cells.values())
            failures = sum(r["failure"] for r in cells.values())
            summary["variants"][name] = {"shots": shots, "failures": failures,
                                        "ler": failures/shots}
        summary["comparisons"] = {}
        if baseline in variants:
            reference = variants[baseline]
            for name, cells in sorted(variants.items()):
                if name == baseline:
                    continue
                a, b = summary["variants"][baseline], summary["variants"][name]
                comparison = {"baseline": baseline, "two_proportion": two_proportion(
                    a["failures"], a["shots"], b["failures"], b["shots"])}
                if cells.keys() == reference.keys():
                    ids = sorted(cells)
                    outcome_a = np.concatenate([decode_outcomes(reference[i]) for i in ids])
                    outcome_b = np.concatenate([decode_outcomes(cells[i]) for i in ids])
                    comparison["mcnemar"] = mcnemar(outcome_a, outcome_b)
                else:
                    comparison["paired_unavailable"] = "seeds or sample hashes differ; no paired test computed"
                summary["comparisons"][name] = comparison
        summary["p_values_multiple_testing_corrected"] = False
        summaries.append(summary)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--baseline", default="bp")
    args = parser.parse_args(argv)
    records = []
    for path in args.files:
        with path.open() as fh:
            records.extend(json.loads(line) for line in fh if line.strip())
    try:
        results = analyze(records, args.baseline)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
