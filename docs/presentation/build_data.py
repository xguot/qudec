#!/usr/bin/env python3
"""Prepare the talk's figures and notes from audited, local source records.

Uses only the Python standard library. This does not run new experiments.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / "docs/results/21501046"
BUILD = HERE / "build"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def exact_mcnemar(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n)


def write_csv(name: str, fields: list[str], rows: list[dict]) -> None:
    with (BUILD / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def timestamp(seconds: int) -> str:
    return f"{seconds // 60}:{seconds % 60:02d}"


def main() -> None:
    report_path = CAMPAIGN / "report.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    audit = json.loads((CAMPAIGN / "audit.json").read_text())
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    require(report_digest == audit["published_report_sha256"], "Report checksum differs from audit")
    require(report["complete"] is True, "Campaign is incomplete")
    require(audit["completed_artifacts"] == 75, "Unexpected artifact coverage")
    primary = report["analyses"]["confirm"]
    require(len(primary) == 12, "Expected twelve primary cells")

    # Reconstruct the paired tests and their complete, predeclared Holm family.
    tests = []
    for group in primary:
        require(group["protocol"] == "closed-v1" and group["d"] == 6, "Unexpected protocol")
        require(group["p_meas"] == group["p"], "Measurement noise differs from data rate")
        for variant in group["variants"].values():
            require(variant["shots"] == 10000, "Unexpected primary sample size")
            require(math.isclose(variant["ler"], variant["failures"] / variant["shots"]), "Rate/count mismatch")
            lo, hi = variant["confidence_interval_95"]
            require(0 <= lo <= variant["ler"] <= hi <= 1, "Invalid marginal interval")
        for variant, comparison in group["comparisons"].items():
            paired = comparison["mcnemar"]
            raw_p = exact_mcnemar(paired["a_only"], paired["b_only"])
            require(math.isclose(raw_p, paired["p_value"], rel_tol=1e-10, abs_tol=1e-12), "Paired test mismatch")
            require(
                group["variants"]["bp"]["failures"] - group["variants"][variant]["failures"]
                == paired["a_only"] - paired["b_only"],
                "Discordant pairs do not match marginal counts",
            )
            tests.append((raw_p, comparison))
    require(len(tests) == report["primary_multiple_testing"]["comparisons"] == 24, "Incomplete test family")
    running = 0.0
    for rank, (raw_p, comparison) in enumerate(sorted(tests, key=lambda item: item[0])):
        running = max(running, min(1.0, (len(tests) - rank) * raw_p))
        require(math.isclose(running, comparison["holm_p_value"], abs_tol=1e-10), "Holm adjustment mismatch")
    require(sum(c["holm_p_value"] < 0.05 for _, c in tests) == 1, "Primary conclusion has changed")

    selected = [g for g in primary if g["code"] == "72" and g["p"] == 0.02 and g["osd_lam"] is None]
    require(len(selected) == 1, "Ambiguous main result")
    selected = selected[0]
    bp = selected["variants"]["bp"]
    plain = selected["variants"]["admm"]
    weighted = selected["variants"]["admm-w"]
    require((bp["failures"], weighted["failures"]) == (339, 294), "Narration needs updating")

    BUILD.mkdir(parents=True, exist_ok=True)
    figure_rows = []
    for variant, y in (("bp", 3), ("admm", 2), ("admm-w", 1)):
        result = selected["variants"][variant]
        lo, hi = result["confidence_interval_95"]
        rate = 100 * result["ler"]
        row = dict(
            y=y,
            rate=rate,
            lower_error=rate - 100 * lo,
            upper_error=100 * hi - rate,
            failures=result["failures"],
        )
        write_csv(f"primary_{variant}.csv", list(row), [row])
        figure_rows.append(dict(variant=variant, **row))

    circuit = report["circuit_preliminary"]
    circuit_cells = [("Z", 0.001), ("Z", 0.002), ("X", 0.001), ("X", 0.002)]
    for variant in ("bp", "admm"):
        rows = []
        for x, (basis, rate) in enumerate(circuit_cells, start=1):
            matches = [v for v in circuit if v["variant"] == variant and v["basis"] == basis and float(v["p"]) == rate]
            require(len(matches) == 1, "Missing or ambiguous circuit result")
            result = matches[0]
            failures, shots = int(result["failures"]), int(result["shots"])
            require(shots == 300 and result["code"] == "72" and result["d"] == "3", "Unexpected circuit protocol")
            lo, hi = result["confidence_interval_95"]
            rows.append(dict(x=x, failures=failures, lower_error=failures - shots * lo, upper_error=shots * hi - failures))
        write_csv(f"circuit_{variant}.csv", list(rows[0]), rows)

    all_rows = []
    for group in sorted(primary, key=lambda g: (int(g["code"]), g["osd_lam"] or 0, g["p"])):
        for variant, result in group["variants"].items():
            lo, hi = result["confidence_interval_95"]
            all_rows.append(dict(
                code=group["code"], p=group["p"], osd="greedy" if group["osd_lam"] is None else "CS(60)",
                variant=variant, shots=result["shots"], failures=result["failures"], rate=result["ler"],
                ci_lower=lo, ci_upper=hi,
                holm_p="" if variant == "bp" else group["comparisons"][variant]["holm_p_value"],
            ))
    write_csv("all_primary.csv", list(all_rows[0]), all_rows)

    relative_reduction = 100 * (bp["failures"] - weighted["failures"]) / bp["failures"]
    paired = selected["comparisons"]["admm-w"]
    macros = {
        "BPFailures": str(bp["failures"]),
        "PlainFailures": str(plain["failures"]),
        "WeightedFailures": str(weighted["failures"]),
        "BPRate": f"{100 * bp['ler']:.2f}",
        "PlainRate": f"{100 * plain['ler']:.2f}",
        "WeightedRate": f"{100 * weighted['ler']:.2f}",
        "RelativeReduction": f"{relative_reduction:.1f}",
        "AbsoluteReduction": f"{100 * (bp['ler'] - weighted['ler']):.2f}",
        "AdjustedP": f"{paired['holm_p_value']:.4f}",
        "BPOnlyFailures": str(paired["mcnemar"]["a_only"]),
        "WeightedOnlyFailures": str(paired["mcnemar"]["b_only"]),
    }
    (BUILD / "results.tex").write_text("% Generated from audited campaign 21501046.\n" + "\n".join(
        rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()
    ) + "\n")

    talk = json.loads((HERE / "narration.json").read_text())
    frame_count = (HERE / "qudec_phd_slides.tex").read_text().count(r"\begin{frame}")
    require(len(talk) == frame_count, "Each slide needs a narration entry")
    require(all(slide["seconds"] > 0 for slide in talk), "Slide timings must be positive")
    duration = sum(slide["seconds"] for slide in talk)
    require(duration <= 300, "Recording schedule exceeds five minutes")
    notes = []
    markdown = ["# Recording script", "", f"{len(talk)} slides in {timestamp(duration)}, including pauses.", ""]
    elapsed, word_count = 0, 0
    for index, slide in enumerate(talk, start=1):
        end = elapsed + slide["seconds"]
        words = len(slide["script"].split())
        word_count += words
        markdown.extend([
            f"## {index}. {slide['title']} ({timestamp(elapsed)}-{timestamp(end)})",
            "", slide["script"], "",
        ])
        notes.append(rf"\expandafter\def\csname talknote{index}\endcsname{{{tex_escape(slide['script'])}}}")
        elapsed = end
    markdown.extend([
        "## Recording cue", "",
        "Pause after the two failure counts on slide 4 so the audience can read the chart.", "",
    ])
    (BUILD / "notes.tex").write_text("% Generated from narration.json.\n" + "\n".join(notes) + "\n")
    (BUILD / "narration.md").write_text("\n".join(markdown))

    provenance = {
        "campaign": "21501046",
        "report": str(report_path.relative_to(ROOT)),
        "report_sha256": report_digest,
        "benchmark_source_revision": audit["environments"][0]["git_revision"],
        "primary_comparisons": len(tests),
        "primary_comparisons_below_005": 1,
        "main_result": figure_rows,
        "main_raw_p": paired["mcnemar"]["p_value"],
        "main_holm_p": paired["holm_p_value"],
        "narration_words": word_count,
        "target_seconds": elapsed,
        "new_experiments_run": False,
    }
    (BUILD / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified report hash, all 24 paired tests, and Holm adjustment. "
          f"Prepared {len(talk)} slides and {word_count} narration words for {timestamp(duration)}.")


if __name__ == "__main__":
    main()
