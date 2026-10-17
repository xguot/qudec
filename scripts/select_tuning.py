"""Freeze plain/weighted ADMM settings using independent tuning records.

  python scripts/select_tuning.py results/tune_*.jsonl --output selected.json

Selection minimizes total failures across the training seeds, separately
for each code, decoder family, and matched OSD setting. Ties choose the
smaller rho, then alpha. Confirmation records must never enter this file
set; the campaign supplies its predeclared training seeds via expected.
"""

import argparse
from collections import defaultdict
import itertools
import json
import math
import os
from pathlib import Path
import tempfile

from qudec.stats import decode_outcomes


EXPECTED_KEYS = {"codes", "seeds", "rhos", "alphas", "osd_lam", "p", "shots",
                 "rounds", "source_sha256", "protocol"}


def select(records, expected=None):
    """Return {code: {greedy/lam60: {family: {rho, alpha}}}}.

    Optional expected is a dictionary declaring codes, seeds, rhos,
    alphas, osd_lam (60), p, shots, rounds, source_sha256, and protocol.
    Supplied axes must match exactly. Without expected, observed axes
    are inferred and every code/seed/family/grid/OSD cell must be present;
    callers should supply declared axes to detect a wholly missing axis.
    All rows must share source, closed protocol, channel, and shot count.
    """
    records = list(records)
    if not records:
        raise ValueError("no tuning records")
    expected = {} if expected is None else dict(expected)
    unknown = set(expected) - EXPECTED_KEYS
    if unknown:
        raise ValueError(f"unknown expected fields: {sorted(unknown)}")
    if expected.get("osd_lam", 60) != 60:
        raise ValueError("selection requires the matched greedy/OSD-CS(60) experiment")
    common = None
    cells = {}
    codes, seeds, rhos, alphas = set(), set(), set(), set()
    settings = {}
    for record in records:
        cfg = record["decoder_config"]
        source = record["environment"]["source_sha256"]
        identity = (source, record["protocol"], record["p"], record["p_meas"],
                    record["model"], record["shots"], record["d"],
                    cfg.get("osd_order", 1))
        if common is None:
            common = identity
        elif identity != common:
            raise ValueError("tuning records mix source, protocol, channel, shots, rounds, or OSD order")
        if record["protocol"] != "closed-v1" or record["projection"] != "exact":
            raise ValueError("tuning selection requires corrected closed-v1 exact-projection records")
        if not source or cfg.get("osd_order", 1) != 1:
            raise ValueError("tuning requires a source hash and OSD-CS order 1")
        shots, seed, failures = record["shots"], record["seed"], record["failure"]
        if (not isinstance(shots, int) or shots <= 0 or not isinstance(seed, int)
                or seed < 0 or not isinstance(failures, int) or not 0 <= failures <= shots):
            raise ValueError("invalid shot, seed, or failure count")
        decode_outcomes(record)
        code = str(record["code"])
        if code not in ("72", "144"):
            raise ValueError(f"unexpected tuning code: {code}")
        lam = cfg.get("osd_lam")
        if lam not in (None, 60):
            raise ValueError("tuning must contain only greedy and OSD-CS(60)")
        mode = "greedy" if lam is None else "lam60"
        if record["decoder"] == "BpOsdDecoder":
            family, rho, alpha = "bp", None, None
        elif record["decoder"] == "AdmmOsdDecoder":
            if cfg.get("ldr", False):
                raise ValueError("LDR exploration cannot enter plain/weighted parameter selection")
            wx, wz = cfg.get("weights_x"), cfg.get("weights_z")
            if (wx is None) != (wz is None):
                raise ValueError("weighted tuning must specify both sectors")
            if wx is not None and (wx != "normalized-channel-llr" or wz != wx):
                raise ValueError("unexpected weighted-ADMM objective")
            family = "admm-w" if wx is not None else "admm"
            rho, alpha = float(cfg["rho"]), float(cfg["alpha"])
            if not math.isfinite(rho) or rho <= 0 or not math.isfinite(alpha) or not 0 < alpha < 2:
                raise ValueError("invalid tuning rho/alpha")
            rhos.add(rho)
            alphas.add(alpha)
        else:
            raise ValueError(f"unexpected tuning decoder: {record['decoder']}")
        # Only rho, alpha, and the explicitly compared OSD setting vary.
        fixed = {k: v for k, v in cfg.items()
                 if k not in ("rho", "alpha", "osd_lam", "device")}
        if family == "bp":
            fixed["alpha"] = cfg.get("alpha")
        if family in settings and settings[family] != fixed:
            raise ValueError(f"inconsistent fixed decoder settings for {family}")
        settings[family] = fixed
        key = (code, seed, mode, family, rho, alpha)
        if key in cells:
            raise ValueError(f"duplicate tuning cell: {key}")
        cells[key] = failures
        codes.add(code)
        seeds.add(seed)
    observed = {"codes": codes, "seeds": seeds, "rhos": rhos, "alphas": alphas}
    for name, values in observed.items():
        if name in expected:
            declared = {str(v) for v in expected[name]} if name == "codes" else set(expected[name])
            if declared != values:
                raise ValueError(f"tuning {name} differ from declared experiment")
    source, protocol, p, _, _, shots, rounds, _ = common
    for name, value in (("source_sha256", source), ("protocol", protocol),
                        ("p", p), ("shots", shots), ("rounds", rounds)):
        if name in expected and expected[name] != value:
            raise ValueError(f"tuning {name} differs from declared experiment")
    required = set()
    for code, seed, mode in itertools.product(codes, seeds, ("greedy", "lam60")):
        required.add((code, seed, mode, "bp", None, None))
        for family, rho, alpha in itertools.product(("admm", "admm-w"), rhos, alphas):
            required.add((code, seed, mode, family, rho, alpha))
    if not rhos or not alphas or set(cells) != required:
        missing = len(required - set(cells))
        extra = len(set(cells) - required)
        raise ValueError(f"incomplete tuning grid: {missing} missing, {extra} unexpected cells")
    result = {}
    for code in sorted(codes):
        result[code] = {}
        for mode in ("greedy", "lam60"):
            result[code][mode] = {}
            for family in ("admm", "admm-w"):
                scores = [(sum(cells[(code, seed, mode, family, rho, alpha)] for seed in seeds),
                           rho, alpha) for rho, alpha in itertools.product(rhos, alphas)]
                _, rho, alpha = min(scores)
                result[code][mode][family] = {"rho": rho, "alpha": alpha}
    return result


def write_atomic(path, value):
    """Publish complete selection JSON with a same-directory atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(value, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", type=Path, help="JSON dictionary of predeclared tuning axes")
    args = parser.parse_args(argv)
    try:
        records = [json.loads(line) for path in args.files
                   for line in path.read_text().splitlines() if line.strip()]
        expected = json.loads(args.expected.read_text()) if args.expected else None
        result = select(records, expected)
        write_atomic(args.output, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
