"""Bounded, paired phenomenological memory benchmarks.

Examples (options may appear before or after p values):
  python scripts/bench_phenom.py 72 6 2500 .01 .02 --seed 4 --batch-size 32
  python scripts/bench_phenom.py 72 6 300 --legacy-open-boundary .02

JSONL records preserve exact counts, per-shot outcomes, sample hashes,
protocol and decoder settings. New closed-v1 results use correct BP
measurement priors and exact parity projection; do not pool old tables.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy
import torch

from qudec.admm import AdmmOsdDecoder
from qudec.bposd import BpOsdDecoder
from qudec.codes import gross_code, logicals, medium_code
from qudec.phenom import PhenomDecoder, benchmark_phenom, phenom_priors, sample_phenom


VARIANTS = ("bp", "admm", "admm-w", "admm-ldr")


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def error_rate(value):
    result = float(value)
    if not 0 < result < 0.5:
        raise argparse.ArgumentTypeError("must be between 0 and 0.5")
    return result


def common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("code", choices=("72", "144"))
    parser.add_argument("rounds", type=positive_int)
    parser.add_argument("shots", type=positive_int, nargs="?", default=300)
    parser.add_argument("ps", type=error_rate, nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--p-meas", type=error_rate)
    parser.add_argument("--legacy-open-boundary", action="store_true",
                        help="historical logical-only open-boundary metric")
    parser.add_argument("--output", type=Path,
                        help="append JSONL records (default results/<kind>_<code>.jsonl)")
    return parser


def parse_args(argv=None):
    parser = common_parser(__doc__)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--osd-lam", type=positive_int)
    parser.add_argument("--admm-rho", type=float, default=2.0)
    parser.add_argument("--admm-alpha", type=float, default=1.0)
    args = parser.parse_intermixed_args(argv)
    args.variants = args.variants.split(",")
    if not args.variants or any(v not in VARIANTS for v in args.variants):
        parser.error("--variants must be a comma-separated selection of " + ",".join(VARIANTS))
    if len(args.variants) != len(set(args.variants)):
        parser.error("--variants must not contain duplicates")
    if (args.seed < 0 or not np.isfinite(args.admm_rho) or args.admm_rho <= 0
            or not 0 < args.admm_alpha < 2):
        parser.error("require seed >= 0, rho > 0, and 0 < alpha < 2")
    args.ps = list(dict.fromkeys(args.ps or [0.005, 0.01, 0.02, 0.03]))
    return args


def environment_metadata():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*root.glob("src/qudec/*.py"), *root.glob("scripts/*.py"),
                        root / "pyproject.toml"]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"source_sha256": digest.hexdigest(), "git_revision": revision,
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "available_accelerator": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None}


def run_variant(args, matrices, p, samples, name, cls, kw, output, environment):
    """Run one variant and append a self-contained paired outcome record."""
    h_x, h_z, l_x, l_z = matrices
    p_x, p_z = phenom_priors(h_x, h_z, args.rounds, p, args.p_meas)
    dec = PhenomDecoder(cls, *matrices, args.rounds,
                        terminal_perfect=not args.legacy_open_boundary,
                        p_x=p_x, p_z=p_z, **kw)
    # Include effective defaults as well as overrides (notably LDR beta).
    config = {key: getattr(dec.base, key) for key in
              ("max_iter", "osd_order", "osd_lam", "alpha", "rho", "max_r",
               "ldr", "ldr_outer", "ldr_beta") if hasattr(dec.base, key)}
    config["device"] = "cpu" if cls is BpOsdDecoder else (
        "cuda" if environment["cuda_available"] else "cpu")
    t0 = time.perf_counter()
    res = benchmark_phenom(dec, *matrices, p, args.rounds, args.shots,
                           seed=args.seed, batch_size=args.batch_size,
                           p_meas=args.p_meas, samples=samples, return_failures=True)
    # Store compact model settings rather than long repeated weight vectors.
    for key in ("weights_x", "weights_z"):
        if key in kw:
            config[key] = "normalized-channel-llr"
    res.update(schema_version=1, code=args.code, variant=name,
               decoder=cls.__name__, decoder_config=config,
               bp_priors="data=2p/3,measurement=p_meas", projection="exact",
               solver_device=config["device"], osd_device="cpu",
               solver_device_name="cpu" if config["device"] == "cpu"
               else environment["available_accelerator"],
               secs=time.perf_counter()-t0, environment=environment,
               created_utc=datetime.now(timezone.utc).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as fh:
        fh.write(json.dumps(res, sort_keys=True) + "\n")
    print(f"[{args.code}] {name} p={p:.3f} ler={res['ler']:.5f} "
          f"failures={res['failure']}/{args.shots} "
          f"invalid={res['invalid_x']}+{res['invalid_z']} "
          f"residual={res['residual_x']}+{res['residual_z']} "
          f"seed={args.seed} protocol={res['protocol']} {res['secs']:.1f}s", flush=True)
    return res


def main(argv=None):
    args = parse_args(argv)
    h_x, h_z = medium_code() if args.code == "72" else gross_code()
    matrices = h_x, h_z, *logicals(h_x, h_z)
    output = args.output or Path("results") / f"phenom_{args.code}.jsonl"
    environment = environment_metadata()
    for p in args.ps:
        samples = sample_phenom(h_x, h_z, p, args.rounds, args.shots,
                                seed=args.seed, p_meas=args.p_meas,
                                terminal_perfect=not args.legacy_open_boundary)
        p_x, p_z = phenom_priors(h_x, h_z, args.rounds, p, args.p_meas)
        for name in args.variants:
            kw = {"max_iter": 30 if name == "bp" else 150,
                  "osd_order": 1, "osd_lam": args.osd_lam}
            cls = BpOsdDecoder if name == "bp" else AdmmOsdDecoder
            if name == "bp":
                kw["alpha"] = 0.9
            else:
                kw.update(rho=args.admm_rho, alpha=args.admm_alpha)
            if name == "admm-w":
                kw.update(weights_x=np.log((1-p_x)/p_x), weights_z=np.log((1-p_z)/p_z))
            if name == "admm-ldr":
                kw.update(ldr=True, ldr_outer=4)
            run_variant(args, matrices, p, samples, name, cls, kw, output, environment)


if __name__ == "__main__":
    main()
