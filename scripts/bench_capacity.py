"""Bounded paired code-capacity sweeps with reproducible JSONL outcomes.

  python scripts/bench_capacity.py --code 144 --shots 500 --seed 0 \
      --ps .03 .05 .08 .1 --variants bp,lp,admm --output results/capacity.jsonl

One isolated worker may select a single variant. Fixed code/rate/seed/shot
settings reproduce the same sampled errors regardless of variant or decode
batch size. The optional reference variant requires the reference extra.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from .bench_phenom import environment_metadata
else:
    from bench_phenom import environment_metadata

from qudec.admm import AdmmOsdDecoder
from qudec.bposd import BpOsdDecoder
from qudec.codes import gross_code, logicals, medium_code
from qudec.lp import LpOsdDecoder
from qudec.noise import sample_iid_errors


VARIANTS = ("bp", "lp", "admm", "reference")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def error_rate(value):
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 0.5:
        raise argparse.ArgumentTypeError("must satisfy 0 < p < 0.5")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", choices=("72", "144"), default="144")
    parser.add_argument("--shots", type=positive_int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--ps", type=error_rate, nargs="+",
                        default=[0.03, 0.05, 0.08, 0.1])
    parser.add_argument("--variants", default="bp,lp,admm")
    parser.add_argument("--osd-lam", type=positive_int,
                        help="in-tree exhaustive OSD sweep; reference keeps its own defaults")
    parser.add_argument("--bp-max-iter", type=positive_int, default=30)
    parser.add_argument("--bp-alpha", type=float, default=0.9)
    parser.add_argument("--admm-max-iter", type=positive_int, default=500)
    parser.add_argument("--admm-rho", type=float, default=2.0)
    parser.add_argument("--admm-alpha", type=float, default=1.0)
    parser.add_argument("--reference-max-iter", type=positive_int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.variants = args.variants.split(",")
    if any(name not in VARIANTS for name in args.variants):
        parser.error("--variants must select from " + ",".join(VARIANTS))
    if len(args.variants) != len(set(args.variants)):
        parser.error("--variants must not contain duplicates")
    if len(args.ps) != len(set(args.ps)):
        parser.error("--ps must not contain duplicate rates")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not np.isfinite(args.bp_alpha) or args.bp_alpha <= 0:
        parser.error("--bp-alpha must be finite and positive")
    if not np.isfinite(args.admm_rho) or args.admm_rho <= 0:
        parser.error("--admm-rho must be finite and positive")
    if not np.isfinite(args.admm_alpha) or not 0 < args.admm_alpha < 2:
        parser.error("--admm-alpha must satisfy 0 < alpha < 2")
    return args


def sample_capacity(h_x, h_z, p, shots, seed):
    """Sample the whole experiment before any decoding or batching."""
    e_x, e_z = sample_iid_errors(h_x.shape[1], p, shots, "depolarizing", seed)
    sx = ((h_z @ e_x.T) % 2).T.astype(np.int8)
    sz = ((h_x @ e_z.T) % 2).T.astype(np.int8)
    return sx, sz, e_x, e_z


def benchmark_capacity(decoder, matrices, samples, batch_size):
    """Decode shared samples and count invalid corrections as failures."""
    h_x, h_z, l_x, l_z = matrices
    sx, sz, e_x, e_z = samples
    shots, n = len(e_x), h_x.shape[1]
    if shots < 1 or batch_size < 1:
        raise ValueError("shots and batch_size must be positive")
    expected = ((shots, h_z.shape[0]), (shots, h_x.shape[0]),
                (shots, n), (shots, n))
    if any(a.shape != shape for a, shape in zip(samples, expected)):
        raise ValueError("sample shapes do not match code and shot count")
    flags = {key: np.zeros(shots, dtype=bool) for key in
             ("logical_x", "logical_z", "invalid_x", "invalid_z",
              "residual_x", "residual_z", "failure_x", "failure_z", "failure")}
    digest = hashlib.sha256()
    for a in samples:
        digest.update(str(a.shape).encode())
        digest.update(np.asarray(a, np.int8).tobytes())
    for start in range(0, shots, batch_size):
        sl = slice(start, min(start + batch_size, shots))
        c_x, c_z = (np.asarray(a) for a in
                    decoder.decode_corrections(sx[sl], sz[sl]))
        if c_x.shape != e_x[sl].shape or c_z.shape != e_z[sl].shape:
            raise ValueError("decoder returned the wrong correction shape")
        binary_x = np.isin(c_x, [0, 1]).all(axis=1)
        binary_z = np.isin(c_z, [0, 1]).all(axis=1)
        if not np.isfinite(c_x).all() or not np.isfinite(c_z).all():
            raise ValueError("decoder returned nonfinite corrections")
        c_x, c_z = c_x.astype(np.int8), c_z.astype(np.int8)
        rx, rz = e_x[sl] ^ c_x, e_z[sl] ^ c_z
        flags["logical_x"][sl] = np.any((l_z @ rx.T) % 2, axis=0)
        flags["logical_z"][sl] = np.any((l_x @ rz.T) % 2, axis=0)
        flags["residual_x"][sl] = np.any((h_z @ rx.T) % 2, axis=0)
        flags["residual_z"][sl] = np.any((h_x @ rz.T) % 2, axis=0)
        flags["invalid_x"][sl] = (~binary_x | np.any(
            ((h_z @ c_x.T) % 2).T != sx[sl], axis=1))
        flags["invalid_z"][sl] = (~binary_z | np.any(
            ((h_x @ c_z.T) % 2).T != sz[sl], axis=1))
    for sector in ("x", "z"):
        flags[f"failure_{sector}"] = (flags[f"logical_{sector}"]
                                        | flags[f"residual_{sector}"]
                                        | flags[f"invalid_{sector}"])
    flags["failure"] = flags["failure_x"] | flags["failure_z"]
    counts = {key: int(value.sum()) for key, value in flags.items()}
    return dict(counts, shots=shots, ler=counts["failure"]/shots,
                ler_x=counts["failure_x"]/shots, ler_z=counts["failure_z"]/shots,
                sample_id=digest.hexdigest(), batch_size=batch_size,
                outcomes={key: "".join(value.astype(np.uint8).astype(str))
                          for key, value in flags.items()})


def load_reference_decoder():
    """Keep optional reference dependencies out of the normal sweep."""
    try:
        from bposd.css import css_code  # noqa: F401: preflight before any runs
        if __package__:
            from .bench_gross_ref import BposdRefDecoder
        else:
            from bench_gross_ref import BposdRefDecoder
    except ImportError as exc:
        raise SystemExit("reference variant requires: pip install -e '.[reference]'") from exc
    return BposdRefDecoder


def prime_reference(decoder, h_x, h_z):
    """Calibrate the adapter independently of the first decoding batch.

    Its legacy calibration must see nonzero X and Z syndromes. A fixed
    synthetic probe avoids selecting a layout from an empty/noiseless batch.
    These probes are excluded from benchmark counts and sampling identity.
    """
    rng = np.random.default_rng(341)
    n = h_x.shape[1]
    sx = ((h_z @ rng.integers(0, 2, (64, n), dtype=np.int8).T) % 2).T
    sz = ((h_x @ rng.integers(0, 2, (64, n), dtype=np.int8).T) % 2).T
    good = sx.any(axis=1) & sz.any(axis=1)
    if good.sum() < 8:
        raise ValueError("reference calibration requires nontrivial X and Z checks")
    sx, sz = sx[good][:8], sz[good][:8]
    c_x, c_z = decoder.decode_corrections(sx, sz)
    if (not np.array_equal((h_z @ c_x.T).T % 2, sx)
            or not np.array_equal((h_x @ c_z.T).T % 2, sz)):
        raise ValueError("reference calibration returned invalid corrections")


def build_decoder(args, matrices, p, name, reference_cls=None):
    common = dict(p_x=2*p/3, p_z=2*p/3, osd_order=1, osd_lam=args.osd_lam)
    if name == "bp":
        dec = BpOsdDecoder(*matrices, max_iter=args.bp_max_iter,
                           alpha=args.bp_alpha, **common)
    elif name == "lp":
        dec = LpOsdDecoder(*matrices, **common)
    elif name == "admm":
        dec = AdmmOsdDecoder(*matrices, max_iter=args.admm_max_iter,
                             rho=args.admm_rho, alpha=args.admm_alpha, **common)
    else:
        dec = reference_cls(*matrices, p, max_iter=args.reference_max_iter)
        prime_reference(dec, matrices[0], matrices[1])
    config = {key: getattr(dec, key) for key in
              ("max_iter", "osd_order", "osd_lam", "alpha", "rho", "max_r",
               "ldr", "ldr_outer", "ldr_beta") if hasattr(dec, key)}
    if name == "reference":
        config.update(max_iter=args.reference_max_iter, osd_lam=None,
                      calibration="fixed-seed-nonzero-syndromes",
                      wrapper="scripts/bench_gross_ref.py")
        backend = getattr(dec, "_decoder", None)
        if backend is not None:
            config["backend"] = f"{type(backend).__module__}.{type(backend).__name__}"
            for key in ("max_iter", "bp_method", "ms_scaling_factor", "schedule",
                        "osd_method", "osd_order"):
                value = getattr(backend, key, None)
                if isinstance(value, (str, int, float, bool)):
                    config[key] = value
    return dec, config


def main(argv=None):
    args = parse_args(argv)
    reference_cls = load_reference_decoder() if "reference" in args.variants else None
    h_x, h_z = medium_code() if args.code == "72" else gross_code()
    matrices = h_x, h_z, *logicals(h_x, h_z)
    environment = environment_metadata()
    if reference_cls is not None:
        for package in ("bposd", "ldpc"):
            try:
                environment[package] = version(package)
            except PackageNotFoundError:
                environment[package] = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for p in args.ps:
        samples = sample_capacity(h_x, h_z, p, args.shots, args.seed)
        for name in args.variants:
            t0 = time.perf_counter()
            decoder, config = build_decoder(args, matrices, p, name, reference_cls)
            setup_secs = time.perf_counter() - t0
            device = "cuda" if name == "admm" and environment["cuda_available"] else "cpu"
            config["device"] = device
            t0 = time.perf_counter()
            result = benchmark_capacity(decoder, matrices, samples, args.batch_size)
            result.update(schema_version=1, code=args.code, variant=name,
                          d=0, p=p, p_meas=0, model="depolarizing",
                          seed=args.seed, protocol="code-capacity-v1", projection="exact",
                          decoder=type(decoder).__name__, decoder_config=config,
                          bp_priors="data=2p/3", solver_device=device, osd_device="cpu",
                          solver_device_name="cpu" if device == "cpu"
                          else environment["available_accelerator"],
                          secs=time.perf_counter()-t0, setup_secs=setup_secs,
                          environment=environment,
                          created_utc=datetime.now(timezone.utc).isoformat())
            with args.output.open("a") as fh:
                fh.write(json.dumps(result, sort_keys=True) + "\n")
            print(f"[capacity {args.code}] {name} p={p:.3f} "
                  f"failures={result['failure']}/{args.shots} ler={result['ler']:.5f} "
                  f"invalid={result['invalid_x']}+{result['invalid_z']} "
                  f"seed={args.seed} {result['secs']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
