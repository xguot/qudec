"""qudec: GPU-accelerated quantum error correction decoders.

Imports are lazy so the numpy-only core (codes, noise, phenom sampling,
OSD post-processing, circuit dem extraction) imports and tests without
torch or stim installed.
"""

__all__ = [
    "BpOsdDecoder",
    "benchmark_iid",
    "bivariate_bicycle",
    "code_info",
    "gf2_rank",
    "gross_code",
    "logicals",
    "medium_code",
    "repetition_code",
    "steane_code",
]


def __getattr__(name):
    if name == "BpOsdDecoder":
        from qudec.bposd import BpOsdDecoder
        return BpOsdDecoder
    if name == "benchmark_iid":
        from qudec.bench import benchmark_iid
        return benchmark_iid
    if name in ("bivariate_bicycle", "code_info", "gf2_rank", "gross_code",
                "logicals", "medium_code", "repetition_code",
                "steane_code"):
        import qudec.codes as _codes
        return getattr(_codes, name)
    raise AttributeError(f"module 'qudec' has no attribute {name!r}")
