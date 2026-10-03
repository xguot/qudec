"""qudec: GPU-accelerated quantum error correction decoders."""

from qudec.bench import benchmark_iid
from qudec.bposd import BpOsdDecoder
from qudec.codes import (
    bivariate_bicycle,
    code_info,
    gf2_rank,
    gross_code,
    logicals,
    medium_code,
    repetition_code,
    steane_code,
)

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
