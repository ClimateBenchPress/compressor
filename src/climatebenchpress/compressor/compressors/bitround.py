__all__ = ["BitRound"]

import numcodecs_wasm_bit_round
import numcodecs_wasm_zstd
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor
from .utils import compute_keepbits


class BitRound(Compressor):
    name = "bitround"
    description = "Bit Rounding"

    @staticmethod
    def rel_bound_codec(dtype, error_bound):
        keepbits = compute_keepbits(dtype, error_bound)
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            numcodecs_wasm_zstd.Zstd(level=3),
        )
