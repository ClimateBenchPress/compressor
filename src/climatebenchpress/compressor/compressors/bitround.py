__all__ = ["BitRound"]

import numcodecs_wasm_bit_round
import numcodecs_wasm_zlib
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor
from .utils import compute_keepbits


class BitRound(Compressor):
    name = "bitround"
    description = "Bit Rounding"

    @staticmethod
    def rel_bound_codec(error_bound, *, dtype=None, **kwargs):
        assert dtype is not None, "dtype must be provided"

        keepbits = compute_keepbits(dtype, error_bound)
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            numcodecs_wasm_zlib.Zlib(level=6),
        )
