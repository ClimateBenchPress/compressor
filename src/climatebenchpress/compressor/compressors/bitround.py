__all__ = ["BitRound"]

import math

import numcodecs_wasm_bit_round
import numcodecs_wasm_zlib
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor

SINGLE_PRECISION_MANTISSA_BITS = 23


class BitRound(Compressor):
    name = "bitround"
    description = "Bit Rounding"

    @staticmethod
    def build(data_min, data_max, abs_error=None, rel_error=None) -> Codec:
        assert (abs_error is None) != (rel_error is None), (
            "Cannot specify both abs_error and rel_error."
        )

        if rel_error is None:
            # In general, rel_error = abs_error / abs(data). This transformation
            # gives us the relative error bound that ensures the absolute error bound is
            # not exceeded for this dataset.
            rel_error = abs_error / data_max

        # - log2(rel_error) specifies the number of mantissa bits needed to satisfy
        # the rel_error bound (https://en.wikipedia.org/wiki/Machine_epsilon).
        # We need to round up to the nearest integer to ensure the error bound is not
        # exceeded.
        keepbits = -math.floor(math.log2(rel_error))
        # Ensure that keepbits is within the range of the mantissa bits of single precision.
        keepbits = max(min(keepbits, SINGLE_PRECISION_MANTISSA_BITS), 0)
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            numcodecs_wasm_zlib.Zlib(level=6),
        )
