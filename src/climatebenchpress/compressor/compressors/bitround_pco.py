__all__ = ["BitRoundPco"]

import math

import numcodecs_wasm_bit_round
import numcodecs_wasm_pco
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor

SINGLE_PRECISION_MANTISSA_BITS = 23


class BitRoundPco(Compressor):
    name = "bitround-pco"
    description = "Bit Rounding + PCodec"

    @staticmethod
    def build(data_min, data_max, abs_error=None, rel_error=None) -> Codec:
        assert (abs_error is None) != (rel_error is None), (
            "Cannot specify both abs_error and rel_error."
        )

        if rel_error is None:
            rel_error = abs_error / data_max

        keepbits = -math.floor(math.log2(rel_error))
        keepbits = max(min(keepbits, SINGLE_PRECISION_MANTISSA_BITS), 0)
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            numcodecs_wasm_pco.Pco(
                level=8,
                mode="auto",
                delta="auto",
                paging="equal-pages-up-to",
            ),
        )
