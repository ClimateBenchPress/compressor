__all__ = ["BitRound"]

import numcodecs_wasm_bit_round
import numcodecs_wasm_zlib
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class BitRound(Compressor):
    name = "bitround"
    description = "Bit Rounding"

    @staticmethod
    def build() -> Codec:
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=9),
            numcodecs_wasm_zlib.Zlib(level=6),
        )
