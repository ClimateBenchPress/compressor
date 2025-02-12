__all__ = ["BitRoundPco"]

import numcodecs_wasm_bit_round
import numcodecs_wasm_pco
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class BitRoundPco(Compressor):
    name = "bitround-pco"
    description = "Bit Rounding + PCodec"

    @staticmethod
    def build() -> Codec:
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=9),
            numcodecs_wasm_pco.Pco(
                level=8,
                mode="auto",
                delta="auto",
                paging="equal-pages-up-to",
            ),
        )
