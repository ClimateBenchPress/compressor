__all__ = ["BitRoundPco"]


import numcodecs_wasm_bit_round
import numcodecs_wasm_pco
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor
from .utils import compute_keepbits


class BitRoundPco(Compressor):
    name = "bitround-pco"
    description = "Bit Rounding + PCodec"

    @staticmethod
    def rel_bound_codec(dtype, error_bound):
        keepbits = compute_keepbits(dtype, error_bound)
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            numcodecs_wasm_pco.Pco(
                level=8,
                mode="auto",
                delta="auto",
                paging="equal-pages-up-to",
            ),
        )
