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
    def rel_bound_codec(error_bound, *, dtype=None, data_min=None, **kwargs):
        assert dtype is not None, "dtype must be provided"
        assert data_min is not None, "data_min must be provided"

        keepbits = compute_keepbits(dtype, error_bound, data_min)
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            numcodecs_wasm_pco.Pco(
                level=8,
                mode="auto",
                delta="auto",
                paging="equal-pages-up-to",
            ),
        )
