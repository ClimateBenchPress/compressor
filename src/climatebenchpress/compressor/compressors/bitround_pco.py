__all__ = ["BitRoundPco"]

import numcodecs_wasm_bit_round
import numcodecs_wasm_pco
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor
from .bitround import compute_keepbits


class BitRoundPco(Compressor):
    name = "bitround-pco"
    description = "Bit Rounding + PCodec"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, abs_error=None, rel_error=None
    ) -> Codec:
        assert (abs_error is None) != (rel_error is None), (
            "Cannot specify both abs_error and rel_error."
        )

        keepbits = compute_keepbits(dtype, data_abs_max, abs_error, rel_error)
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            numcodecs_wasm_pco.Pco(
                level=8,
                mode="auto",
                delta="auto",
                paging="equal-pages-up-to",
            ),
        )
