__all__ = ["StochRoundPco"]

import numcodecs_wasm_pco
import numcodecs_wasm_stochastic_rounding
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class StochRoundPco(Compressor):
    name = "stochround-pco"
    description = "Stochastic Rounding + PCodec"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        precision = error_bound
        return CodecStack(
            numcodecs_wasm_stochastic_rounding.StochasticRounding(
                precision=precision, seed=42
            ),
            numcodecs_wasm_pco.Pco(
                level=8,
                mode="auto",
                delta="auto",
                paging="equal-pages-up-to",
            ),
        )
