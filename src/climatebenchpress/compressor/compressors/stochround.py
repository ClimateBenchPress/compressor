__all__ = ["StochRound"]

import numcodecs_wasm_stochastic_rounding
import numcodecs_wasm_zstd
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class StochRound(Compressor):
    """Stochastic Rounding + PCodec compressor.

    This compressor first applies stochastic rounding to the data, which adds noise to the data
    while rounding it. After that, it uses Zstandard for further compression.
    """

    name = "stochround"
    description = "Stochastic Rounding"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        precision = error_bound
        return CodecStack(
            numcodecs_wasm_stochastic_rounding.StochasticRounding(
                precision=precision, seed=42
            ),
            numcodecs_wasm_zstd.Zstd(level=3),
        )
