__all__ = ["StochRoundPco"]

import numcodecs_wasm_pco
import numcodecs_wasm_round
import numcodecs_wasm_uniform_noise
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class StochRoundPco(Compressor):
    """Stochastic Rounding + PCodec compressor.

    This compressor first applies stochastic rounding to the data, which adds noise to the data
    while rounding it. After that, it uses PCodec for further compression.
    """

    name = "stochround-pco"
    description = "Stochastic Rounding + PCodec"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        precision = error_bound
        return CodecStack(
            numcodecs_wasm_uniform_noise.UniformNoise(scale=precision, seed=42),
            numcodecs_wasm_round.Round(precision=precision),
            numcodecs_wasm_pco.Pco(
                level=8,
                mode="auto",
                delta="auto",
                paging="equal-pages-up-to",
            ),
        )
