__all__ = ["StochRoundPco"]

import numcodecs_wasm_pco
import numcodecs_wasm_round
import numcodecs_wasm_uniform_noise
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class StochRoundPco(Compressor):
    name = "stochround-pco"
    description = "Stochastic Rounding + PCodec"

    @staticmethod
    def abs_bound_codec(dtype, error_bound):
        precision = error_bound
        return CodecStack(
            numcodecs_wasm_uniform_noise.UniformNoise(scale=precision / 2, seed=42),
            numcodecs_wasm_round.Round(precision=precision),
            numcodecs_wasm_pco.Pco(
                level=8,
                mode="auto",
                delta="auto",
                paging="equal-pages-up-to",
            ),
        )
