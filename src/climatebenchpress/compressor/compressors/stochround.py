__all__ = ["StochRound"]

import numcodecs_wasm_round
import numcodecs_wasm_uniform_noise
import numcodecs_wasm_zstd
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class StochRound(Compressor):
    name = "stochround"
    description = "Stochastic Rounding"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        precision = error_bound
        return CodecStack(
            numcodecs_wasm_uniform_noise.UniformNoise(scale=precision, seed=42),
            numcodecs_wasm_round.Round(precision=precision),
            numcodecs_wasm_zstd.Zstd(level=3),
        )
