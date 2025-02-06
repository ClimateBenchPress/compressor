__all__ = ["StochRound"]

import numcodecs_wasm_round
import numcodecs_wasm_uniform_noise
import numcodecs_wasm_zlib
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class StochRound(Compressor):
    name = "stochround"
    description = "Stochastic Rounding"

    @staticmethod
    def build() -> Codec:
        precision = 0.01

        return CodecStack(
            numcodecs_wasm_uniform_noise.UniformNoise(scale=precision / 2, seed=42),  # type: ignore
            numcodecs_wasm_round.Round(precision=precision),  # type: ignore
            numcodecs_wasm_zlib.Zlib(level=6),  # type: ignore
        )
