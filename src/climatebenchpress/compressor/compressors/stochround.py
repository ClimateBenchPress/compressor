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
    def build(data_min, data_max, abs_error=None, rel_error=None) -> Codec:
        assert (abs_error is None) != (rel_error is None), (
            "Cannot specify both abs_error and rel_error."
        )
        if abs_error is None:
            # In general, rel_error = abs_error / abs(data). This transformation
            # gives us the absolute error bound that ensures the relative error bound is
            # not exceeded for this dataset.
            abs_error = rel_error * data_min
        precision = abs_error

        return CodecStack(
            numcodecs_wasm_uniform_noise.UniformNoise(scale=precision / 2, seed=42),
            numcodecs_wasm_round.Round(precision=precision),
            numcodecs_wasm_zlib.Zlib(level=6),
        )
