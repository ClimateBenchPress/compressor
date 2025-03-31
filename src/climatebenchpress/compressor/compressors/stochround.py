__all__ = ["StochRound"]

import numcodecs_wasm_round
import numcodecs_wasm_uniform_noise
import numcodecs_wasm_zlib
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor, NamedCodec
from .utils import convert_rel_error_to_abs_error


class StochRound(Compressor):
    name = "stochround"
    description = "Stochastic Rounding"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, error_bounds
    ) -> dict[str, list[NamedCodec]]:
        codecs = {StochRound.name: []}
        bounds = list(zip([StochRound.name] * len(error_bounds), error_bounds))
        for name, eb in bounds:
            if eb.abs_error is None:
                bounds += convert_rel_error_to_abs_error(
                    name, data_abs_min, eb.rel_error
                )
                continue

            precision = eb.abs_error
            codec = CodecStack(
                numcodecs_wasm_uniform_noise.UniformNoise(scale=precision / 2, seed=42),
                numcodecs_wasm_round.Round(precision=precision),
                numcodecs_wasm_zlib.Zlib(level=6),
            )
            codecs[StochRound.name].append(NamedCodec(name=eb.name, codec=codec))

        return codecs
