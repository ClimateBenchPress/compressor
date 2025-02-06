__all__ = ["Zfp"]

import numcodecs_wasm_swizzle_reshape
import numcodecs_wasm_zfp
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class Zfp(Compressor):
    name = "zfp"
    description = "ZFP"

    @staticmethod
    def build() -> Codec:
        return CodecStack(
            # collapse into ((realization, time,), (vertical,), (latitude,), (longitude,))
            numcodecs_wasm_swizzle_reshape.SwizzleReshape(
                axes=[[0, 1], [2], [3], [4]],
            ),  # type: ignore
            numcodecs_wasm_zfp.Zfp(mode="fixed-accuracy", tolerance=0.01),  # type: ignore
        )
