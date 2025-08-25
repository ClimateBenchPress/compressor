__all__ = ["RPDct"]

import numcodecs_random_projection
import numcodecs_wasm_swizzle_reshape
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class RPDct(Compressor):
    name = "rp-dct"
    description = "Random Projection (DCT)"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return CodecStack(
            numcodecs_wasm_swizzle_reshape.SwizzleReshape(axes=[[0, 1, 2], [3, 4]]),
            numcodecs_random_projection.RPCodec(mae=error_bound, method="dct", seed=0),
        )
