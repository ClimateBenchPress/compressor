__all__ = ["Jpeg2000"]

import math

import numcodecs.astype
import numcodecs_wasm_fixed_offset_scale
import numcodecs_wasm_jpeg2000
import numcodecs_wasm_round
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class Jpeg2000(Compressor):
    name = "jpeg2000"
    description = "JPEG 2000"

    @staticmethod
    def abs_bound_codec(dtype, error_bound):
        precision = 0.01
        max_pixel_val = 2**25  # maximum pixel value for our integer encoding.

        # Here we use the formula for the PSNR (https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
        # to convert between the absolute error and the PSNR value.
        # The original PSNR formula uses the root mean square error (RMSE),
        # therefore JPEG does not guaruantee pointwise error bounds but only
        # average error bounds.
        psnr = 20 * (math.log10(max_pixel_val) - math.log10(error_bound))

        return CodecStack(
            numcodecs_wasm_fixed_offset_scale.FixedOffsetScale(
                offset=0,
                scale=precision,
            ),
            numcodecs_wasm_round.Round(precision=1),
            numcodecs.astype.AsType(
                encode_dtype="int32",
                decode_dtype="float32",
            ),
            numcodecs_wasm_jpeg2000.Jpeg2000(mode="psnr", psnr=psnr),
        )
