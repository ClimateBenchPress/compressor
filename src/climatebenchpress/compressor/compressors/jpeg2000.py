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
    def abs_bound_codec(
        error_bound,
        *,
        data_min=None,
        data_max=None,
        **kwargs,
    ):
        assert data_min is not None, "data_min must be provided"
        assert data_max is not None, "data_max must be provided"

        max_pixel_val = 2**25 - 1  # maximum pixel value for our integer encoding.

        data_range = data_max - data_min

        # Here we use the formula for the PSNR (https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
        # to convert between the absolute error and the PSNR value.
        # The original PSNR formula uses the root mean square error (RMSE),
        # therefore JPEG does not guaruantee pointwise error bounds but only
        # average error bounds.
        psnr = 20 * (math.log10(data_range) - math.log10(error_bound))

        return CodecStack(
            # increase precision for better rounding during linear quantization
            numcodecs.astype.AsType(
                encode_dtype="float64",
                decode_dtype="float32",
            ),
            # remap from [min, max] to [0, max_pixel_val]
            numcodecs_wasm_fixed_offset_scale.FixedOffsetScale(
                offset=data_min,
                scale=data_range / max_pixel_val,
            ),
            # round and truncate to integer values
            numcodecs_wasm_round.Round(precision=1),
            numcodecs.astype.AsType(
                encode_dtype="uint32",
                decode_dtype="float64",
            ),
            # apply the PSNR error bound
            numcodecs_wasm_jpeg2000.Jpeg2000(mode="psnr", psnr=psnr),
        )
