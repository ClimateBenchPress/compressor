__all__ = ["FFmpeg"]

import numpy as np
import numcodecs.astype
import numcodecs.compat
import numcodecs_wasm_fixed_offset_scale
import numcodecs_wasm_round
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class FFmpeg(Compressor):
    name = "ffmpeg"
    description = "FFmpeg"

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

        max_pixel_val = 2**12 - 1  # maximum pixel value for our integer encoding.

        data_range = data_max - data_min

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
            # apply the ffmpeg codec
            FFmpegCodec(),
        )


class FFmpegCodec(Codec):
    codec_id = "ffmpeg"

    def __init__(self, *args, **kwargs):
        # handle config
        pass

    def encode(self, buf) -> bytes:
        a = numcodecs.compat.ensure_ndarray(buf)

        # use ffmpeg to encode the array as a video

        return b"encoded"

    def decode(self, buf, out=None):
        b = numcodecs.compat.ensure_bytes(buf)

        # use ffmpeg to decode the data
        decoded = np.array()

        return numcodecs.compat.ndarray_copy(decoded, out)
