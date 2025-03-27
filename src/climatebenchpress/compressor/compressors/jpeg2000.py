__all__ = ["Jpeg2000"]

import numcodecs.astype
import numcodecs_wasm_fixed_offset_scale
import numcodecs_wasm_jpeg2000
import numcodecs_wasm_round
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class Jpeg2000(Compressor):
    name = "jpeg2000"
    description = "JPEG 2000"

    @staticmethod
    def build() -> Codec:
        precision = 0.01
        rate = 10.0  # x10 factor compression

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
            numcodecs_wasm_jpeg2000.Jpeg2000(mode="rate", rate=rate),
        )
