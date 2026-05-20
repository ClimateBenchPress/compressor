__all__ = ["Ebcc"]

import numcodecs.astype
import numcodecs_wasm_ebcc
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class Ebcc(Compressor):
    """EBCC compressor."""

    name = "ebcc"
    description = "EBCC"

    @staticmethod
    def abs_bound_codec(error_bound, dtype=None, **kwargs):
        assert dtype is not None, "dtype must be provided"

        return CodecStack(
            # EBCC only supports float32 data
            numcodecs.astype.AsType(
                encode_dtype="float32",
                decode_dtype=dtype.name,
            ),
            numcodecs_wasm_ebcc.Ebcc(
                # reasonable default recommended by Langwen Huang
                base_cr=100,
                residual="absolute",
                error=error_bound,
            ),
        )

    @staticmethod
    def rel_bound_codec(error_bound, dtype=None, **kwargs):
        assert dtype is not None, "dtype must be provided"

        return CodecStack(
            # EBCC only supports float32 data
            numcodecs.astype.AsType(
                encode_dtype="float32",
                decode_dtype=dtype.name,
            ),
            numcodecs_wasm_ebcc.Ebcc(
                # reasonable default recommended by Langwen Huang
                base_cr=100,
                residual="relative",
                error=error_bound,
            ),
        )
