__all__ = ["SafeguardedEbcc"]

import numcodecs.astype
import numcodecs_replace
import numcodecs_safeguards
import numcodecs_wasm_ebcc
import numpy as np
from numcodecs_combinators.stack import CodecStack

from ..abc import Compressor


class SafeguardedEbcc(Compressor):
    """Safeguarded EBCC compressor."""

    name = "safeguarded-ebcc"
    description = "Safeguarded(EBCC)"

    @staticmethod
    def abs_bound_codec(error_bound, dtype=None, **kwargs):
        assert dtype is not None, "dtype must be provided"

        return numcodecs_safeguards.SafeguardedCodec(
            codec=CodecStack(
                # EBCC only supports float32 data
                numcodecs.astype.AsType(
                    encode_dtype="float32",
                    decode_dtype=dtype.name,
                ),
                # inspired by H5Z-SPERR's treatment of NaN values:
                # https://github.com/NCAR/H5Z-SPERR/blob/72ebcb00e382886c229c5ef5a7e237fe451d5fb8/src/h5z-sperr.c#L464-L473
                # https://github.com/NCAR/H5Z-SPERR/blob/72ebcb00e382886c229c5ef5a7e237fe451d5fb8/src/h5zsperr_helper.cpp#L179-L212
                numcodecs_replace.ReplaceFilterCodec(replacements={np.nan: "nan_mean"}),
                numcodecs_wasm_ebcc.Ebcc(
                    # reasonable default recommended by Langwen Huang
                    base_cr=100,
                    residual="absolute",
                    error=error_bound,
                    chunk_shape="auto",
                ),
            ),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
            ],
        )

    @staticmethod
    def rel_bound_codec(error_bound, dtype=None, **kwargs):
        assert dtype is not None, "dtype must be provided"

        return numcodecs_safeguards.SafeguardedCodec(
            codec=CodecStack(
                # EBCC only supports float32 data
                numcodecs.astype.AsType(
                    encode_dtype="float32",
                    decode_dtype=dtype.name,
                ),
                # inspired by H5Z-SPERR's treatment of NaN values:
                # https://github.com/NCAR/H5Z-SPERR/blob/72ebcb00e382886c229c5ef5a7e237fe451d5fb8/src/h5z-sperr.c#L464-L473
                # https://github.com/NCAR/H5Z-SPERR/blob/72ebcb00e382886c229c5ef5a7e237fe451d5fb8/src/h5zsperr_helper.cpp#L179-L212
                numcodecs_replace.ReplaceFilterCodec(replacements={np.nan: "nan_mean"}),
                numcodecs_wasm_ebcc.Ebcc(
                    # reasonable default recommended by Langwen Huang
                    base_cr=100,
                    residual="relative",
                    error=error_bound,
                    chunk_shape="auto",
                ),
            ),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )
