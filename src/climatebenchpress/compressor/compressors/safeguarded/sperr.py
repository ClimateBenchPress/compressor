__all__ = ["SafeguardedSperr"]

import numcodecs
import numcodecs.abc
import numcodecs.compat
import numcodecs_safeguards
import numcodecs_wasm_sperr
import numpy as np
from numcodecs_combinators.stack import CodecStack

from ..abc import Compressor


class SafeguardedSperr(Compressor):
    """Safeguarded SPERR compressor."""

    name = "safeguarded-sperr"
    description = "Safeguarded(SPERR)"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardedCodec(
            codec=CodecStack(
                NaNToMean(),
                numcodecs_wasm_sperr.Sperr(mode="pwe", pwe=error_bound),
            ),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
            ],
        )

    @staticmethod
    def rel_bound_codec(error_bound, *, data_abs_min=None, **kwargs):
        assert data_abs_min is not None, "data_abs_min must be provided"

        return numcodecs_safeguards.SafeguardedCodec(
            codec=CodecStack(
                NaNToMean(),
                # conservative rel->abs error bound transformation,
                #  same as convert_rel_error_to_abs_error
                # so that we can inform the safeguards of the rel bound
                numcodecs_wasm_sperr.Sperr(mode="pwe", pwe=error_bound * data_abs_min),
            ),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )


# inspired by H5Z-SPERR's treatment of NaN values:
# https://github.com/NCAR/H5Z-SPERR/blob/72ebcb00e382886c229c5ef5a7e237fe451d5fb8/src/h5z-sperr.c#L464-L473
# https://github.com/NCAR/H5Z-SPERR/blob/72ebcb00e382886c229c5ef5a7e237fe451d5fb8/src/h5zsperr_helper.cpp#L179-L212
class NaNToMean(numcodecs.abc.Codec):
    codec_id = "nan-to-mean"  # type: ignore

    def encode(self, buf):
        return np.nan_to_num(buf, nan=np.nanmean(buf), posinf=np.inf, neginf=-np.inf)

    def decode(self, buf, out=None):
        return numcodecs.compat.ndarray_copy(buf, out)
