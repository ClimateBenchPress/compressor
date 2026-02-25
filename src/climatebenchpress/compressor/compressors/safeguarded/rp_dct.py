__all__ = ["SafeguardedRPDct"]

import numcodecs_random_projection
import numcodecs_safeguards
import numcodecs_wasm_swizzle_reshape
from numcodecs_combinators.framed import FramedCodecStack

from ..abc import Compressor


class SafeguardedRPDct(Compressor):
    """Safeguarded RP (DCT) compressor."""

    name = "safeguarded-rp-dct"
    description = "Safeguarded(RP[DCT])"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardedCodec(
            codec=FramedCodecStack(
                numcodecs_wasm_swizzle_reshape.SwizzleReshape(axes=[[0, 1, 2], [3, 4]]),
                numcodecs_random_projection.RPCodec(
                    mae=error_bound, method="dct", seed=0, debug=True
                ),
            ),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
            ],
        )

    @staticmethod
    def rel_bound_codec(error_bound, *, data_abs_min=None, **kwargs):
        assert data_abs_min is not None, "data_abs_min must be provided"

        return numcodecs_safeguards.SafeguardedCodec(
            # conservative rel->abs error bound transformation,
            #  same as convert_rel_error_to_abs_error
            # so that we can inform the safeguards of the rel bound
            codec=FramedCodecStack(
                numcodecs_wasm_swizzle_reshape.SwizzleReshape(axes=[[0, 1, 2], [3, 4]]),
                numcodecs_random_projection.RPCodec(
                    mae=error_bound * data_abs_min,
                    method="dct",
                    seed=0,
                    debug=True,
                ),
            ),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )
