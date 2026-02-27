__all__ = ["SafeguardedRP"]

import numcodecs_random_projection
import numcodecs_safeguards
import numcodecs_wasm_swizzle_reshape
from numcodecs_combinators.framed import FramedCodecStack

from ..abc import Compressor


class SafeguardedRP(Compressor):
    """Safeguarded RP compressor."""

    name = "safeguarded-rp"
    description = "Safeguarded(RP[Gaussian])"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardedCodec(
            codec=FramedCodecStack(
                numcodecs_wasm_swizzle_reshape.SwizzleReshape(axes=[[0, 1, 2], [3, 4]]),
                numcodecs_random_projection.RPCodec(
                    mae=error_bound,
                    method="gaussian",
                    seed=42,
                    max_block_memory=2**28,  # 256 MiB
                    debug=True,
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
                    method="gaussian",
                    seed=42,
                    max_block_memory=2**28,  # 256 MiB
                    debug=True,
                ),
            ),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )
