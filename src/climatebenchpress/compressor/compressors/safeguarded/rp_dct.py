__all__ = ["SafeguardedRPDct"]

import numcodecs_random_projection
import numcodecs_safeguards

from ..abc import Compressor


class SafeguardedRPDct(Compressor):
    """Safeguarded RP (DCT) compressor."""

    name = "safeguarded-rp-dct"
    description = "Safeguarded(RP[DCT])"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardedCodec(
            codec=numcodecs_random_projection.RPCodec(
                mae=error_bound,
                method="dct",
                max_block_memory=2**28,  # 256 MiB
                debug=True,
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
            codec=numcodecs_random_projection.RPCodec(
                mae=error_bound * data_abs_min,
                method="dct",
                max_block_memory=2**28,  # 256 MiB
                debug=True,
            ),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )
