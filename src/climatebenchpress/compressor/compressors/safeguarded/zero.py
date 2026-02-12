__all__ = ["SafeguardedZero"]

import numcodecs_safeguards
import numcodecs_zero

from ..abc import Compressor


class SafeguardedZero(Compressor):
    """Safeguarded all-zero compressor."""

    name = "safeguarded-zero"
    description = "Safeguarded(0)"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardedCodec(
            codec=numcodecs_zero.ZeroCodec(),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
            ],
        )

    @staticmethod
    def rel_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardedCodec(
            codec=numcodecs_zero.ZeroCodec(),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )
