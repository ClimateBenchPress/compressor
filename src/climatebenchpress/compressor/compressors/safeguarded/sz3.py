__all__ = ["SafeguardedSz3"]

import numcodecs_safeguards
import numcodecs_wasm_sz3

from ..abc import Compressor


class SafeguardedSz3(Compressor):
    """Safeguarded SZ3 compressor."""

    name = "safeguarded-sz3"
    description = "Safeguarded(SZ3)"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardedCodec(
            codec=numcodecs_wasm_sz3.Sz3(eb_mode="abs", eb_abs=error_bound),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
            ],
        )

    @staticmethod
    def rel_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardedCodec(
            codec=numcodecs_wasm_sz3.Sz3(eb_mode="rel", eb_rel=error_bound),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )
