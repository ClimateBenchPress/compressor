__all__ = ["SafeguardsSperr"]

import numcodecs_safeguards
import numcodecs_wasm_sperr

from ..abc import Compressor


class SafeguardsSperr(Compressor):
    """Safeguarded SPERR compressor."""

    name = "safeguards-sperr"
    description = "Safeguards(SPERR)"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardsCodec(
            codec=numcodecs_wasm_sperr.Sperr(mode="pwe", pwe=error_bound),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
            ],
        )
