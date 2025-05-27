__all__ = ["Sperr"]

import numcodecs_wasm_sperr

from .abc import Compressor


class Sperr(Compressor):
    name = "sperr"
    description = "SPERR"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_wasm_sperr.Sperr(mode="pwe", pwe=error_bound)
