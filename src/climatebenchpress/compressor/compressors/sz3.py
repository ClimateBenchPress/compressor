__all__ = ["Sz3"]

import numcodecs_wasm_sz3

from .abc import Compressor


class Sz3(Compressor):
    name = "sz3"
    description = "SZ3"

    @staticmethod
    def abs_bound_codec(dtype, error_bound):
        return numcodecs_wasm_sz3.Sz3(eb_mode="abs", eb_abs=error_bound)

    @staticmethod
    def rel_bound_codec(dtype, error_bound):
        return numcodecs_wasm_sz3.Sz3(eb_mode="rel", eb_rel=error_bound)
