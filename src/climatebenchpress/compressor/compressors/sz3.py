__all__ = ["Sz3"]

import numcodecs_wasm_sz3

from .abc import Compressor


class Sz3(Compressor):
    name = "sz3"
    description = "SZ3"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_wasm_sz3.Sz3(eb_mode="abs", eb_abs=error_bound)

    @staticmethod
    def rel_bound_codec(error_bound, **kwargs):
        # SZ3 will not ensure that the relative error bound is strictly met.
        # Internally, SZ3 transforms the relative error bound to an absolute error bound
        # based on the range of the input data:
        # https://github.com/szcompressor/SZ3/blob/e8a6b1569067abdd6b7d4276e91eced115be4f14/include/SZ3/utils/Statistic.hpp#L36
        return numcodecs_wasm_sz3.Sz3(eb_mode="rel", eb_rel=error_bound)
