__all__ = ["Zfp"]

import numcodecs_wasm_zfp

from .abc import Compressor


class Zfp(Compressor):
    name = "zfp"
    description = "ZFP"

    @staticmethod
    def abs_bound_codec(dtype, error_bound):
        return numcodecs_wasm_zfp.Zfp(mode="fixed-accuracy", tolerance=error_bound)
