__all__ = ["Zfp"]

import numcodecs_wasm_zfp
from numcodecs.abc import Codec

from .abc import Compressor


class Zfp(Compressor):
    name = "zfp"
    description = "ZFP"

    @staticmethod
    def build() -> Codec:
        return numcodecs_wasm_zfp.Zfp(mode="fixed-accuracy", tolerance=0.01)
