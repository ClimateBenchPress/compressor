__all__ = ["Sz3"]

import numcodecs_wasm_sz3
from numcodecs.abc import Codec

from .abc import Compressor


class Sz3(Compressor):
    name = "sz3"
    description = "SZ3"

    @staticmethod
    def build() -> Codec:
        return numcodecs_wasm_sz3.Sz3(eb_mode="abs", eb_abs=0.01)
