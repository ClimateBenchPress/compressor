__all__ = ["Tthresh"]

import numcodecs_wasm_tthresh
from numcodecs.abc import Codec

from .abc import Compressor


class Tthresh(Compressor):
    name = "tthresh"
    description = "tthresh"

    @staticmethod
    def build() -> Codec:
        return numcodecs_wasm_tthresh.Tthresh(eb_mode="rmse", eb_rmse=0.0001)
