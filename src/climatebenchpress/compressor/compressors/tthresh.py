__all__ = ["Tthresh"]

import numcodecs_wasm_tthresh
from numcodecs.abc import Codec

from .abc import Compressor


class Tthresh(Compressor):
    name = "tthresh"
    description = "tthresh"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, abs_error=None, rel_error=None
    ) -> Codec:
        assert (abs_error is None) != (rel_error is None), (
            "Cannot specify both abs_error and rel_error."
        )

        if abs_error is not None:
            return numcodecs_wasm_tthresh.Tthresh(eb_mode="rmse", eb_rmse=abs_error)
        else:
            return numcodecs_wasm_tthresh.Tthresh(eb_mode="eps", eb_eps=rel_error)
