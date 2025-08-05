__all__ = ["Tthresh"]

import numcodecs_wasm_tthresh

from .abc import Compressor


class Tthresh(Compressor):
    """Tthresh compressor."""

    name = "tthresh"
    description = "tthresh"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_wasm_tthresh.Tthresh(eb_mode="rmse", eb_rmse=error_bound)

    @staticmethod
    def rel_bound_codec(error_bound, **kwargs):
        return numcodecs_wasm_tthresh.Tthresh(eb_mode="eps", eb_rmse=error_bound)
