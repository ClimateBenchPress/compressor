__all__ = ["Zfp"]

import numcodecs_wasm_zfp
from numcodecs.abc import Codec

from .abc import Compressor


class Zfp(Compressor):
    name = "zfp"
    description = "ZFP"

    @staticmethod
    def build(data_min, data_max, abs_error=None, rel_error=None) -> Codec:
        assert (abs_error is None) != (rel_error is None), (
            "Cannot specify both abs_error and rel_error."
        )

        if abs_error is None:
            # In general, rel_error = abs_error / abs(data). This transformation
            # gives us the absolute error bound that ensures the relative error bound is
            # not exceeded for this dataset.
            abs_error = rel_error * data_min
        return numcodecs_wasm_zfp.Zfp(mode="fixed-accuracy", tolerance=abs_error)
