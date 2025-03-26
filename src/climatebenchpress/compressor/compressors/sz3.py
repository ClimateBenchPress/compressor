__all__ = ["Sz3"]

import numcodecs_wasm_sz3
from numcodecs.abc import Codec

from .abc import Compressor


class Sz3(Compressor):
    name = "sz3"
    description = "SZ3"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, abs_error=None, rel_error=None
    ) -> Codec:
        assert (abs_error is None) != (rel_error is None), (
            "Cannot specify both abs_error and rel_error."
        )

        if abs_error is not None:
            return numcodecs_wasm_sz3.Sz3(eb_mode="abs", eb_abs=abs_error)
        else:
            return numcodecs_wasm_sz3.Sz3(eb_mode="rel", eb_rel=rel_error)
