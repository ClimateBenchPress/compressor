__all__ = ["Sz3"]

import numcodecs_wasm_sz3

from .abc import Compressor, NamedCodec


class Sz3(Compressor):
    name = "sz3"
    description = "SZ3"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, error_bounds
    ) -> dict[str, list[NamedCodec]]:
        codecs = {Sz3.name: []}
        for eb in error_bounds:
            if eb.abs_error is not None:
                codec = numcodecs_wasm_sz3.Sz3(eb_mode="abs", eb_abs=eb.abs_error)
            else:
                codec = numcodecs_wasm_sz3.Sz3(eb_mode="rel", eb_rel=eb.rel_error)
            codecs[Sz3.name].append(NamedCodec(name=eb.name, codec=codec))

        return codecs
