__all__ = ["Zfp"]

import numcodecs_wasm_zfp

from .abc import Compressor, NamedCodec
from .utils import convert_rel_error_to_abs_error


class Zfp(Compressor):
    name = "zfp"
    description = "ZFP"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, error_bounds
    ) -> dict[str, list[NamedCodec]]:
        codecs = {Zfp.name: []}
        bounds = list(zip([Zfp.name] * len(error_bounds), error_bounds))
        for name, eb in bounds:
            if eb.abs_error is None:
                bounds += convert_rel_error_to_abs_error(
                    name, data_abs_max, eb.abs_error
                )
                continue

            codec = numcodecs_wasm_zfp.Zfp(
                mode="fixed-accuracy", tolerance=eb.abs_error
            )
            codecs[Zfp.name].append(NamedCodec(name=eb.name, codec=codec))

        return codecs
