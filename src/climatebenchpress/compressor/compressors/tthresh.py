__all__ = ["Tthresh"]

import numcodecs_wasm_tthresh

from .abc import Compressor, NamedCodec


class Tthresh(Compressor):
    name = "tthresh"
    description = "tthresh"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, error_bounds
    ) -> dict[str, list[NamedCodec]]:
        codecs = {Tthresh.name: []}
        for eb in error_bounds:
            if eb.abs_error is not None:
                codec = numcodecs_wasm_tthresh.Tthresh(
                    eb_mode="rmse", eb_rmse=eb.abs_error
                )
            else:
                codec = numcodecs_wasm_tthresh.Tthresh(
                    eb_mode="eps", eb_eps=eb.rel_error
                )
            codecs[Tthresh.name].append(NamedCodec(name=eb.name, codec=codec))

        return codecs
