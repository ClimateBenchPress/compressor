__all__ = ["RP"]

import numcodecs_random_projection

from .abc import Compressor


class RP(Compressor):
    name = "rp"
    description = "Random Projection (Gaussian)"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_random_projection.RPCodec(
            mae=error_bound,
            method="gaussian",
            seed=42,
            max_block_memory=2**28,  # 256 MiB
            debug=True,
        )
