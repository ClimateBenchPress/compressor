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


def create_rp_cr_compressor(cr: float) -> type[Compressor]:
    class RPCr(Compressor):
        name = f"rp-{cr}"
        description = f"Random Projection (Gaussian) with CR={cr}"

        @staticmethod
        def abs_bound_codec(error_bound, **kwargs):
            return numcodecs_random_projection.RPCodec(
                cr=cr,
                method="gaussian",
                seed=42,
                max_block_memory=2**28,  # 256 MiB
                debug=True,
            )

        @staticmethod
        def rel_bound_codec(error_bound, **kwargs):
            return numcodecs_random_projection.RPCodec(
                cr=cr,
                method="gaussian",
                seed=42,
                max_block_memory=2**28,  # 256 MiB
                debug=True,
            )

    return RPCr

create_rp_cr_compressor(2.0)
create_rp_cr_compressor(5.0)
create_rp_cr_compressor(10.0)
create_rp_cr_compressor(50.0)
create_rp_cr_compressor(100.0)
