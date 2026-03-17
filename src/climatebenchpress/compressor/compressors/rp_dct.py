__all__ = ["RPDct"]

import numcodecs_random_projection

from .abc import Compressor


class RPDct(Compressor):
    name = "rp-dct"
    description = "Random Projection (DCT)"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_random_projection.RPCodec(
            mae=error_bound,
            method="dct",
            max_block_memory=2**28,  # 256 MiB
            debug=True,
        )


def create_rp_dct_cr_compressor(cr: float) -> type[Compressor]:
    class RPDctCr(Compressor):
        name = f"rp-dct-{cr}"
        description = f"Random Projection (DCT) with CR={cr}"

        @staticmethod
        def abs_bound_codec(error_bound, **kwargs):
            return numcodecs_random_projection.RPCodec(
                cr=cr,
                method="dct",
                max_block_memory=2**28,  # 256 MiB
                debug=True,
            )

        @staticmethod
        def rel_bound_codec(error_bound, **kwargs):
            return numcodecs_random_projection.RPCodec(
                cr=cr,
                method="dct",
                max_block_memory=2**28,  # 256 MiB
                debug=True,
            )

    return RPDctCr


create_rp_dct_cr_compressor(2.0)
create_rp_dct_cr_compressor(5.0)
create_rp_dct_cr_compressor(10.0)
create_rp_dct_cr_compressor(50.0)
create_rp_dct_cr_compressor(100.0)
