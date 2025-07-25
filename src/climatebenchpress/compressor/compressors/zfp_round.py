__all__ = ["ZfpRound"]

import numcodecs_wasm_zfp

from .abc import Compressor
from .utils import NONMANTISSA_BITS, compute_keepbits


class ZfpRound(Compressor):
    name = "zfp-round"
    description = "ZFP-ROUND"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_wasm_zfp.Zfp(mode="fixed-accuracy", tolerance=error_bound)

    # NOTE:
    # ZFP mechanism for strictly supporting relative error bounds is to
    # apply bitshaving and then use ZFP's lossless mode for compression.
    # See https://zfp.readthedocs.io/en/release1.0.1/faq.html#q-relerr for more details.
    @staticmethod
    def rel_bound_codec(error_bound, *, dtype=None, **kwargs):
        assert dtype is not None, "dtype must be provided"

        mantissa_keepbits = compute_keepbits(dtype, error_bound)
        total_keepbits = mantissa_keepbits + NONMANTISSA_BITS[dtype]
        return numcodecs_wasm_zfp.Zfp(
            mode="expert",
            min_bits=0,
            max_bits=0,
            max_prec=total_keepbits,
            min_exp=-1075,
        )
