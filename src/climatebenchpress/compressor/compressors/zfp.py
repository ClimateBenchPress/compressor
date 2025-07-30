__all__ = ["Zfp"]

import numcodecs_wasm_zfp_classic

from .abc import Compressor
from .utils import NONMANTISSA_BITS, compute_keepbits

# From: https://github.com/LLNL/zfp/blob/4baa4c7eeae8e0b6a7ace4dde242ac165bcd59d9/include/zfp.h#L18
ZFP_MIN_BITS = 1
ZFP_MAX_BITS = 16658


class Zfp(Compressor):
    name = "zfp"
    description = "ZFP"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_wasm_zfp_classic.ZfpClassic(
            mode="fixed-accuracy", tolerance=error_bound
        )

    # NOTE:
    # ZFP mechanism for strictly supporting relative error bounds is to
    # apply bitshaving and then use ZFP's lossless mode for compression.
    # See https://zfp.readthedocs.io/en/release1.0.1/faq.html#q-relerr for more details.
    @staticmethod
    def rel_bound_codec(error_bound, *, dtype=None, **kwargs):
        assert dtype is not None, "dtype must be provided"

        mantissa_keepbits = compute_keepbits(dtype, error_bound)
        total_keepbits = mantissa_keepbits + NONMANTISSA_BITS[dtype]
        return numcodecs_wasm_zfp_classic.ZfpClassic(
            mode="expert",
            min_bits=ZFP_MIN_BITS,
            max_bits=ZFP_MAX_BITS,
            max_prec=total_keepbits,
            min_exp=-1075,
        )
