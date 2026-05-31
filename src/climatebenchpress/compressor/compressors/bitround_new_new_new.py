__all__ = ["BitRoundNewNewNew"]

import bitshuffle
import numcodecs_wasm_bit_round
import numcodecs_wasm_zstd
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous_ndarray, ndarray_copy
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class BitRoundNewNewNew(Compressor):
    """Bit Rounding compressor.

    This compressor applies bit rounding to the data, which reduces the precision of the data
    while preserving its overall structure. It then applies the Zstandard lossless codec
    for further compression.
    """

    name = "bitround-new-new-new"
    description = "Bit Rounding"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(mode="abs", eb_abs=error_bound),
            BitShuffle(),
            numcodecs_wasm_zstd.Zstd(level=3),
        )

    @staticmethod
    def rel_bound_codec(error_bound, **kwargs):
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(mode="rel", eb_rel=error_bound),
            BitShuffle(),
            numcodecs_wasm_zstd.Zstd(level=3),
        )


class BitShuffle(Codec):
    codec_id = "bitshuffle"

    def encode(self, buf):
        buf = ensure_contiguous_ndarray(buf)

        self.dtype = buf.dtype

        return bitshuffle.bitshuffle(buf)

    def decode(self, buf, out=None):
        # FIXME: hack
        buf = ensure_contiguous_ndarray(buf).view(self.dtype)

        unshuffled = bitshuffle.bitunshuffle(buf)

        return ndarray_copy(unshuffled, out)
