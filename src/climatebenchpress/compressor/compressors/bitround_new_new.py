__all__ = ["BitRoundNewNew"]

import numcodecs_shuffle
import numcodecs_wasm_bit_round
import numcodecs_wasm_zstd
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor


class BitRoundNewNew(Compressor):
    """Bit Rounding compressor.

    This compressor applies bit rounding to the data, which reduces the precision of the data
    while preserving its overall structure. It then applies the Zstandard lossless codec
    for further compression.
    """

    name = "bitround-new-new"
    description = "Bit Rounding"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(mode="abs", eb_abs=error_bound),
            numcodecs_shuffle.TypedByteShuffleCodec(),
            numcodecs_wasm_zstd.Zstd(level=3),
        )

    @staticmethod
    def rel_bound_codec(error_bound, **kwargs):
        return CodecStack(
            numcodecs_wasm_bit_round.BitRound(mode="rel", eb_rel=error_bound),
            numcodecs_shuffle.TypedByteShuffleCodec(),
            numcodecs_wasm_zstd.Zstd(level=3),
        )
