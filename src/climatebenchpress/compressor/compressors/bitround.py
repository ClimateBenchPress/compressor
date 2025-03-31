__all__ = ["BitRound"]

from collections import defaultdict

import numcodecs_wasm_bit_round
import numcodecs_wasm_zlib
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor, NamedCodec
from .utils import compute_keepbits, convert_abs_error_to_rel_error


class BitRound(Compressor):
    name = "bitround"
    description = "Bit Rounding"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, error_bounds
    ) -> dict[str, list[NamedCodec]]:
        codecs = defaultdict(list)
        bounds = list(zip([BitRound.name] * len(error_bounds), error_bounds))
        for name, eb in bounds:
            if eb.rel_error is None:
                bounds += convert_abs_error_to_rel_error(
                    name, data_abs_max, eb.abs_error
                )
                continue

            keepbits = compute_keepbits(dtype, eb.rel_error)
            codecs[name].append(
                NamedCodec(
                    name=eb.name,
                    codec=CodecStack(
                        numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
                        numcodecs_wasm_zlib.Zlib(level=6),
                    ),
                )
            )

        return codecs
