__all__ = ["BitRoundPco"]

from collections import defaultdict

import numcodecs_wasm_bit_round
import numcodecs_wasm_pco
from numcodecs_combinators.stack import CodecStack

from .abc import Compressor, NamedCodec
from .utils import compute_keepbits, convert_abs_error_to_rel_error


class BitRoundPco(Compressor):
    name = "bitround-pco"
    description = "Bit Rounding + PCodec"

    @staticmethod
    def build(
        dtype, data_abs_min, data_abs_max, error_bounds
    ) -> dict[str, list[NamedCodec]]:
        codecs = defaultdict(list)
        bounds = list(zip([BitRoundPco.name] * len(error_bounds), error_bounds))
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
                        numcodecs_wasm_pco.Pco(
                            level=8,
                            mode="auto",
                            delta="auto",
                            paging="equal-pages-up-to",
                        ),
                    ),
                )
            )

        return codecs
