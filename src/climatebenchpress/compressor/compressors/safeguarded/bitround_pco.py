__all__ = ["SafeguardedBitRoundPco"]


import numcodecs_safeguards
import numcodecs_wasm_bit_round
import numcodecs_wasm_pco

from ..abc import Compressor
from ..utils import compute_keepbits


class SafeguardedBitRoundPco(Compressor):
    """Safeguarded Bit Rounding + PCodec compressor.

    This compressor first applies bit rounding to the data, which reduces the precision of the data
    while preserving its overall structure. After that, it uses PCodec for further compression.
    """

    name = "safeguarded-bitround-pco"
    description = "Safeguarded(Bit Rounding + PCodec)"

    @staticmethod
    def abs_bound_codec(error_bound, *, dtype=None, data_abs_max=None, **kwargs):
        assert dtype is not None, "dtype must be provided"
        assert data_abs_max is not None, "data_abs_max must be provided"

        # conservative abs->rel error bound transformation,
        #  same as convert_abs_error_to_rel_error
        # so that we can inform the safeguards of the abs bound
        keepbits = compute_keepbits(dtype, error_bound / data_abs_max)

        return numcodecs_safeguards.SafeguardsCodec(
            codec=numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            lossless=numcodecs_safeguards.lossless.Lossless(
                for_codec=numcodecs_wasm_pco.Pco(
                    level=8,
                    mode="auto",
                    delta="auto",
                    paging="equal-pages-up-to",
                )
            ),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
            ],
        )

    @staticmethod
    def rel_bound_codec(error_bound, *, dtype=None, **kwargs):
        assert dtype is not None, "dtype must be provided"

        keepbits = compute_keepbits(dtype, error_bound)

        return numcodecs_safeguards.SafeguardsCodec(
            codec=numcodecs_wasm_bit_round.BitRound(keepbits=keepbits),
            lossless=numcodecs_safeguards.lossless.Lossless(
                for_codec=numcodecs_wasm_pco.Pco(
                    level=8,
                    mode="auto",
                    delta="auto",
                    paging="equal-pages-up-to",
                )
            ),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )
