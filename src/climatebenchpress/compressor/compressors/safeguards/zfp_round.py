__all__ = ["SafeguardsZfpRound"]

import numcodecs_safeguards
import numcodecs_wasm_zfp

from ..abc import Compressor


class SafeguardsZfpRound(Compressor):
    """Safeguarded ZFP-ROUND compressor.

    This is an adjusted version of the ZFP compressor with an improved rounding mechanism
    for the transform coefficients.
    """

    name = "safeguards-zfp-round"
    description = "Safeguards(ZFP-ROUND)"

    # NOTE:
    # ZFP mechanism for strictly supporting relative error bounds is to
    # truncate the floating point bit representation and then use ZFP's lossless
    # mode for compression. This is essentially equivalent to the BitRound
    # compressors we are already implementing (with a difference what the lossless
    # compression algorithm is).
    # See https://zfp.readthedocs.io/en/release1.0.1/faq.html#q-relerr for more details.

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardsCodec(
            codec=numcodecs_wasm_zfp.Zfp(mode="fixed-accuracy", tolerance=error_bound),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
            ],
        )

    @staticmethod
    def rel_bound_codec(error_bound, *, data_abs_min=None, **kwargs):
        assert data_abs_min is not None, "data_abs_min must be provided"

        return numcodecs_safeguards.SafeguardsCodec(
            # conservative rel->abs error bound transformation,
            #  same as convert_rel_error_to_abs_error
            # so that we can inform the safeguards of the rel bound
            codec=numcodecs_wasm_zfp.Zfp(
                mode="fixed-accuracy", tolerance=error_bound * data_abs_min
            ),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
            ],
        )
