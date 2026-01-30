__all__ = ["SafeguardedZeroDssim"]

import numcodecs_safeguards
import numcodecs_zero

from ..abc import Compressor


class SafeguardedZeroDssim(Compressor):
    """Safeguarded all-zero compressor that also safeguards the dSSIM score."""

    name = "safeguarded-zero-dssim"
    description = "Safeguarded(0, dSSIM)"

    @staticmethod
    def abs_bound_codec(error_bound, data_min=None, data_max=None, **kwargs):
        assert data_min is not None, "data_min must be provided"
        assert data_max is not None, "data_max must be provided"

        return numcodecs_safeguards.SafeguardsCodec(
            codec=numcodecs_zero.ZeroCodec(),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
                # guarantee that the global minimum and maximum are preserved,
                #  which simplifies the rescaling
                dict(kind="sign", offset="x_min"),
                dict(kind="sign", offset="x_max"),
                dict(
                    kind="qoi_eb_pw",
                    qoi="""
                    # we guarantee that
                    #  min(data) = min(corrected) and
                    #  max(data) = max(corrected)
                    # with the sign safeguards above
                    v["smin"] = c["x_min"];
                    v["smax"] = c["x_max"];
                    v["r"] = v["smax"] - v["smin"];

                    # re-scale to [0-1] and quantize to 256 bins
                    v["sc_a2"] = round_ties_even(((x - v["smin"]) / v["r"]) * 255) / 255;

                    # force the quantized value to stay the same
                    return v["sc_a2"];
                    """,
                    type="abs",
                    eb=0,
                ),
            ],
            # use data_min instead of $x_min to allow for chunking
            fixed_constants=dict(x_min=data_min, x_max=data_max),
        )

    @staticmethod
    def rel_bound_codec(error_bound, data_min=None, data_max=None, **kwargs):
        assert data_min is not None, "data_min must be provided"
        assert data_max is not None, "data_max must be provided"

        return numcodecs_safeguards.SafeguardsCodec(
            codec=numcodecs_zero.ZeroCodec(),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
                # guarantee that the global minimum and maximum are preserved,
                #  which simplifies the rescaling
                dict(kind="sign", offset="x_min"),
                dict(kind="sign", offset="x_max"),
                dict(
                    kind="qoi_eb_pw",
                    qoi="""
                    # we guarantee that
                    #  min(data) = min(corrected) and
                    #  max(data) = max(corrected)
                    # with the sign safeguards above
                    v["smin"] = c["x_min"];
                    v["smax"] = c["x_max"];
                    v["r"] = v["smax"] - v["smin"];

                    # re-scale to [0-1] and quantize to 256 bins
                    v["sc_a2"] = round_ties_even(((x - v["smin"]) / v["r"]) * 255) / 255;

                    # force the quantized value to stay the same
                    return v["sc_a2"];
                    """,
                    type="abs",
                    eb=0,
                ),
            ],
            # use data_min instead of $x_min to allow for chunking
            fixed_constants=dict(x_min=data_min, x_max=data_max),
        )
