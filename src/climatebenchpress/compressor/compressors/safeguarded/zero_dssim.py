__all__ = ["SafeguardedZeroDssim"]

import numcodecs_safeguards
import numcodecs_zero

from ..abc import Compressor


class SafeguardedZeroDssim(Compressor):
    """Safeguarded all-zero compressor that also safeguards the dSSIM score."""

    name = "safeguarded-zero-dssim"
    description = "Safeguarded(0, dSSIM)"

    @staticmethod
    def abs_bound_codec(error_bound, data_min_2d=None, data_max_2d=None, **kwargs):
        assert data_min_2d is not None, "data_min_2d must be provided"
        assert data_max_2d is not None, "data_max_2d must be provided"

        return numcodecs_safeguards.SafeguardedCodec(
            codec=numcodecs_zero.ZeroCodec(),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
                # guarantee that the per-latitude-longitude-slice minimum and
                #  maximum are preserved, which simplifies the rescaling
                dict(kind="sign", offset="x_min"),
                dict(kind="sign", offset="x_max"),
                dict(
                    kind="qoi_eb_pw",
                    qoi="""
                    # === pointwise dSSIM quantity of interest === #

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
            # use data_min_2d instead of $x_min since we need the minimum per
            #  2d latitude-longitude slice
            fixed_constants=dict(x_min=data_min_2d, x_max=data_max_2d),
        )

    @staticmethod
    def rel_bound_codec(error_bound, data_min_2d=None, data_max_2d=None, **kwargs):
        assert data_min_2d is not None, "data_min_2d must be provided"
        assert data_max_2d is not None, "data_max_2d must be provided"

        return numcodecs_safeguards.SafeguardedCodec(
            codec=numcodecs_zero.ZeroCodec(),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
                # guarantee that the per-latitude-longitude-slice minimum and
                #  maximum are preserved, which simplifies the rescaling
                dict(kind="sign", offset="x_min"),
                dict(kind="sign", offset="x_max"),
                dict(
                    kind="qoi_eb_pw",
                    qoi="""
                    # === pointwise dSSIM quantity of interest === #

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
            # use data_min_2d instead of $x_min since we need the minimum per
            #  2d latitude-longitude slice
            fixed_constants=dict(x_min=data_min_2d, x_max=data_max_2d),
        )
