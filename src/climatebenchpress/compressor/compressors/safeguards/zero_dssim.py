__all__ = ["SafeguardsZeroDssim"]

import numcodecs_safeguards
import numcodecs_zero

from ..abc import Compressor


class SafeguardsZeroDssim(Compressor):
    """Safeguarded all-zero compressor that also safeguards the dSSIM score."""

    name = "safeguards-zero-dssim"
    description = "Safeguards(0, dSSIM)"

    @staticmethod
    def abs_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardsCodec(
            codec=numcodecs_zero.ZeroCodec(),
            safeguards=[
                dict(kind="eb", type="abs", eb=error_bound, equal_nan=True),
                # guarantee that the global minimum and maximum are preserved,
                #  which simplifies the rescaling
                dict(kind="sign", offset="$x_min"),
                dict(kind="sign", offset="$x_max"),
                dict(
                    kind="qoi_eb_pw",
                    qoi="""
                    # we guarantee that
                    #  min(data) = min(corrected) and
                    #  max(data) = max(corrected)
                    # with the sign safeguards above
                    v["smin"] = c["$x_min"];
                    v["smax"] = c["$x_max"];
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
        )

    @staticmethod
    def rel_bound_codec(error_bound, **kwargs):
        return numcodecs_safeguards.SafeguardsCodec(
            codec=numcodecs_zero.ZeroCodec(),
            safeguards=[
                dict(kind="eb", type="rel", eb=error_bound, equal_nan=True),
                # guarantee that the global minimum and maximum are preserved,
                #  which simplifies the rescaling
                dict(kind="sign", offset="$x_min"),
                dict(kind="sign", offset="$x_max"),
                dict(
                    kind="qoi_eb_pw",
                    qoi="""
                    # we guarantee that
                    #  min(data) = min(corrected) and
                    #  max(data) = max(corrected)
                    # with the sign safeguards above
                    v["smin"] = c["$x_min"];
                    v["smax"] = c["$x_max"];
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
        )
