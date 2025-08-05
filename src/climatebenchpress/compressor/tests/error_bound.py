import numpy as np
import xarray as xr

from .abc import Test


class ErrorBound(Test):
    """Tests whether the absolute or relative error between two arrays is below a threshold.

    Parameters
    ----------
    error_type : str
        Either "abs_error" for absolute error or "rel_error" for relative error.
    threshold : float
        The threshold value for the error.
    """

    def __init__(self, error_type: str, threshold: float):
        self.threshold = threshold
        assert error_type in [
            "abs_error",
            "rel_error",
        ], f"error_type must be either 'abs_error' or 'rel_error', not {error_type}"
        self.error_type = error_type

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> tuple[bool, float]:
        # Check that two arrays are both floats
        assert x.dtype.kind == "f", f"Expected x to be float, got {x.dtype}"
        assert y.dtype.kind == "f", f"Expected y to be float, got {y.dtype}"

        if self.error_type == "abs_error":
            # absolute error: |x-y| <= threshold
            satisfied = np.abs(x - y) <= self.threshold
        else:
            # relative error: |x-y| / |x| <= threshold
            #   to avoid dividing by zero, multiply both sides by |x|
            satisfied = np.abs(x - y) <= np.abs(x) * self.threshold

        # The comparison does not work for NaN values, as `np.nan < threshold` is False.
        # This check ensures that if x contains a NaN then y must also contain a NaN at
        # the same location.
        # Note, it is an error to have a NaN in y and not in x which will be caught.
        x_and_y_nan = np.isnan(x) & np.isnan(y)
        satisfied = satisfied | x_and_y_nan

        # Similarly, np.inf - np.inf is NaN but should pass the test.
        # The x == y condition ensures that their sign is the same.
        x_and_y_inf = np.isinf(x) & np.isinf(y) & (x == y)
        satisfied = satisfied | x_and_y_inf

        # Proportion of entries that exceed the threshold.
        exceed_thresh = np.sum(~satisfied) / x.size
        return bool(exceed_thresh == 0.0), float(exceed_thresh)
