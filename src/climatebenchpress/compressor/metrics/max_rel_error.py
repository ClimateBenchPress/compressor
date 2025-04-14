__all__ = ["MaxRelError"]

import numpy as np
import xarray as xr

from .abc import Metric


class MaxRelError(Metric):
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        """
        Compute the maximum relative error between two inputs.

        Parameters
        ----------
        x : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        y : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        """
        # Avoid dividing by zero when x is zero and y is also zero.
        rel_error = xr.where((x == 0) & (x == y), 0.0, np.abs(x - y) / np.abs(x))
        return float(rel_error.max(skipna=True))
