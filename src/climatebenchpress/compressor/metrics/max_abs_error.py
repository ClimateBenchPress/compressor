__all__ = ["MaxAbsError"]

import numpy as np
import xarray as xr

from .abc import Metric


class MaxAbsError(Metric):
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        """
        Compute the maximum absolute error between two inputs.

        Parameters
        ----------
        x : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        y : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        """
        return float(np.abs(x - y).max(skipna=True))
