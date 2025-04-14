__all__ = ["MAE"]

import numpy as np
import xarray as xr

from .abc import Metric


class MAE(Metric):
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        """
        Compute the mean absolute error between two inputs.

        Parameters
        ----------
        x : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        y : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        """
        return float(np.abs(x - y).mean(skipna=True))
