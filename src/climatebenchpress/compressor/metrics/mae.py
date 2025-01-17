import numpy as np
import xarray as xr

from .abstract_metric import Metric


class MAE(Metric):
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        """
        Compute the mean squared error between two inputs.

        Parameters
        ----------
        x : xr.DataArray
            Shape (time, lon, lat, plev, realization)
        y : xr.DataArray
            Shape (time, lon, lat, plev, realization)
        """
        return float(np.mean(np.abs(x - y)))
