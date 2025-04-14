__all__ = ["MAE"]

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
        # If we don't use xr.ufuncs, mypy cannot infer that the result is a DataArray
        abs_error = xr.ufuncs.abs(x - y)
        return float(abs_error.mean(skipna=True))
