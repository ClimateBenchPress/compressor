__all__ = ["PSNR"]

import numpy as np
import xarray as xr

from .abc import Metric


class PSNR(Metric):
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        """
        Compute the peak signal-to-noise ratio (PSNR) between an input `x` and
        its reconstruction `y`.

        This implementation assumes that the input data has shape
        (realization, time, vertical, latitude, longitude). The PSNR gets computed
        over the vertical, latitude, and longitude dimensions. The return value is then
        the average over the other dimensions (realization and time).

        Parameters
        ----------
        x : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        y : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        """
        mse = np.mean((x - y) ** 2, axis=(-3, -2, -1))
        max_pixel = np.max(x)
        psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
        return float(np.mean(psnr))
