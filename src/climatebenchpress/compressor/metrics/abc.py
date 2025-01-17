from abc import ABC, abstractmethod

import xarray as xr


class Metric(ABC):
    @abstractmethod
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        """
        Compute a metric between two inputs.

        Parameters
        ----------
        x : xr.DataArray
            Shape (time, lon, lat, plev, realization)
        y : xr.DataArray
            Shape (time, lon, lat, plev, realization)

        Returns
        -------
        metric : float
            The value of the computed metric
        """

        pass
