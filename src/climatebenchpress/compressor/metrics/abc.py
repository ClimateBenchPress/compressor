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
            Shape (realization, time, vertical, latitude, longitude)
        y : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)

        Returns
        -------
        metric : float
            The value of the computed metric
        """

        pass
