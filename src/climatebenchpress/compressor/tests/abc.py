from abc import ABC, abstractmethod

import xarray as xr


class Test(ABC):
    @abstractmethod
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> tuple[bool, float]:
        """
        Perform a test between two inputs.

        Parameters
        ----------
        x : xr.DataArray
            Shape (time, lon, lat, plev, realization)
        y : xr.DataArray
            Shape (time, lon, lat, plev, realization)

        Returns
        -------
        success, value : tuple[bool, float]
            The success of the test and the value of its statistic
        """

        pass
