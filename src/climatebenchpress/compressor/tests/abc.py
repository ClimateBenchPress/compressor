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
            Shape (realization, time, vertical, latitude, longitude)
        y : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)

        Returns
        -------
        success, value : tuple[bool, float]
            The success of the test and the value of its statistic
        """

        pass
