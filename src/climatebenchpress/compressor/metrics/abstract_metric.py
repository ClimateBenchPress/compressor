from abc import ABC, abstractmethod

import xarray as xr


class Metric(ABC):
    @abstractmethod
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        pass
