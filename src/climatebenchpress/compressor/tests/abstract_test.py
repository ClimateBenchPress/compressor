from abc import ABC, abstractmethod

import xarray as xr


class Test(ABC):
    @abstractmethod
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> tuple[bool, float]:
        pass
