import numpy as np
import xarray as xr

from .abstract_metric import Metric


class MAE(Metric):
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        return float(np.mean(np.abs(x - y)))
