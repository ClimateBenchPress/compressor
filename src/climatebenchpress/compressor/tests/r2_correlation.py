import numpy as np
import scipy.stats
import xarray as xr

from .abc import Test


class R2(Test):
    def __init__(self, threshold: float = 0.99999):
        self.threshold = threshold

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> tuple[bool, float]:
        nan_mask = ~np.isnan(x) & ~np.isnan(y)
        result = scipy.stats.linregress(
            x.values[nan_mask].flatten(), y.values[nan_mask].flatten()
        )
        r2 = result.rvalue**2
        return bool(r2 > self.threshold), r2
