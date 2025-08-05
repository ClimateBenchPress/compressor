import numpy as np
import scipy.stats
import xarray as xr

from .abc import Test


class R2(Test):
    """Test whether the R^2 correlation coefficient between two arrays exceeds a threshold.

    Default threshold is taken from [1].

    [1] Baker, Allison H., Haiying Xu, Dorit M. Hammerling, Shaomeng Li, and John P. Clyne. "Toward a multi-method approach: Lossy data compression for climate simulation data." In International conference on high Performance computing, pp. 30-42. Cham: Springer International Publishing, 201
    """

    def __init__(self, threshold: float = 0.99999):
        self.threshold = threshold

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> tuple[bool, float]:
        nan_mask = ~np.isnan(x) & ~np.isnan(y)
        result = scipy.stats.linregress(
            x.values[nan_mask].flatten(), y.values[nan_mask].flatten()
        )
        r2 = float(result.rvalue**2)
        return bool(r2 > self.threshold), r2
