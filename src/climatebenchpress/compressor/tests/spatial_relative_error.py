import numpy as np
import xarray as xr

from .abc import Test


class SRE(Test):
    """Test whether the spatial relative error between two arrays exceeds a threshold.

    The spatial relative error is defined as the percentage of grid points where the relative error
    exceeds a given delta, i.e. $\\frac{1}{N} \\sum_i \\mathbb{I}[|x_i - y_i| / |x_i| > \\delta]$.

    Default threshold and delta is taken from [1].


    [1] Baker, Allison H., Haiying Xu, Dorit M. Hammerling, Shaomeng Li, and John P. Clyne. "Toward a multi-method approach: Lossy data compression for climate simulation data." In International conference on high Performance computing, pp. 30-42. Cham: Springer International Publishing, 201

    Parameters
    ----------
    delta : float
        The relative error threshold for each grid point.
    threshold : float
        The maximum proportion of grid points that can exceed the relative error threshold.
    """

    def __init__(self, delta: float = 1e-4, threshold: float = 0.05):
        self.delta = delta
        self.threshold = threshold

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> tuple[bool, float]:
        relative_error = (np.abs(x - y) / x) > self.delta
        # Proportion of entries that exceed the threshold.
        exceed_thresh = np.sum(relative_error) / x.size
        return bool(exceed_thresh < self.threshold), float(exceed_thresh)
