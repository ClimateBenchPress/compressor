import numpy as np
import xarray as xr

from .abstract_test import Test


class SRE(Test):
    def __init__(self, delta: float = 1e-4, threshold: float = 0.05):
        self.delta = delta
        self.threshold = threshold

    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> tuple[bool, float]:
        relative_error = (np.abs(x - y) / x) > self.delta
        # Proportion of entries that exceed the threshold.
        exceed_thresh = np.sum(relative_error) / x.size
        return bool(exceed_thresh < self.threshold), float(exceed_thresh)
