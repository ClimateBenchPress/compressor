__all__ = ["DSSIM"]

import numpy as np
import xarray as xr
from astropy.convolution import Gaussian2DKernel, convolve

from .abc import Metric


class DSSIM(Metric):
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        """
        Implementation of the data-SSIM (dSSIM) metric presented in [1]. This is an
        extension of the standard structural similarity index (SSIM) to floating
        point data.

        Here we assume that the input data has shape (realization, time, vertical, latitude, longitude).
        The dSSIM metric is defined for 2D fields, so we compute the dSSIM for each vertical slice
        and then take the minimum value over all vertical slices (this follows the official implementation
        of [1]). The final dSSIM value is the average over the realization and time dimensions.

        NOTE: This implementation can return values > 1.0 in the case that one of the inputs
          has large regions with NaNs and the other input does not. This is because the
          `astropy.convolution.convolve` function linearly interpolates the NaN values.
          The interpolation of NaN is an explicit design decision made in [1]. In practice,
          this metric should not be used for data with large regions of NaNs.

        References:
        [1] A. H. Baker, A. Pinard and D. M. Hammerling, "On a Structural Similarity
            Index Approach for Floating-Point Data," in IEEE Transactions on Visualization
            and Computer Graphics

        Parameters
        ----------
        x : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        y : xr.DataArray
            Shape (realization, time, vertical, latitude, longitude)
        """
        _, _, num_vert, num_lat, num_lon = x.shape
        x_ = x.values.reshape(-1, num_vert, num_lat, num_lon)
        y_ = y.values.reshape(-1, num_vert, num_lat, num_lon)
        dssims = np.zeros(x_.shape[0])
        for i in range(x_.shape[0]):
            vertical_dssims = np.zeros(num_vert)
            for j in range(num_vert):
                vertical_dssims[j] = _dssim(x_[i, j], y_[i, j])
            dssims[i] = vertical_dssims.min()
        return dssims.mean()


def _dssim(
    a1: np.ndarray,
    a2: np.ndarray,
    eps: float = 1e-8,
    kernel_size: tuple[int, int] = (11, 11),
) -> float:
    """
    Implementation adapted from the official dSSIM implementation at
    https://github.com/NCAR/ldcpy/blob/6c5bcb8149ec7876a4f53b0e784e9c528f6f14cb/ldcpy/calcs.py#L2516

    The official implementation makes assumptions about the input data that are
    specific to models developed at NCAR which is why we cannot use the official
    implementation directly.

    Parameters
    ----------
    x : np.ndarray
        Shape: (latitude, longitude)
    y : np.ndarray
        Shape: (latitude, longitude)

    Returns
    -------
    float
        The data-SSIM value between the two input arrays.
    """
    # re-scale  to [0,1] - if not constant
    smin = min(np.nanmin(a1), np.nanmin(a2))
    smax = max(np.nanmax(a1), np.nanmax(a2))
    r = smax - smin
    if r == 0.0:  # scale by smax if field is a constant (and smax != 0)
        if smax == 0.0:
            sc_a1 = a1
            sc_a2 = a2
        else:
            sc_a1 = a1 / smax
            sc_a2 = a2 / smax
    else:
        sc_a1 = (a1 - smin) / r
        sc_a2 = (a2 - smin) / r

    # now quantize to 256 bins
    sc_a1 = np.round(sc_a1 * 255) / 255
    sc_a2 = np.round(sc_a2 * 255) / 255

    # gaussian filter
    kernel = Gaussian2DKernel(
        x_stddev=1.5, x_size=kernel_size[0], y_size=kernel_size[1]
    )
    k = 5
    filter_args = {"boundary": "fill", "preserve_nan": True}

    a1_mu = convolve(sc_a1, kernel, **filter_args)
    a2_mu = convolve(sc_a2, kernel, **filter_args)

    a1a1 = convolve(sc_a1 * sc_a1, kernel, **filter_args)
    a2a2 = convolve(sc_a2 * sc_a2, kernel, **filter_args)

    a1a2 = convolve(sc_a1 * sc_a2, kernel, **filter_args)

    ###########
    var_a1 = a1a1 - a1_mu * a1_mu
    var_a2 = a2a2 - a2_mu * a2_mu
    cov_a1a2 = a1a2 - a1_mu * a2_mu

    # ssim constants
    C1 = eps
    C2 = eps

    ssim_t1 = 2 * a1_mu * a2_mu + C1
    ssim_t2 = 2 * cov_a1a2 + C2

    ssim_b1 = a1_mu * a1_mu + a2_mu * a2_mu + C1
    ssim_b2 = var_a1 + var_a2 + C2

    ssim_1 = ssim_t1 / ssim_b1
    ssim_2 = ssim_t2 / ssim_b2
    ssim_mat = ssim_1 * ssim_2

    # cropping (the border region)
    ssim_mat = ssim_mat[k : ssim_mat.shape[0] - k, k : ssim_mat.shape[1] - k]
    return np.nanmean(ssim_mat)
