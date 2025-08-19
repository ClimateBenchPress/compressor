# This module contains code derived from the PySTEPS library.
# BSD 3-Clause License

# Copyright (c) 2019, PySteps developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__all__ = ["SpectralError"]

import numpy as np
import xarray as xr

from .abc import Metric


class SpectralError(Metric):
    def __call__(self, x: xr.DataArray, y: xr.DataArray) -> float:
        """
        Compute the mean squared error in the spectral energy between the two inputs.
        Spectral energy is computed using the radially averaged power spectral density
        on the (lon, lat) 2D field and then the error is averaged over the remaining
        dimensions.

        Parameters
        ----------
        x : xr.DataArray
            Shape: (realization, time, vertical, latitude, longitude)
        y : xr.DataArray
            Shape: (realization, time, vertical, latitude, longitude)
        """
        num_lat, num_lon = x.shape[3], x.shape[4]
        # transpose to (realization, time, vertical), latitude, longitude
        x_ = x.values.reshape(-1, num_lat, num_lon)
        y_ = y.values.reshape(-1, num_lat, num_lon)
        # Filter out rows with NaNs
        valid_rows = np.logical_not(
            np.any(np.isnan(x_), axis=(1, 2)) | np.any(np.isnan(y_), axis=(1, 2))
        )

        true_spectrum = vec_rapsd(x_[valid_rows])
        pred_spectrum = vec_rapsd(y_[valid_rows])
        mean_error = np.mean((true_spectrum - pred_spectrum) ** 2)
        return mean_error


def compute_centred_coord_array(M: int, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a 2D coordinate array, where the origin is at the center.
    This function is based of https://github.com/pySTEPS/pysteps/blob/a7dae548ad26da74c7c2f1706a154defb03377aa/pysteps/utils/arrays.py#L16

    Parameters
    ----------
    M : int
      The height of the array.
    N : int
      The width of the array.

    Returns
    -------
    out : ndarray
      The coordinate array.

    Examples
    --------
    >>> compute_centred_coord_array(2, 2)

    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))
    """

    if M % 2 == 1:
        s1 = np.s_[-int(M / 2) : int(M / 2) + 1]
    else:
        s1 = np.s_[-int(M / 2) : int(M / 2)]

    if N % 2 == 1:
        s2 = np.s_[-int(N / 2) : int(N / 2) + 1]
    else:
        s2 = np.s_[-int(N / 2) : int(N / 2)]

    YC, XC = np.ogrid[s1, s2]

    return YC, XC


def vec_rapsd(field, return_freq=False, d=1.0, normalize=False):
    """
    Compute radially averaged power spectral density (RAPSD) from the given
    input field. This is a vectorized version of the rapsd function at:
    https://github.com/pySTEPS/pysteps/blob/a7dae548ad26da74c7c2f1706a154defb03377aa/pysteps/utils/spectral.py#L100

    The input field can now be a 3D array.

    Parameters
    ----------
    field: array_like
        A 3D array of shape (b, m, n) containing the input field, where b is the
        batch dimension.
    return_freq: bool
        Whether to also return the Fourier frequencies.
    d: scalar
        Sample spacing (inverse of the sampling rate). Defaults to 1.
        Applicable if return_freq is 'True'.
    normalize: bool
        If True, normalize the power spectrum so that it sums to one.

    Returns
    -------
    out: ndarray
      2D array containing the RAPSD with shape (b, l), where l is
      int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
    freq: ndarray
      One-dimensional array containing the Fourier frequencies.
    """

    b, m, n = field.shape

    yc, xc = compute_centred_coord_array(m, n)
    r_grid = np.sqrt(xc * xc + yc * yc).round()
    max_l = max(m, n)

    if max_l % 2 == 1:
        r_range = np.arange(0, int(max_l / 2) + 1)
    else:
        r_range = np.arange(0, int(max_l / 2))

    psd = np.fft.fftshift(np.fft.fft2(field))
    psd = np.abs(psd) ** 2 / (m * n)

    result = np.zeros((b, len(r_range)))
    for r in r_range:
        mask = r_grid == r
        result[:, r] = np.mean(psd, axis=(-1, -2), where=mask[None, :, :])

    if normalize:
        result /= np.sum(result, axis=1)

    if return_freq:
        freq = np.fft.fftfreq(max_l, d=d)
        freq = freq[r_range]
        return result, freq
    else:
        return result
