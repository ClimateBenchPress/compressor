__all__ = [
    "compute_keepbits",
]

import math

import numpy as np

MANTISSA_BITS = {
    np.dtype("float32"): 23,
    np.dtype("float64"): 52,
    np.dtype("float16"): 10,
}

NONMANTISSA_BITS = {
    np.dtype("float32"): 32 - MANTISSA_BITS[np.dtype("float32")],
    np.dtype("float64"): 64 - MANTISSA_BITS[np.dtype("float64")],
    np.dtype("float16"): 16 - MANTISSA_BITS[np.dtype("float16")],
}


def compute_keepbits(dtype: np.dtype, rel_error: float) -> int:
    """
    Computes the number of mantissa bits to keep in order to satisfy a relative error bound.

    Parameters
    ----------
    dtype : numpy.dtype
        Data type of the input array.
    rel_error : float
        Relative error bound.

    Returns
    -------
    int
        Number of mantissa bits to keep.
    """
    # - log2(rel_error) specifies the number of mantissa bits needed to satisfy
    # the rel_error bound (https://en.wikipedia.org/wiki/Machine_epsilon).
    # We need to round up to the nearest integer to ensure the error bound is not
    # exceeded.
    keepbits = -math.floor(math.log2(rel_error)) - 1
    # Ensure that keepbits is within the range of the mantissa bits of single precision.
    keepbits = max(min(keepbits, MANTISSA_BITS[dtype]), 0)
    return keepbits
