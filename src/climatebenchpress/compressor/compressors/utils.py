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

SUBNORMAL = {
    # https://en.wikipedia.org/wiki/Single-precision_floating-point_format#Notable_single-precision_cases
    np.dtype("float32"): np.float32(2 ** (-126)) * (1 - np.float32(2 ** (-23))),
    # https://en.wikipedia.org/wiki/Double-precision_floating-point_format#Double-precision_examples
    np.dtype("float64"): np.float64(2 ** (-1022) * (1 - 2 ** (-52))),
    # https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples
    np.dtype("float16"): 2 ** (-14) * np.float16(1023 / 1024),
}


def compute_keepbits(dtype: np.dtype, rel_error: float, data_abs_min: float) -> int:
    """
    Computes the number of mantissa bits to keep in order to satisfy a relative error bound.

    Parameters
    ----------
    dtype : numpy.dtype
        Data type of the input array.
    rel_error : float
        Relative error bound.
    data_abs_min : float
        Minimum absolute value of the data, used to determine if the data is subnormal.

    Returns
    -------
    int
        Number of mantissa bits to keep.
    """
    if data_abs_min <= SUBNORMAL[dtype]:
        # If the data contains subnormal values, i.e. values with exponent  = 0,
        # then the formula below does not apply. Here we use the simple heuristic
        # of keeping all mantissa bits in that case. Tighter bounds on the error
        # exist, but they are not implemented here.
        return MANTISSA_BITS[dtype]

    # - log2(rel_error) specifies the number of mantissa bits needed to satisfy
    # the rel_error bound (https://en.wikipedia.org/wiki/Machine_epsilon).
    # We need to round up to the nearest integer to ensure the error bound is not
    # exceeded.
    keepbits = -math.floor(math.log2(rel_error)) - 1
    # Ensure that keepbits is within the range of the mantissa bits of single precision.
    keepbits = max(min(keepbits, MANTISSA_BITS[dtype]), 0)
    return keepbits
