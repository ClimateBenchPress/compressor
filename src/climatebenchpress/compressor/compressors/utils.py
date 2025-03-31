__all__ = [
    "convert_rel_error_to_abs_error",
    "convert_abs_error_to_rel_error",
    "compute_keepbits",
]

import math

import numpy as np

from .abc import ErrorBound

MANTISSA_BITS = {
    np.dtype("float32"): 23,
    np.dtype("float64"): 52,
    np.dtype("float16"): 10,
}


def convert_rel_error_to_abs_error(
    name: str, data_abs_min: dict[str, float], rel_error: float
) -> list[tuple[str, ErrorBound]]:
    # In general, rel_error = abs_error / abs(data). This transformation
    # gives us the relative error bound that ensures the absolute error bound is
    # not exceeded for this dataset.
    new_name = f"{name}-conservative-abs"
    abs_min_val = min(data_abs_min.values())
    error_bound = ErrorBound(abs_error=rel_error / abs_min_val)
    error_bound.name = f"rel_error={rel_error}"
    return [(new_name, error_bound)]


def convert_abs_error_to_rel_error(
    name: str, data_abs_max: dict[str, float], abs_error: float
) -> list[tuple[str, ErrorBound]]:
    # In general, rel_error = abs_error / abs(data). This transformation
    # gives us the relative error bound that ensures the absolute error bound is
    # not exceeded for this dataset.
    new_name = f"{name}-conservative-rel"
    abs_max_val = max(data_abs_max.values())
    error_bound = ErrorBound(rel_error=abs_error / abs_max_val)
    error_bound.name = f"abs_error={abs_error}"
    return [(new_name, error_bound)]


def compute_keepbits(dtype, rel_error):
    # - log2(rel_error) specifies the number of mantissa bits needed to satisfy
    # the rel_error bound (https://en.wikipedia.org/wiki/Machine_epsilon).
    # We need to round up to the nearest integer to ensure the error bound is not
    # exceeded.
    keepbits = -math.floor(math.log2(rel_error)) - 1
    # Ensure that keepbits is within the range of the mantissa bits of single precision.
    keepbits = max(min(keepbits, MANTISSA_BITS[dtype]), 0)
    return keepbits
