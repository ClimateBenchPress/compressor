__all__ = ["Compressor", "NamedCodec", "ErrorBound"]

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

import numpy as np
from numcodecs.abc import Codec
from typed_classproperties import classproperty


@dataclass
class NamedCodec:
    name: str
    codec: Codec


@dataclass
class ErrorBound:
    abs_error: Optional[float] = None
    rel_error: Optional[float] = None

    def __post_init__(self):
        if self.abs_error is not None and self.rel_error is not None:
            raise ValueError(
                "Only one of 'abs_error' or 'rel_error' can be specified, not both."
            )
        if self.abs_error is None and self.rel_error is None:
            raise ValueError(
                "At least one of 'abs_error' or 'rel_error' must be specified."
            )

        self.name = (
            f"abs_error={self.abs_error}"
            if self.abs_error is not None
            else f"rel_error={self.rel_error}"
        )


class Compressor(ABC):
    # Abstract interface, must be implemented by subclasses
    name: str
    description: str

    @staticmethod
    @abstractmethod
    def abs_bound_codec(dtype: np.dtype, error_bound: float) -> Codec:
        pass

    @staticmethod
    @abstractmethod
    def rel_bound_codec(dtype: np.dtype, error_bound: float) -> Codec:
        pass

    @classmethod
    def build(
        cls,
        dtype: np.dtype,
        data_abs_min: dict[str, float],
        data_abs_max: dict[str, float],
        error_bounds: list[ErrorBound],
    ) -> dict[str, list[NamedCodec]]:
        """
        Constructs a dictionary of codecs based on the provided error bounds.
        The dictionary has a separate entry for compressor variant. Compressor
        variants are created when transforming between absolute and relative
        error bounds (each variant accounts for a different way to transform the
        error bound). The dictionary values are lists of `NamedCodec` instances
        where each element in the list corresponds to a different value for the
        error bound.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type of the input data.
        data_abs_min : dict[str, float]
            Minimum absolute value of the input data.
        data_abs_max : dict[str, float]
            Maximum absolute value of the input data.
        error_bounds: list[ErrorBound]
            List of error bounds to use for the compressor.

        Returns
        -------
        dict[str, list[NamedCodec]]
            A dictionary where keys are codec names and values are lists of
            `NamedCodec` instances configured with the specified error bounds.
        """
        codecs: dict[str, list[NamedCodec]] = defaultdict(list)
        bounds = list(zip([cls.name] * len(error_bounds), error_bounds))

        for variant_name, eb in bounds:
            if eb.abs_error is not None and cls.has_abs_error_impl:
                codec = cls.abs_bound_codec(dtype, eb.abs_error)
                codecs[variant_name].append(NamedCodec(name=eb.name, codec=codec))
            elif eb.rel_error is not None and cls.has_rel_error_impl:
                codec = cls.rel_bound_codec(dtype, eb.rel_error)
                codecs[variant_name].append(NamedCodec(name=eb.name, codec=codec))
            else:
                bounds += convert_error_bound(
                    variant_name, data_abs_min, data_abs_max, eb
                )

        return codecs

    # Class interface
    @classproperty
    def registry(cls) -> Mapping:
        return MappingProxyType(Compressor._registry)

    # Implementation details
    _registry: dict[str, type["Compressor"]] = dict()

    @classproperty
    def has_abs_error_impl(cls) -> bool:
        return "abs_bound_codec" in cls.__dict__

    @classproperty
    def has_rel_error_impl(cls) -> bool:
        return "rel_bound_codec" in cls.__dict__

    @classmethod
    def __init_subclass__(cls: type["Compressor"]) -> None:
        name = getattr(cls, "name", None)

        if name is None:
            raise TypeError(f"Compressor {cls} must have a name")

        if not (cls.has_abs_error_impl or cls.has_rel_error_impl):
            raise TypeError(
                f"Compressor {cls} must implement at least one of `abs_bound_codec` and `rel_bound_codec`."
            )

        if name in Compressor._registry:
            raise TypeError(
                f"duplicate Compressor name {name} for {cls} vs {Compressor._registry[name]}"
            )

        Compressor._registry[name] = cls

        return super().__init_subclass__()


def convert_error_bound(
    name: str,
    data_abs_min: dict[str, float],
    data_abs_max: dict[str, float],
    error_bound: ErrorBound,
) -> list[tuple[str, ErrorBound]]:
    if error_bound.abs_error is not None:
        new_ebs = convert_abs_error_to_rel_error(name, data_abs_max, error_bound)
    else:
        new_ebs = convert_rel_error_to_abs_error(name, data_abs_min, error_bound)

    # Keep the old name for all the new error bounds. This ensures we can group
    # together all transformed error bounds that came from the same original bound.
    for n, eb in new_ebs:
        eb.name = error_bound.name

    return new_ebs


def convert_rel_error_to_abs_error(
    name: str, data_abs_min: dict[str, float], old_error: ErrorBound
) -> list[tuple[str, ErrorBound]]:
    # In general, rel_error = abs_error / abs(data). This transformation
    # gives us the relative error bound that ensures the absolute error bound is
    # not exceeded for this dataset.
    assert old_error.rel_error is not None, "Expected relative error to be set."

    new_name = f"{name}-conservative-abs"
    abs_min_val = min(data_abs_min.values())
    error_bound = ErrorBound(abs_error=old_error.rel_error / abs_min_val)
    return [(new_name, error_bound)]


def convert_abs_error_to_rel_error(
    name: str, data_abs_max: dict[str, float], old_error: ErrorBound
) -> list[tuple[str, ErrorBound]]:
    # Same reasoning for error bound transformation as in `convert_rel_error_to_abs_error`.
    assert old_error.abs_error is not None, "Expected absolute error to be set."

    new_name = f"{name}-conservative-rel"
    abs_max_val = max(data_abs_max.values())
    error_bound = ErrorBound(rel_error=old_error.abs_error / abs_max_val)
    return [(new_name, error_bound)]
