__all__ = ["Compressor", "NamedCodec", "ErrorBound"]

from abc import ABC, abstractmethod
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
    def build(
        dtype: np.dtype,
        data_abs_min: dict[str, float],
        data_abs_max: dict[str, float],
        error_bounds: list[ErrorBound],
    ) -> dict[str, list[NamedCodec]]:
        """
        Initialize a Codec instance for this particular compressor. Note that
        only one of `abs_error` or `rel_error` should be specified. The remaining
        arguments are passed to the function in order to be able to transform
        an absolute error bound to a relative error bound, and vice versa, if necessary.

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
        """
        pass

    # Class interface
    @classproperty
    def registry(cls) -> Mapping:
        return MappingProxyType(Compressor._registry)

    # Implementation details
    _registry: dict[str, type["Compressor"]] = dict()

    @classmethod
    def __init_subclass__(cls: type["Compressor"]) -> None:
        name = getattr(cls, "name", None)

        if name is None:
            raise TypeError(f"Compressor {cls} must have a name")

        if name in Compressor._registry:
            raise TypeError(
                f"duplicate Compressor name {name} for {cls} vs {Compressor._registry[name]}"
            )

        Compressor._registry[name] = cls

        return super().__init_subclass__()
