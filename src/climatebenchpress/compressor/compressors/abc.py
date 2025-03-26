__all__ = ["Compressor"]

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType

from numcodecs.abc import Codec
from typed_classproperties import classproperty


class Compressor(ABC):
    # Abstract interface, must be implemented by subclasses
    name: str
    description: str

    @staticmethod
    @abstractmethod
    def build(
        dtype, data_abs_min, data_abs_max, abs_error=None, rel_error=None
    ) -> Codec:
        """
        Initialize a Codec instance for this particular compressor. Note that
        only one of `abs_error` or `rel_error` should be specified. The remaining
        arguments are passed to the function in order to be able to transform
        an absolute error bound to a relative error bound, and vice versa, if necessary.

        Parameters
        ----------
        dtype : numpy.dtype
            Data type of the input data.
        data_abs_min : float
            Minimum absolute value of the input data.
        data_abs_max : float
            Maximum absolute value of the input data.
        abs_error : float, optional
            Absolute error bound.
        rel_error : float, optional
            Relative error bound.
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
