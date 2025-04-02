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

type ErrorBoundName = str
type VariableName = str
type VariantName = str


@dataclass
class NamedCodec:
    name: ErrorBoundName
    codecs: dict[VariableName, Codec]


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


@dataclass
class VariantInfo:
    name: VariantName
    error_bounds: dict[VariableName, ErrorBound]


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
        dtypes: dict[VariableName, np.dtype],
        data_abs_min: dict[VariableName, float],
        data_abs_max: dict[VariableName, float],
        error_bounds: list[dict[VariableName, ErrorBound]],
    ) -> dict[VariantName, list[NamedCodec]]:
        """
        Constructs a dictionary of codecs based on the provided error bounds.
        The dictionary has a separate entry for each compressor variant. Compressor
        variants are created when transforming between absolute and relative
        error bounds (each variant accounts for a different way to transform the
        error bound). The dictionary values are lists of `NamedCodec` instances
        where each element in the list corresponds to a different value for the
        error bound.

        Parameters
        ----------
        dtype : dict[VariableName, numpy.dtype]
            Dict mapping from variable name to data type of the input data.
        data_abs_min : dict[VariableName, float]
            Dict mapping from variable name to minimum absolute value for the variable.
        data_abs_max : dict[VariableName, float]
            Dict mapping from variable name to maximum absolute value for the variable.
        error_bounds: list[ErrorBound]
            List of error bounds to use for the compressor.

        Returns
        -------
        dict[VariantName, list[NamedCodec]]
            A dictionary where keys are codec variant names (for separate error bound conversions)
            and values are lists of `NamedCodec` instances configured with the specified error bounds.
        """
        codecs: dict[VariantName, list[NamedCodec]] = defaultdict(list)
        transformed_bounds: list[VariantInfo] = []

        # Loop over all the error bounds and ensure that they are compatible with the
        # compressor. If the error bound is not compatible, transform it into a new
        # error bound that is compatible.
        for eb_per_var in error_bounds:
            new_bounds = cls._get_variant_bounds(
                data_abs_min, data_abs_max, cls.name, eb_per_var
            )
            if new_bounds is not None:
                transformed_bounds += new_bounds
            else:
                transformed_bounds.append(VariantInfo(cls.name, eb_per_var))

        # For each error bound, create a new codec.
        for variant_info in transformed_bounds:
            variant_name, eb_per_var = variant_info.name, variant_info.error_bounds
            new_codecs: dict[VariableName, Codec] = dict()
            for var, eb in eb_per_var.items():
                if eb.abs_error is not None and cls.has_abs_error_impl:
                    new_codecs[var] = cls.abs_bound_codec(dtypes[var], eb.abs_error)
                elif eb.rel_error is not None and cls.has_rel_error_impl:
                    new_codecs[var] = cls.rel_bound_codec(dtypes[var], eb.rel_error)
                else:
                    # This should never happen as we have already transformed the error bounds.
                    # If this happens, it means there is a bug in the implementation.
                    # We raise an error here to avoid silent failures.
                    raise ValueError(
                        "Error bound is not compatible with the compressor."
                    )

            error_bound_name = "_".join(
                f"{var}-{eb.name}" for var, eb in eb_per_var.items()
            )
            codecs[variant_name].append(
                NamedCodec(name=error_bound_name, codecs=new_codecs)
            )

        return codecs

    @classmethod
    def _get_variant_bounds(
        cls,
        data_abs_min: dict[VariableName, float],
        data_abs_max: dict[VariableName, float],
        variant_name: VariantName,
        error_bounds: dict[VariableName, ErrorBound],
    ) -> Optional[list[VariantInfo]]:
        """
        Check whether the supplied `error_bounds` are compatible with the current
        compressor. If they are not compatible return a list of new transformed
        error bounds.
        """
        new_bounds: Optional[list[VariantInfo]] = None
        for var, error_bound in error_bounds.items():
            abs_bound_codec = (
                error_bound.abs_error is not None and cls.has_abs_error_impl
            )
            rel_bound_codec = (
                error_bound.rel_error is not None and cls.has_rel_error_impl
            )
            if abs_bound_codec or rel_bound_codec:
                # If codec is compatible with the error bound no transformation
                # is needed.
                continue

            converted_bounds = convert_error_bound(
                variant_name, data_abs_min[var], data_abs_max[var], error_bound
            )
            if new_bounds is not None:
                # Update the already created new variant bounds with the transformed
                # error bounds for the current variable. Before this update, the
                # error_bounds `new_bounds` will not have any entries for `var`.
                for i, (new_variant_name, bound) in enumerate(converted_bounds):
                    # These assertions checks code correctness rather than
                    # input validity. All the error bounds that need to be transformed
                    # should lead to the same number and type of new bounds.
                    # If this assertion fails, it means there is a bug in the implementation.
                    assert new_bounds[i].name == new_variant_name, (
                        f"Cannot assign bound of variant '{new_variant_name}' to bound of variant '{new_bounds[i].name}'"
                    )
                    assert var not in new_bounds[i].error_bounds
                    new_bounds[i].error_bounds[var] = bound
            else:
                # Create new variant bounds and initialize them with the transformed error
                # bounds for the current variable.
                new_bounds = [
                    VariantInfo(name=variant_name, error_bounds={var: eb})
                    for variant_name, eb in converted_bounds
                ]

        if new_bounds is None:
            # The error bounds for all variables are compatible with the codec.
            return new_bounds

        # For the variables which didn't need their error bounds transformed,
        # we need to copy the untransformed error bounds into the `new_bounds`.
        all_vars = set(error_bounds.keys())
        for i in range(len(new_bounds)):
            updated_vars = set(new_bounds[i].error_bounds.keys())
            missing_vars = all_vars - updated_vars
            for var in missing_vars:
                new_bounds[i].error_bounds[var] = error_bounds[var]

        return new_bounds

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
    data_abs_min: float,
    data_abs_max: float,
    error_bound: ErrorBound,
) -> list[tuple[VariantName, ErrorBound]]:
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
    name: str, data_abs_min: float, old_error: ErrorBound
) -> list[tuple[str, ErrorBound]]:
    # In general, rel_error = abs_error / abs(data). This transformation
    # gives us the relative error bound that ensures the absolute error bound is
    # not exceeded for this dataset.
    assert old_error.rel_error is not None, "Expected relative error to be set."

    new_name = f"{name}-conservative-abs"
    error_bound = ErrorBound(abs_error=old_error.rel_error * data_abs_min)
    return [(new_name, error_bound)]


def convert_abs_error_to_rel_error(
    name: str, data_abs_max: float, old_error: ErrorBound
) -> list[tuple[str, ErrorBound]]:
    # Same reasoning for error bound transformation as in `convert_rel_error_to_abs_error`.
    assert old_error.abs_error is not None, "Expected absolute error to be set."

    new_name = f"{name}-conservative-rel"
    error_bound = ErrorBound(rel_error=old_error.abs_error / data_abs_max)
    return [(new_name, error_bound)]
