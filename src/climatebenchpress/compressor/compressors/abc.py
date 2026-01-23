__all__ = ["Compressor", "NamedPerVariableCodec", "ErrorBound"]

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Callable, Optional

import numpy as np
from numcodecs.abc import Codec
from typed_classproperties import classproperty

type ErrorBoundName = str
type VariableName = str
type VariantName = str


@dataclass
class NamedPerVariableCodec:
    """Dataclass representing a codec for one dataset and compressor.

    Attributes
    ----------
    name : str
        Name of the error bound used to create the codecs, a combination of variable
        names and error bounds.
    codecs : dict[VariableName, Callable[[], Codec]]
        Dictionary mapping variable names to codec constructors.
    """

    name: ErrorBoundName
    codecs: dict[VariableName, Callable[[], Codec]]


@dataclass
class ErrorBound:
    """Dataclass representing an error bound for a variable.

    Can only have one of `abs_error` or `rel_error` set, not both.

    Attributes
    ----------
    abs_error : Optional[float]
        Absolute error bound for the variable.
    rel_error : Optional[float]
        Relative error bound for the variable.
    """

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
class VariantErrorBoundPerVariable:
    name: VariantName
    error_bounds: dict[VariableName, ErrorBound]


class Compressor(ABC):
    """Abstract base class for compressors."""

    name: str
    description: str

    @staticmethod
    @abstractmethod
    def abs_bound_codec(
        error_bound: float,
        *,
        dtype: Optional[np.dtype] = None,
        data_min: Optional[float] = None,
        data_max: Optional[float] = None,
        data_abs_min: Optional[float] = None,
        data_abs_max: Optional[float] = None,
    ) -> Codec:
        """Create a codec with an absolute error bound."""
        pass

    @staticmethod
    @abstractmethod
    def rel_bound_codec(
        error_bound: float,
        *,
        dtype: Optional[np.dtype] = None,
        data_min: Optional[float] = None,
        data_max: Optional[float] = None,
        data_abs_min: Optional[float] = None,
        data_abs_max: Optional[float] = None,
    ) -> Codec:
        """Create a codec with a relative error bound."""
        pass

    @classmethod
    def build(
        cls,
        dtypes: dict[VariableName, np.dtype],
        data_abs_min: dict[VariableName, float],
        data_abs_max: dict[VariableName, float],
        data_min: dict[VariableName, float],
        data_max: dict[VariableName, float],
        error_bounds: list[dict[VariableName, ErrorBound]],
    ) -> dict[VariantName, list[NamedPerVariableCodec]]:
        """
        Constructs a dictionary of codecs based on the provided error bounds.
        The dictionary has a separate entry for each compressor variant. Compressor
        variants are created when transforming between absolute and relative
        error bounds (each variant accounts for a different way to transform the
        error bound). The dictionary values are lists of `NamedPerVariableCodec` instances
        where each element in the list corresponds to a different value for the
        error bound.

        Parameters
        ----------
        dtypes : dict[VariableName, numpy.dtype]
            Dict mapping from variable name to data type of the input data.
        data_abs_min : dict[VariableName, float]
            Dict mapping from variable name to minimum absolute value for the variable.
        data_abs_max : dict[VariableName, float]
            Dict mapping from variable name to maximum absolute value for the variable.
        data_min : dict[VariableName, float]
            Dict mapping from variable name to minimum value for the variable.
        data_max : dict[VariableName, float]
            Dict mapping from variable name to maximum value for the variable.
        error_bounds: list[ErrorBound]
            List of error bounds to use for the compressor.

        Returns
        -------
        dict[VariantName, list[NamedPerVariableCodec]]
            A dictionary where keys are codec variant names (for separate error bound conversions)
            and values are lists of `NamedPerVariableCodec` instances configured with the specified error bounds.
        """
        codecs: dict[VariantName, list[NamedPerVariableCodec]] = defaultdict(list)
        transformed_bounds: list[VariantErrorBoundPerVariable] = []

        # Loop over all the error bounds and ensure that they are compatible with the
        # compressor. If the error bound is not compatible, transform it into a new
        # error bound that is compatible.
        for eb_per_var in error_bounds:
            transformed_bounds += cls._get_variant_bounds(
                data_abs_min, data_abs_max, cls.name, eb_per_var
            )

        # For each error bound, create a new codec.
        for variant_info in transformed_bounds:
            variant_name, eb_per_var = variant_info.name, variant_info.error_bounds
            new_codecs: dict[VariableName, Callable[[], Codec]] = dict()
            for var, eb in eb_per_var.items():
                if eb.abs_error is not None and cls.has_abs_error_impl:
                    new_codecs[var] = partial(
                        cls.abs_bound_codec,
                        eb.abs_error,
                        dtype=dtypes[var],
                        data_min=data_min[var],
                        data_max=data_max[var],
                        data_abs_min=data_abs_min[var],
                        data_abs_max=data_abs_max[var],
                    )
                elif eb.rel_error is not None and cls.has_rel_error_impl:
                    new_codecs[var] = partial(
                        cls.rel_bound_codec,
                        eb.rel_error,
                        dtype=dtypes[var],
                        data_min=data_min[var],
                        data_max=data_max[var],
                        data_abs_min=data_abs_min[var],
                        data_abs_max=data_abs_max[var],
                    )
                else:
                    # This should never happen as we have already transformed the error bounds.
                    # If this happens, it means there is a bug in the implementation.
                    # We raise an error here to avoid silent failures.
                    raise ValueError(
                        "Error bound is not compatible with the compressor."
                    )

            # Sort the error bounds by variable name to ensure consistent ordering.
            error_bound_name = "_".join(
                f"{var}-{eb.name}" for var, eb in sorted(eb_per_var.items())
            )
            codecs[variant_name].append(
                NamedPerVariableCodec(name=error_bound_name, codecs=new_codecs)
            )

        return codecs

    # Class interface
    @classproperty
    def registry(cls) -> Mapping[str, type["Compressor"]]:
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

    @classmethod
    def _get_variant_bounds(
        cls,
        data_abs_min: dict[VariableName, float],
        data_abs_max: dict[VariableName, float],
        variant_name: VariantName,
        error_bounds: dict[VariableName, ErrorBound],
    ) -> list[VariantErrorBoundPerVariable]:
        """
        Check whether the supplied `error_bounds` are compatible with the current
        compressor. If they are not compatible return a list of new transformed
        error bounds.
        """
        converted_bounds: dict[VariableName, dict[VariantName, ErrorBound]] = dict()
        variant_names = {cls.name}
        for var, error_bound in error_bounds.items():
            cls_has_abs_error_impl: bool = cls.has_abs_error_impl  # type: ignore
            abs_bound_codec = (
                error_bound.abs_error is not None and cls_has_abs_error_impl
            )
            cls_has_rel_error_impl: bool = cls.has_rel_error_impl  # type: ignore
            rel_bound_codec = (
                error_bound.rel_error is not None and cls_has_rel_error_impl
            )
            if abs_bound_codec or rel_bound_codec:
                # If codec is compatible with the error bound no transformation
                # is needed.
                continue

            converted_bounds[var] = convert_error_bound(
                variant_name, data_abs_min[var], data_abs_max[var], error_bound
            )
            if variant_names == {cls.name}:
                # This is the first time we are transforming the error bounds,
                # therefore we need to update the names of the generated variants.
                variant_names = set(converted_bounds[var].keys())
            else:
                # For all the variables if we are converting the error bounds
                # they should lead to the same number of variants.
                # If this is not the case, we are somehow using different mechanisms
                # to transform the same type of error bound which should be avoided.
                # This holds true as long as we have only two types of error bounds
                # (absolute and relative). If we add more types of error bounds then
                # this property no longer holds.
                assert variant_names == set(converted_bounds[var].keys()), (
                    "Error bounds for different variables must have the same variant names."
                )

        if len(converted_bounds) == 0:
            # The error bounds for all variables are compatible with the codec.
            # Just return the original error bounds.
            return [VariantErrorBoundPerVariable(variant_name, error_bounds)]

        # converted_bounds contains entries for all variables for which we needed
        # to transform the error bounds. We now transform the dictionary
        # dict[VariableName, dict[VariantName, ErrorBound]] into a list in which
        # each entry represents one way to transform the error bound (i.e. one
        # *variant* of the error bound). Additionally, each variant needs to contain
        # information about the error bounds for all variables.
        variable_names = set(error_bounds.keys())
        result: list[VariantErrorBoundPerVariable] = []
        for variant in variant_names:
            eb_per_variable: dict[VariableName, ErrorBound] = dict()
            for variable in variable_names:
                if variable in converted_bounds:
                    eb_per_variable[variable] = converted_bounds[variable][variant]
                else:
                    eb_per_variable[variable] = error_bounds[variable]
            result.append(
                VariantErrorBoundPerVariable(name=variant, error_bounds=eb_per_variable)
            )

        return result


def convert_error_bound(
    name: str,
    data_abs_min: float,
    data_abs_max: float,
    error_bound: ErrorBound,
) -> dict[VariantName, ErrorBound]:
    if error_bound.abs_error is not None:
        new_ebs = convert_abs_error_to_rel_error(name, data_abs_max, error_bound)
    else:
        new_ebs = convert_rel_error_to_abs_error(name, data_abs_min, error_bound)

    # Keep the old name for all the new error bounds. This ensures we can group
    # together all transformed error bounds that came from the same original bound.
    for n in new_ebs.keys():
        new_ebs[n].name = error_bound.name

    return new_ebs


def convert_rel_error_to_abs_error(
    name: str, data_abs_min: float, old_error: ErrorBound
) -> dict[VariantName, ErrorBound]:
    # In general, rel_error = abs_error / abs(data). This transformation
    # gives us the relative error bound that ensures the absolute error bound is
    # not exceeded for this dataset.
    assert old_error.rel_error is not None, "Expected relative error to be set."

    new_name = f"{name}-conservative-abs"
    error_bound = ErrorBound(abs_error=old_error.rel_error * data_abs_min)
    return {new_name: error_bound}


def convert_abs_error_to_rel_error(
    name: str, data_abs_max: float, old_error: ErrorBound
) -> dict[VariantName, ErrorBound]:
    # Same reasoning for error bound transformation as in `convert_rel_error_to_abs_error`.
    assert old_error.abs_error is not None, "Expected absolute error to be set."

    new_name = f"{name}-conservative-rel"
    error_bound = ErrorBound(rel_error=old_error.abs_error / data_abs_max)
    return {new_name: error_bound}
