__all__ = ["create_error_bounds"]

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import xarray as xr
from compression_recommendations import Recommendations
from compression_recommendations.filters.cf import CfShortNameFilter
from compression_recommendations.filters.combinators import AnyFilter
from compression_recommendations.filters.tag import TagFilter
from compression_recommendations.recommendation import Recommendation
from compression_recommendations.requirements.abc import Requirement
from compression_recommendations.requirements.error_bounds.max import (
    MaxPointwiseAbsoluteErrorBoundRequirement,
    MaxPointwiseRelativeErrorBoundRequirement,
)
from semver.version import Version

# Location of the error bounds derived from ERA5 ensemble uncertainty
ERROR_BOUNDS = Path(
    "/Users/junityre/era5-ensemble/recommendations/ensemble-spread.yaml"
)

# Bitwise real information for no2 data computed from the CAMS dataset.
# The numbers are taken from the supplementary material of:
# https://www.nature.com/articles/s43588-021-00156-2
# "Compressing atmospheric data into its real information content"
# Milan Klöwer, Miha Razinger, Juan J. Dominguez, Peter D. Düben & Tim N. Palmer
# Exact data was extracted from the CSV file at:
# https://static-content.springer.com/esm/art%3A10.1038%2Fs43588-021-00156-2/MediaObjects/43588_2021_156_MOESM2_ESM.zip
# Accessed on: 29/08/2025.
NO2_REAL_INFORMATION = [
    0.0,
    0.0,
    0.0,
    0.82609236,
    0.82609236,
    0.8401368,
    0.7781777,
    0.65409344,
    0.47173682,
    0.2915112,
    0.16627097,
    0.08410827,
    0.03733299,
    0.015123328,
    0.0058184885,
    0.0020212175,
    0.00053772517,
    9.423521e-5,
    8.782874e-6,
    5.8530753e-7,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]


VAR_NAME_TO_ERA5 = {
    # NextGEMS Icon Outgoing Longwave Radiation (OLR).
    # Closest ERA5 equivalent Mean flux top net long-wave radiation
    # (https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf).
    # which is the negative of OLR.
    # NOTE: Be careful in using the flux instead of the time-accumulated variables.
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/235040
    # ERA5 unit: W m-2
    # NextGEMS unit: W m-2
    "rlut": "avg_tnlwrf",
    # NextGEMS Icon Precipitation
    # NOTE: Be careful in using the flux instead of the time-accumulated variables.
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/235055
    # ERA5 unit: kg m-2 s-1
    # NextGEMS unit: kg m-2 s-1
    "pr": "avg_tprate",
    # Air temperature.
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/130
    # ERA5 unit: K
    # CMIP6 unit: K
    "ta": "t",
    # Sea surface temperature.
    # NOTE: Difference in units means we should use absolute error bounds.
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/34
    # ERA5 unit: K
    # CMIP6 unit: degC
    "tos": "sst",
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/165
    # Units will match because data source is ERA5.
    "10m_u_component_of_wind": "u10",
    "10u": "u10",
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/166
    # Units will match because data source is ERA5.
    "10m_v_component_of_wind": "v10",
    "10v": "v10",
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/151
    # Units will match because data source is ERA5.
    "mean_sea_level_pressure": "msl",
    "msl": "msl",
    # Humidity
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/133
    # Units will match because data source is ERA5.
    "q": "q",
}


ABS_ERROR = "abs_error"
REL_ERROR = "rel_error"
VAR_NAME_TO_ERROR_BOUND = {
    "rlut": ABS_ERROR,
    "agb": REL_ERROR,
    "pr": REL_ERROR,
    "ta": ABS_ERROR,
    "tos": ABS_ERROR,
    "10m_u_component_of_wind": ABS_ERROR,
    "10u": ABS_ERROR,
    "10m_v_component_of_wind": ABS_ERROR,
    "10v": ABS_ERROR,
    "mean_sea_level_pressure": ABS_ERROR,
    "msl": ABS_ERROR,
    "no2": REL_ERROR,
    "q": REL_ERROR,
}


def create_error_bounds(
    basepath: Path = Path(),
    data_loader_basepath: None | Path = None,
):
    """Create three error bounds for all datasets and the variables in them.

    Parameters
    ----------
    basepath : Path
        The base path where the error bounds will be stored.
        The error bounds will be stored in `basepath / datasets-error-bounds`.
    data_loader_basepath : Path, optional
        The base path where the datasets are stored. If not provided, it defaults to `basepath / .. / data-loader`.
        The datasets will be loaded from `data_loader_basepath / datasets`.
    """
    datasets = (data_loader_basepath or basepath) / "datasets"
    datasets_error_bounds = basepath / "datasets-error-bounds"

    with ERROR_BOUNDS.open() as bounds:
        recommendations = Recommendations.load(bounds)

    for dataset in datasets.iterdir():
        if dataset.name == ".gitignore":
            continue

        if not (dataset / "standardized.zarr").exists():
            print(f"No input dataset at {dataset / 'standardized.zarr'}")
            continue

        print(dataset.name)
        ds = xr.open_dataset(
            dataset / "standardized.zarr",
            chunks=dict(),
            engine="zarr",
            decode_times=False,
        )

        low_error_bounds: dict[str, dict[str, float | None]] = dict()
        mid_error_bounds: dict[str, dict[str, float | None]] = dict()
        high_error_bounds: dict[str, dict[str, float | None]] = dict()

        for v in ds:
            v = str(v)
            if v in VAR_NAME_TO_ERA5:
                low_error_bounds[v], mid_error_bounds[v], high_error_bounds[v] = (
                    get_error_bounds(
                        recommendations,
                        VAR_NAME_TO_ERA5[str(v)],
                        VAR_NAME_TO_ERROR_BOUND[str(v)],
                        percentiles=[0.0, 0.01, 0.05],
                    )
                )
            elif v == "agb":
                low_error_bounds[v], mid_error_bounds[v], high_error_bounds[v] = (
                    get_error_bounds(
                        Recommendations(
                            recommendations=compute_agb_recommendations(
                                datasets, percentiles=[0.0, 0.01, 0.05]
                            ),
                            version=Version.parse("1.0.0"),
                            metadata={},
                        ),
                        v,
                        VAR_NAME_TO_ERROR_BOUND[str(v)],
                        percentiles=[0.0, 0.01, 0.05],
                        round_bounds=False,
                    )
                )
            elif v == "no2":
                low_error_bounds[v], mid_error_bounds[v], high_error_bounds[v] = (
                    get_error_bounds(
                        Recommendations(
                            recommendations=compute_no2_recommendations(
                                percentiles=[1.00, 0.99, 0.95]
                            ),
                            version=Version.parse("1.0.0"),
                            metadata={},
                        ),
                        v,
                        VAR_NAME_TO_ERROR_BOUND[str(v)],
                        percentiles=[1.0, 0.99, 0.95],
                        round_bounds=False,
                    )
                )
            else:
                data_range: float = (ds[v].max() - ds[v].min()).values.item()  # type: ignore
                low_error_bounds[v] = {
                    ABS_ERROR: 0.0001 * data_range,
                    REL_ERROR: None,
                }
                mid_error_bounds[v] = {
                    ABS_ERROR: 0.001 * data_range,
                    REL_ERROR: None,
                }
                high_error_bounds[v] = {
                    ABS_ERROR: 0.01 * data_range,
                    REL_ERROR: None,
                }

        error_bounds = [low_error_bounds, mid_error_bounds, high_error_bounds]

        dataset_error_bounds = datasets_error_bounds / dataset.name
        dataset_error_bounds.mkdir(parents=True, exist_ok=True)
        with (dataset_error_bounds / "error_bounds.json").open("w") as f:
            json.dump(error_bounds, f)


def get_error_bounds(
    recommendations: Recommendations,
    era5_var: str,
    error_bound_type: str,
    percentiles: list[float] = [0.0, 0.01, 0.05],
    pressure_levels: list[float] = [50.0, 500.0, 850.0, 1000.0],
    round_bounds: bool = True,
) -> list[dict[str, None | float]]:
    error_bound_tag = {
        ABS_ERROR: "absolute",
        REL_ERROR: "relative",
    }[error_bound_type]

    var_ebs: list[dict[str, None | float]] = []

    requirements: list[Requirement]
    for percentile in percentiles:
        try:
            requirements = [
                req
                for pressure in pressure_levels
                for req in recommendations.search(
                    markers={
                        "cf-short-name": era5_var,
                        "grib-short-name": era5_var,
                        "level-kind": "pressure",
                        "level-value": pressure,
                        "tags": f"{percentile * 100}%,{error_bound_tag}",
                    }
                )
            ]
            assert len(requirements) == len(pressure_levels)
        except KeyError:
            requirements = list(
                recommendations.search(
                    markers={
                        "cf-short-name": era5_var,
                        "grib-short-name": era5_var,
                        "level-kind": "single",
                        "tags": f"{percentile * 100}%,{error_bound_tag}",
                    }
                )
            )
            assert len(requirements) == 1

        bounds: list[int | float] = []
        for requirement in requirements:
            assert isinstance(
                requirement,
                MaxPointwiseAbsoluteErrorBoundRequirement
                | MaxPointwiseRelativeErrorBoundRequirement,
            )
            assert isinstance(
                requirement,
                {
                    ABS_ERROR: MaxPointwiseAbsoluteErrorBoundRequirement,
                    REL_ERROR: MaxPointwiseRelativeErrorBoundRequirement,
                }[error_bound_type],
            )
            bound = requirement.value
            # Explicitly round the bounds as done in ClimateBenchPress v1.0.0
            bounds.append(
                {
                    ABS_ERROR: float(f"{bound:.1e}"),
                    REL_ERROR: float(f"{bound:.2%}".rstrip("%")) / 100,
                }[error_bound_type]
                if round_bounds
                else bound
            )

        # For variables with multiple levels, only air temperature at this
        # point, we take the average error bound across all levels.
        # Compute the correctly-rounded order-independent mean
        bound = float(sum(Decimal(b) for b in bounds) / len(bounds))

        var_ebs.append(
            {
                ABS_ERROR: {error_bound_type: bound}.get(ABS_ERROR),
                REL_ERROR: {error_bound_type: bound}.get(REL_ERROR),
            }
        )

    return var_ebs


def compute_no2_recommendations(
    percentiles=[1.00, 0.99, 0.95],
) -> list[Recommendation]:
    # First we need to transform the bitwise real information into a cumulative
    # distribution function.
    # We need to be careful that np.cumsum(x)[-1] may be unequal np.sum(x).
    real_information_cumsum = np.cumsum(NO2_REAL_INFORMATION)
    real_information_dist = real_information_cumsum / real_information_cumsum[-1]
    base_filters = [AnyFilter(filters=[CfShortNameFilter(value="no2")])]
    recommendations = []
    for p in percentiles:
        filters = base_filters + [TagFilter(value=f"{p * 100}%")]
        # Find the first position where cumulative distribution is >= p.
        # Add one for 1-based indexing.
        keepbits = np.searchsorted(real_information_dist, p) + 1
        # There are 9 non-mantissa bits in the 32-bit floating point representation.
        mantissa_keepbits = max(int(keepbits - 9), 0)
        # 2^(-mantissa_keepbits) indicates the spacing between representable values.
        # 2^(-mantissa_keepbits) / 2 = 2 ** (-mantissa_keepbits - 1)
        # then is the maximum rounding error.
        # See: https://en.wikipedia.org/wiki/Machine_epsilon
        rel_error = 2 ** (-mantissa_keepbits - 1)
        recommendations.append(
            Recommendation(
                filters=filters,
                requirements=[
                    MaxPointwiseRelativeErrorBoundRequirement(value=rel_error)
                ],
            )
        )
    return recommendations


def compute_agb_recommendations(
    datasets: Path, percentiles=[0.0, 0.01, 0.05]
) -> list[Recommendation]:
    # Define rough bounding box coordinates for mainland France.
    # Format: [min_longitude, min_latitude, max_longitude, max_latitude].
    FRANCE_BBOX = [-5.5, 42.3, 9.6, 51.1]

    agb = xr.open_dataset(
        datasets
        / "esa-biomass-cci"
        / "download"
        / "ESACCI-BIOMASS-L4-AGB-MERGED-100m-2020-fv5.01.nc"
    )
    agb = agb.sel(
        lon=slice(FRANCE_BBOX[0], FRANCE_BBOX[2]),
        lat=slice(FRANCE_BBOX[3], FRANCE_BBOX[1]),
    )

    return compute_ensemble_spread_recommendations(
        mean=agb.agb, spread=agb.agb_sd, percentiles=percentiles
    )


def compute_ensemble_spread_recommendations(
    mean: xr.DataArray, spread: xr.DataArray, percentiles: list[float]
) -> list[Recommendation]:
    da = mean

    mean_values = mean.values.flatten()
    spread_values = spread.values.flatten()

    spread_nonzero = spread_values[(spread_values > 0.0) & np.isfinite(spread_values)]

    # compute the absolute error bound
    absolute: list[float]
    if len(spread_nonzero) > 0:
        absolute = [
            float(s) for s in np.nanquantile(spread_nonzero, [p for p in percentiles])
        ]
    else:
        absolute = [0.0 for _ in percentiles]

    # compute the relative error bound
    abs_mean = np.abs(mean_values)
    rel = spread_values[abs_mean > 0.0] / abs_mean[abs_mean > 0.0]
    rel_nonzero = rel[(rel > 0.0) & np.isfinite(rel)]

    relative: list[float]
    if len(rel_nonzero) > 0:
        relative = [
            float(s) for s in np.nanquantile(rel_nonzero, [p for p in percentiles])
        ]
    else:
        relative = [0.0 for _ in percentiles]

    base_filters = [
        AnyFilter(filters=[CfShortNameFilter(value=str(da.name))]),
    ]

    recommendations = []

    # compile the recommendations for each percentile and error bound kind
    for i, p in enumerate(percentiles):
        filters = base_filters + [TagFilter(value=f"{p * 100}%")]

        recommendations.append(
            Recommendation(
                filters=filters + [AnyFilter(filters=[TagFilter(value="absolute")])],
                requirements=[
                    MaxPointwiseAbsoluteErrorBoundRequirement(value=float(absolute[i]))
                ],
            )
        )
        recommendations.append(
            Recommendation(
                filters=filters + [AnyFilter(filters=[TagFilter(value="relative")])],
                requirements=[
                    MaxPointwiseRelativeErrorBoundRequirement(value=float(relative[i]))
                ],
            )
        )

    return recommendations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create error bounds for datasets")
    parser.add_argument("--basepath", type=Path, default=Path())
    parser.add_argument(
        "--data-loader-basepath", type=Path, default=Path() / ".." / "data-loader"
    )
    args = parser.parse_args()

    create_error_bounds(
        basepath=args.basepath,
        data_loader_basepath=args.data_loader_basepath,
    )
