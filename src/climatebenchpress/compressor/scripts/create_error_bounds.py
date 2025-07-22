__all__ = ["create_error_bounds"]

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# Table has header:
# var,level,percentile,min,range,max,lpbits,lpabsolute,lprelative,brlqabsolute,brabsolute,brrelative,esabsolute,esrelative,esquadratic,unabsolute,cabsolute,crelative,cquadratic,pick,crlinquant,crbitround,crlinquantquadstep,exabsmean,exabsmax,exrelmean,exrelmax
#
# esabsolute and esrelative are respectively the absolute and relative error bounds
# derived from the ERA5 ensembles.
ERROR_BOUNDS = "https://gist.githubusercontent.com/juntyr/bbe2780256e5f91d8f2cb2f606b7935f/raw/table-raw.csv"


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
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/166
    # Units will match because data source is ERA5.
    "10m_v_component_of_wind": "v10",
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/151
    # Units will match because data source is ERA5.
    "mean_sea_level_pressure": "msl",
}


ABS_ERROR = "abs_error"
REL_ERROR = "rel_error"
VAR_NAME_TO_ERROR_BOUND = {
    "rlut": REL_ERROR,
    "agb": REL_ERROR,
    "pr": REL_ERROR,
    "ta": ABS_ERROR,
    "tos": ABS_ERROR,
    "10m_u_component_of_wind": REL_ERROR,
    "10m_v_component_of_wind": REL_ERROR,
    "mean_sea_level_pressure": ABS_ERROR,
    "no2": ABS_ERROR,
}


def create_error_bounds(
    basepath: Path = Path(),
    data_loader_basepath: None | Path = None,
):
    datasets = (data_loader_basepath or basepath) / "datasets"
    datasets_error_bounds = basepath / "datasets-error-bounds"

    era5_error_bounds = pd.read_csv(ERROR_BOUNDS)

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

        low_error_bounds, mid_error_bounds, high_error_bounds = dict(), dict(), dict()
        for v in ds:
            if v in VAR_NAME_TO_ERA5:
                low_error_bounds[v], mid_error_bounds[v], high_error_bounds[v] = (
                    get_error_bounds(
                        era5_error_bounds,
                        VAR_NAME_TO_ERA5[str(v)],
                        VAR_NAME_TO_ERROR_BOUND[str(v)],
                    )
                )
            elif v == "agb":
                low_error_bounds[v], mid_error_bounds[v], high_error_bounds[v] = (
                    get_agb_bound(datasets, percentiles=[1.00, 0.99, 0.95])
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
    error_bounds: pd.DataFrame, era5_var: str, error_bound_type: str
) -> list[dict[str, Optional[float]]]:
    var_error_bounds = error_bounds[error_bounds["var"] == era5_var].copy()
    single_level = var_error_bounds["level"].unique()[0] == "single"
    if single_level:
        assert len(var_error_bounds) == 3, (
            "Expected three error bounds for each variable."
        )
    else:
        # For variables with multiple levels (only air temperature at this point)
        # take the average error bound across all levels.
        var_error_bounds["esrelative_float"] = (
            var_error_bounds["esrelative"].str.rstrip("%").astype(float)
        )
        grouped = (
            var_error_bounds.groupby(["percentile"])
            .agg({"esabsolute": "mean", "esrelative_float": "mean"})
            .reset_index()
        )
        grouped["esrelative"] = grouped["esrelative_float"].astype(str) + "%"
        var_error_bounds = grouped[["percentile", "esabsolute", "esrelative"]]

    # Ordered from strictest to most relaxed error bounds.
    percentiles = ["100%", "99%", "95%"]
    var_ebs = []
    for percentile in percentiles:
        eb_row = var_error_bounds[var_error_bounds["percentile"] == percentile]

        if error_bound_type == REL_ERROR:
            # Relative error bounds are given as a percentage with an "%" at the end,
            # so we need to convert them to a fraction.
            rel_error = float(eb_row["esrelative"].item()[:-1]) / 100.0
            var_ebs.append({ABS_ERROR: None, REL_ERROR: rel_error})
        elif error_bound_type == ABS_ERROR:
            abs_error = float(eb_row["esabsolute"].item())
            var_ebs.append({ABS_ERROR: abs_error, REL_ERROR: None})
        else:
            raise ValueError(f"Unknown error bound type: {error_bound_type}")

    return var_ebs


def get_agb_bound(
    datasets: Path, percentiles=[1.00, 0.99, 0.95]
) -> list[dict[str, Optional[float]]]:
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

    ensemble_bounds = compute_ensemble_spread_bounds(
        mean=agb.agb, spread=agb.agb_sd, percentile=percentiles
    )

    error_bounds = []
    for a, r in zip(ensemble_bounds.absolute, ensemble_bounds.relative):
        if VAR_NAME_TO_ERROR_BOUND["agb"] == ABS_ERROR:
            error_bounds.append(
                {
                    "abs_error": float(a),
                    "rel_error": None,
                }
            )
        elif VAR_NAME_TO_ERROR_BOUND["agb"] == REL_ERROR:
            error_bounds.append(
                {
                    "abs_error": None,
                    "rel_error": float(r),
                }
            )

    return error_bounds


@dataclass
class EnsembleSpreadBounds:
    percentile: list[float]
    absolute: list[float]
    relative: list[float]


def compute_ensemble_spread_bounds(
    mean: xr.DataArray, spread: xr.DataArray, percentile: list[float]
) -> EnsembleSpreadBounds:
    mean_values = mean.values.flatten()
    spread_values = spread.values.flatten()

    spread_nonzero = spread_values[spread_values > 0.0]

    if len(spread_nonzero) > 0:
        absolute = np.nanquantile(spread_nonzero, [1 - p for p in percentile])
    else:
        absolute = [0.0 for _ in percentile]

    abs_mean = np.abs(mean_values)
    rel = spread_values[abs_mean > 0.0] / abs_mean[abs_mean > 0.0]
    rel_nonzero = rel[rel > 0.0]

    if len(rel_nonzero) > 0:
        relative = np.nanquantile(rel_nonzero, [1 - p for p in percentile])
    else:
        relative = [0.0 for _ in percentile]

    return EnsembleSpreadBounds(
        percentile=percentile,
        absolute=absolute,
        relative=relative,
    )


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
