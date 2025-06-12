__all__ = ["create_error_bounds"]

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# Table has header:
# var,level,percentile,min,range,max,lpbits,lpabsolute,lprelative,brlqabsolute,brabsolute,brrelative,esabsolute,esrelative,esquadratic,unabsolute,cabsolute,crelative,cquadratic,pick,crlinquant,crbitround,crlinquantquadstep
#
# esabsolute and esrelative are respectively the absolute and relative error bounds
# derived from the ERA5 ensembles.
ERROR_BOUNDS = "https://raw.githubusercontent.com/juntyr/era5-ensemble/refs/heads/main/table-raw.csv?token=GHSAT0AAAAAACTGGFLKSCEPFNNUEGWSWPEA2CJOYSQ"


VAR_NAME_TO_ERA5 = {
    # NextGEMS Icon Outgoing Longwave Radiation (OLR).
    # Closest ERA5 equivalent Top net long-wave (thermal) radiation
    # (https://www.ecmwf.int/sites/default/files/elibrary/2015/18490-radiation-quantities-ecmwf-model-and-mars.pdf).
    # which is the negative of OLR.
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/179
    "rlut": "ttr",
    # NextGEMS Icon Precipitation
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/228
    "pr": "tp",
    # Air temperature.
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/130
    # The CMIP6 data contains temperature data for multiple pressure levels,
    # we use the 2m ERA5 temperature data to derive the error bound for all
    # pressure levels.
    "ta": "t2m",
    # Sea surface temperature.
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/34
    "tos": "sst",
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/165
    "10m_u_component_of_wind": "u10",
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/166
    "10m_v_component_of_wind": "v10",
    # ERA5 documentation: https://codes.ecmwf.int/grib/param-db/151
    "mean_sea_level_pressure": "msl",
}


def create_error_bounds(
    basepath: Path = Path(),
    data_loader_base_path: None | Path = None,
):
    datasets = (data_loader_base_path or basepath) / "datasets"
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

        # TODO: This is a temporary solution that should be replaced by a more
        #       principled method to selct the error bounds.
        low_error_bounds, mid_error_bounds, high_error_bounds = dict(), dict(), dict()
        for v in ds:
            if v in VAR_NAME_TO_ERA5:
                low_error_bounds[v], mid_error_bounds[v], high_error_bounds[v] = (
                    get_error_bounds(era5_error_bounds, VAR_NAME_TO_ERA5[str(v)])
                )
            elif v == "agb":
                low_error_bounds[v], mid_error_bounds[v], high_error_bounds[v] = (
                    get_agb_bound(datasets, percentiles=[1.00, 0.99, 0.95])
                )
            else:
                data_range: float = (ds[v].max() - ds[v].min()).values.item()  # type: ignore
                low_error_bounds[v] = {
                    "abs_error": 0.0001 * data_range,
                    "rel_error": None,
                }
                mid_error_bounds[v] = {
                    "abs_error": 0.001 * data_range,
                    "rel_error": None,
                }
                high_error_bounds[v] = {
                    "abs_error": 0.01 * data_range,
                    "rel_error": None,
                }

        error_bounds = [low_error_bounds, mid_error_bounds, high_error_bounds]

        dataset_error_bounds = datasets_error_bounds / dataset.name
        dataset_error_bounds.mkdir(parents=True, exist_ok=True)
        with (dataset_error_bounds / "error_bounds.json").open("w") as f:
            json.dump(error_bounds, f)


def get_error_bounds(
    error_bounds: pd.DataFrame, era5_var: str
) -> list[dict[str, Optional[float]]]:
    var_error_bounds = error_bounds[error_bounds["var"] == era5_var]
    assert len(var_error_bounds) == 3, "Expected three error bounds for each variable."

    # Ordered from strictest to most relaxed error bounds.
    percentiles = ["100%", "99%", "95%"]
    var_ebs = []
    for percentile in percentiles:
        eb_row = var_error_bounds[var_error_bounds["percentile"] == percentile]
        eb_type = eb_row["pick"].item()

        if eb_type == "quadratic":
            # Right now no compressor supports quadratic error bounds. We therefore
            # fall back to absolute error bounds for them.
            eb_type = "absolute"

        if eb_type == "relative":
            # Relative error bounds are given as a percentage with an "%" at the end,
            # so we need to convert them to a fraction.
            rel_error = float(eb_row[f"es{eb_type}"].item()[:-1]) / 100.0
            var_ebs.append(
                {
                    "abs_error": None,
                    "rel_error": rel_error,
                }
            )
        elif eb_type == "absolute":
            abs_error = float(eb_row[f"es{eb_type}"].item())
            var_ebs.append(
                {
                    "abs_error": abs_error,
                    "rel_error": None,
                }
            )
        else:
            raise ValueError(f"Unknown error bound type: {eb_type}")

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

    minfo = compute_minimum_bound(
        mean=agb.agb,
        spread=agb.agb_sd,
        percentile=percentiles,
    )

    error_bounds = []
    for a, r, b in zip(
        ensemble_bounds.absolute, ensemble_bounds.relative, minfo.mean_bin_spread_bounds
    ):
        cabs = f"{np.nansum(np.abs(np.array(b) - a) * minfo.mean_bin_counts) / np.sum(minfo.mean_bin_counts):.1e}"
        crel = f"{np.nansum(np.abs(np.array(b) - np.abs((np.array(minfo.mean_bin_edges[:-1]) + np.array(minfo.mean_bin_edges[1:])) / 2) * r) * minfo.mean_bin_counts) / np.sum(minfo.mean_bin_counts):.1e}"

        bounds = [("absolute", cabs), ("relative", crel)]
        bound_pick = sorted(bounds, key=lambda x: float(x[1]))[0][0]

        if bound_pick == "absolute":
            error_bounds.append(
                {
                    "abs_error": float(a),
                    "rel_error": None,
                }
            )
        elif bound_pick == "relative":
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


@dataclass
class MinimumBounds:
    percentile: list[float]
    mean_bin_edges: list[float]
    mean_bin_counts: list[int]
    mean_bin_spread_bounds: list[list[float]]


def compute_minimum_bound(
    mean: xr.DataArray,
    spread: xr.DataArray,
    percentile: list[float],
    nbins: int = 100,
) -> MinimumBounds:
    mean_values = mean.copy(deep=True).values.flatten()
    spread_values = spread.copy(deep=True).values.flatten()

    mean_bin_edges = np.nanquantile(mean_values, np.linspace(0.0, 1.0, nbins + 1))

    ibin = np.minimum(np.searchsorted(mean_bin_edges, mean_values), nbins - 1)

    mean_bin_spread_bounds = [np.zeros(nbins) for _ in percentile]

    for i in range(nbins):
        ispread = spread_values[ibin == i]

        if len(ispread) > 0:
            bs = np.nanquantile(ispread, [1 - p for p in percentile])
        else:
            bs = [np.nan for _ in percentile]

        for bd, b in zip(mean_bin_spread_bounds, bs):
            bd[i] = b

    mean_bin_counts, _ = np.histogram(mean_values, bins=mean_bin_edges)

    return MinimumBounds(
        percentile=percentile,
        mean_bin_edges=list(mean_bin_edges),
        mean_bin_counts=list(mean_bin_counts),
        mean_bin_spread_bounds=[list(bs) for bs in mean_bin_spread_bounds],
    )


if __name__ == "__main__":
    create_error_bounds(
        basepath=Path(),
        data_loader_base_path=Path() / ".." / "data-loader",
    )
