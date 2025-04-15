__all__ = ["create_error_bounds"]

import json
from pathlib import Path

import xarray as xr


def create_error_bounds(
    basepath: Path = Path(),
    data_loader_base_path: None | Path = None,
):
    datasets = (data_loader_base_path or basepath) / "datasets"
    datasets_error_bounds = basepath / "datasets-error-bounds"

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
            data_range: float = (ds[v].max() - ds[v].min()).values.item()  # type: ignore
            low_error_bounds[v] = {"abs_error": 0.0001 * data_range, "rel_error": None}
            mid_error_bounds[v] = {"abs_error": 0.001 * data_range, "rel_error": None}
            high_error_bounds[v] = {"abs_error": 0.01 * data_range, "rel_error": None}

        error_bounds = [low_error_bounds, mid_error_bounds, high_error_bounds]

        dataset_error_bounds = datasets_error_bounds / dataset.name
        dataset_error_bounds.mkdir(parents=True, exist_ok=True)
        with (dataset_error_bounds / "error_bounds.json").open("w") as f:
            json.dump(error_bounds, f)


if __name__ == "__main__":
    create_error_bounds(
        basepath=Path(),
        data_loader_base_path=Path() / ".." / "data-loader",
    )
