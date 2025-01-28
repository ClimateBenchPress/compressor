import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import fcbench  # type: ignore
import xarray as xr
from dask.diagnostics.progress import ProgressBar


def _convert_to_json_serializable(o):
    if isinstance(o, Mapping):
        return {
            _convert_to_json_serializable(k): _convert_to_json_serializable(v)
            for k, v in o.items()
        }
    if isinstance(o, Sequence) and not isinstance(o, str):
        return [_convert_to_json_serializable(e) for e in o]
    return o


def compress_decompress(
    compressor: fcbench.compressor.Compressor, ds: xr.Dataset
) -> tuple[xr.Dataset, dict]:
    variables = dict()
    measurements = dict()
    for v in ds:
        v_measurements: list = []
        variables[v] = fcbench.compressor.compress_decompress(
            ds[v], compressor, measurements=v_measurements
        )
        measurements[v] = _convert_to_json_serializable(v_measurements)
    return xr.Dataset(variables, coords=ds.coords, attrs=ds.attrs), measurements


parser = argparse.ArgumentParser()
parser.add_argument("--exclude-dataset", type=str, nargs="+", default=[])
args = parser.parse_args()

repo = Path(__file__).parent.parent

datasets = repo.parent / "data-loader" / "datasets"
compressors = repo / "compressors"
compressed_datasets = repo / "compressed-datasets"

for dataset in datasets.iterdir():
    if dataset.name == ".gitignore" or dataset.name in args.exclude_dataset:
        continue

    dataset /= "standardized.zarr"

    for compressor_config in compressors.iterdir():
        compressed_dataset = (
            compressed_datasets / dataset.parent.name / compressor_config.stem
        )
        compressed_dataset.mkdir(parents=True, exist_ok=True)

        compressed_dataset_path = compressed_dataset / "decompressed.zarr"

        if compressed_dataset_path.exists():
            continue

        print(f"Compressing {dataset.parent.name} with {compressor_config.stem}...")

        compressor = fcbench.compressor.Compressor.from_config_file(compressor_config)
        compressor = list(compressor.concrete)
        assert len(compressor) == 1, (
            "only non-parametric compressors are supported for now"
        )
        compressor = compressor[0].build()

        ds = xr.open_dataset(dataset, chunks=dict(), engine="zarr")
        ds_new, measurements = compress_decompress(compressor, ds)

        with (compressed_dataset / "measurements.json").open("w") as f:
            json.dump(measurements, f)

        with ProgressBar():
            ds_new.to_zarr(
                compressed_dataset_path, encoding=dict(), compute=False
            ).compute()
