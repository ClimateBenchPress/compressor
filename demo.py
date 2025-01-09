import json

from collections.abc import Mapping, Sequence
from pathlib import Path

import fcbench
import xarray as xr

from dask.diagnostics.progress import ProgressBar


fcbench.codecs.preload()


def _convert_to_json_serializable(o):
    if isinstance(o, Mapping):
        return {
            _convert_to_json_serializable(k): _convert_to_json_serializable(v)
            for k, v in o.items()
        }
    if isinstance(o, Sequence) and not isinstance(o, str):
        return [_convert_to_json_serializable(e) for e in o]
    return o


datasets = Path("..") / "data-loader" / "datasets"
compressors = Path("compressors")
compressed_datasets = Path("compressed-datasets")

for dataset in datasets.iterdir():
    if dataset.name == ".gitignore":
        continue

    dataset /= "standardized.zarr"

    for compressor in compressors.iterdir():
        compressed_dataset = compressed_datasets / dataset.parent.name / compressor.stem
        compressed_dataset.mkdir(parents=True, exist_ok=True)

        compressed_dataset_path = compressed_dataset / "decompressed.zarr"

        if compressed_dataset_path.exists():
            continue

        compressor = fcbench.compressor.Compressor.from_config_file(compressor)
        compressor = list(compressor.concrete)
        assert (
            len(compressor) == 1
        ), "only non-paramteric compressors are supported for now"
        compressor = compressor[0].build()

        ds = xr.open_dataset(dataset, chunks=dict())

        measurements = []

        ds_new = {
            v: fcbench.compressor.compress_decompress(
                da, compressor, measurements=measurements
            )
            for v, da in ds.items()
        }
        ds_new = xr.Dataset(ds_new, coords=ds.coords, attrs=ds.attrs)

        with (compressed_dataset / "measurements.json").open("w") as f:
            json.dump(_convert_to_json_serializable(measurements), f)

        with ProgressBar():
            ds_new.to_zarr(
                compressed_dataset_path, encoding=dict(), compute=False
            ).compute()
