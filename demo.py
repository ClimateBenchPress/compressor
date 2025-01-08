import json

from collections.abc import Mapping, Sequence
from pathlib import Path

import fcbench
import xarray as xr

from dask.diagnostics import ProgressBar


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


compressor = [
    fcbench.codecs.Sz3(eb_mode="abs", eb_abs=0.01),
]


decompressed = Path("decompressed.zarr")
if not decompressed.exists():
    ds = xr.open_dataset("../data-loader/data/cmip6/standardized.zarr", chunks=dict())

    measurements = []

    ds_new = {
        v: fcbench.compressor.compress_decompress(
            da, compressor, measurements=measurements
        )
        for v, da in ds.items()
    }
    ds_new = xr.Dataset(ds_new, coords=ds.coords, attrs=ds.attrs)

    with open("measurements.json", "w") as f:
        json.dump(_convert_to_json_serializable(measurements), f)

    with ProgressBar():
        ds_new.to_zarr(decompressed, encoding=dict(), compute=False).compute()
