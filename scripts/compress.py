import argparse
import json
from pathlib import Path
from typing import Optional

import numcodecs_observers
import xarray as xr
from climatebenchpress.compressor.compressors.abc import Compressor
from dask.diagnostics.progress import ProgressBar
from numcodecs_combinators.stack import CodecStack
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.hash import HashableCodec
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver

REPO = Path(__file__).parent.parent


def main(exclude_dataset, include_dataset, exclude_compressor, include_compressor):
    datasets = REPO.parent / "data-loader" / "datasets"
    compressed_datasets = REPO / "test-compressed-datasets"
    datasets_error_bounds = REPO / "datasets-error-bounds"

    for dataset in datasets.iterdir():
        if dataset.name == ".gitignore" or dataset.name in exclude_dataset:
            continue
        if include_dataset and dataset.name not in include_dataset:
            continue

        dataset /= "standardized.zarr"
        ds = xr.open_dataset(dataset, chunks=dict(), engine="zarr")
        ds_stats = dict()
        for v in ds:
            abs_vals = xr.ufuncs.abs(ds[v])
            ds_stats[v] = {
                "min": abs_vals.min().values.item(),
                "max": abs_vals.max().values.item(),
            }

        error_bounds = get_error_bounds(datasets_error_bounds, dataset.parent.name)
        for error_bound in error_bounds:
            error_bound_name = "=".join(
                f"{k}={v}" for k, v in error_bound.items() if v is not None
            )
            for compressor in Compressor.registry.values():
                if compressor.name in exclude_compressor:
                    continue
                if include_compressor and compressor.name not in include_compressor:
                    continue

                compressed_dataset = (
                    compressed_datasets
                    / dataset.parent.name
                    / error_bound_name
                    / compressor.name
                )
                compressed_dataset.mkdir(parents=True, exist_ok=True)

                compressed_dataset_path = compressed_dataset / "decompressed.zarr"

                if compressed_dataset_path.exists():
                    continue

                print(
                    f"Compressing {dataset.parent.name} with {compressor.description} ..."
                )

                try:
                    ds_new, measurements = compress_decompress(
                        compressor, ds, ds_stats, **error_bound
                    )
                except Exception as e:
                    print(
                        f"Error compressing {dataset.parent.name} with {compressor.name}: {e}"
                    )
                    print("Skipping...")
                    continue

                with (compressed_dataset / "measurements.json").open("w") as f:
                    json.dump(measurements, f)

                with ProgressBar():
                    ds_new.to_zarr(
                        compressed_dataset_path, encoding=dict(), compute=False
                    ).compute()


def compress_decompress(
    compressor: Compressor,
    ds: xr.Dataset,
    ds_stats: dict,
    abs_error: Optional[float] = None,
    rel_error: Optional[float] = None,
) -> tuple[xr.Dataset, dict]:
    variables = dict()
    measurements = dict()

    for v in ds:
        data_min, data_max = ds_stats[v]["min"], ds_stats[v]["max"]
        codec = compressor.build(data_min, data_max, abs_error, rel_error)
        if not isinstance(codec, CodecStack):
            codec = CodecStack(codec)

        nbytes = BytesizeObserver()
        timing = WalltimeObserver()
        instructions = WasmCodecInstructionCounterObserver()

        with numcodecs_observers.observe(
            codec,
            observers=[
                nbytes,
                instructions,
                timing,
            ],
        ) as codec_:
            variables[v] = codec_.encode_decode_data_array(ds[v]).compute()

        measurements[v] = {
            "encoded_bytes": sum(
                b.post for b in nbytes.encode_sizes[HashableCodec(codec[-1])]
            ),
            "decoded_bytes": sum(
                b.post for b in nbytes.decode_sizes[HashableCodec(codec[0])]
            ),
            "encode_timing": sum(t for ts in timing.encode_times.values() for t in ts),
            "decode_timing": sum(t for ts in timing.decode_times.values() for t in ts),
            "encode_instructions": sum(
                i for is_ in instructions.encode_instructions.values() for i in is_
            )
            or None,
            "decode_instructions": sum(
                i for is_ in instructions.decode_instructions.values() for i in is_
            )
            or None,
        }

    return xr.Dataset(variables, coords=ds.coords, attrs=ds.attrs), measurements


def get_error_bounds(datasets_error_bounds, dataset_name):
    dataset_error_bounds = datasets_error_bounds / dataset_name
    with open(dataset_error_bounds / "error_bounds.json") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dataset", type=str, nargs="+", default=[])
    parser.add_argument("--include-dataset", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-compressor", type=str, nargs="+", default=[])
    parser.add_argument("--include-compressor", type=str, nargs="+", default=None)
    args = parser.parse_args()
    main(**vars(args))
