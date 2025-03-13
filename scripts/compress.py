import argparse
import json
from pathlib import Path

import numcodecs_observers
import xarray as xr
from climatebenchpress.compressor.compressors.abc import Compressor
from dask.diagnostics.progress import ProgressBar
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.hash import HashableCodec
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver

REPO = Path(__file__).parent.parent


def main(exclude_dataset, include_dataset, exclude_compressor, include_compressor):
    datasets = REPO.parent / "data-loader" / "datasets"
    compressed_datasets = REPO / "compressed-datasets"

    for dataset in datasets.iterdir():
        if dataset.name == ".gitignore" or dataset.name in exclude_dataset:
            continue
        if include_dataset and dataset.name not in include_dataset:
            continue

        dataset /= "standardized.zarr"

        for compressor in Compressor.registry.values():
            if compressor.name in exclude_compressor:
                continue
            if include_compressor and compressor.name not in include_compressor:
                continue

            compressed_dataset = (
                compressed_datasets / dataset.parent.name / compressor.name
            )
            compressed_dataset.mkdir(parents=True, exist_ok=True)

            compressed_dataset_path = compressed_dataset / "decompressed.zarr"

            if compressed_dataset_path.exists():
                continue

            print(
                f"Compressing {dataset.parent.name} with {compressor.description} ..."
            )

            ds = xr.open_dataset(dataset, chunks=dict(), engine="zarr")
            ds_new, measurements = compress_decompress(compressor.build(), ds)

            with (compressed_dataset / "measurements.json").open("w") as f:
                json.dump(measurements, f)

            with ProgressBar():
                ds_new.to_zarr(
                    compressed_dataset_path, encoding=dict(), compute=False
                ).compute()


def compress_decompress(codec: Codec, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
    if not isinstance(codec, CodecStack):
        codec = CodecStack(codec)

    variables = dict()
    measurements = dict()

    for v in ds:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dataset", type=str, nargs="+", default=[])
    parser.add_argument("--include-dataset", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-compressor", type=str, nargs="+", default=[])
    parser.add_argument("--include-compressor", type=str, nargs="+", default=None)
    args = parser.parse_args()
    main(**vars(args))
