import argparse
import json
from pathlib import Path

import numcodecs_observers
import xarray as xr
from climatebenchpress.compressor.compressors.abc import Compressor
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.hash import HashableCodec
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver
from dask.diagnostics.progress import ProgressBar


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


parser = argparse.ArgumentParser()
parser.add_argument("--exclude-dataset", type=str, nargs="+", default=[])
args = parser.parse_args()

repo = Path(__file__).parent.parent

datasets = repo.parent / "data-loader" / "datasets"
compressed_datasets = repo / "compressed-datasets"

for dataset in datasets.iterdir():
    if dataset.name == ".gitignore" or dataset.name in args.exclude_dataset:
        continue

    dataset /= "standardized.zarr"

    for compressor in Compressor.registry.values():
        compressed_dataset = compressed_datasets / dataset.parent.name / compressor.name
        compressed_dataset.mkdir(parents=True, exist_ok=True)

        compressed_dataset_path = compressed_dataset / "decompressed.zarr"

        if compressed_dataset_path.exists():
            continue

        print(f"Compressing {dataset.parent.name} with {compressor.description} ...")

        ds = xr.open_dataset(dataset, chunks=dict(), engine="zarr")
        ds_new, measurements = compress_decompress(compressor.build(), ds)

        with (compressed_dataset / "measurements.json").open("w") as f:
            json.dump(measurements, f)

        with ProgressBar():
            ds_new.to_zarr(
                compressed_dataset_path, encoding=dict(), compute=False
            ).compute()
