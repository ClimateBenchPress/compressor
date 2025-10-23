__all__ = ["compress"]

import argparse
import json
import traceback
from collections.abc import Container, Mapping
from pathlib import Path
from typing import Callable

import numcodecs_observers
import numpy as np
import xarray as xr
from numcodecs.abc import Codec
from numcodecs_combinators.stack import CodecStack
from numcodecs_observers.bytesize import BytesizeObserver
from numcodecs_observers.hash import HashableCodec
from numcodecs_observers.walltime import WalltimeObserver
from numcodecs_wasm import WasmCodecInstructionCounterObserver

from ..compressors.abc import Compressor, ErrorBound, NamedPerVariableCodec
from ..monitor import progress_bar


def compress(
    basepath: Path = Path(),
    exclude_dataset: Container[str] = tuple(),
    include_dataset: None | Container[str] = None,
    exclude_compressor: Container[str] = tuple(),
    include_compressor: None | Container[str] = None,
    data_loader_basepath: None | Path = None,
    progress: bool = True,
):
    """Compress datasets with compressors.

    Parameters
    ----------
    basepath : Path
        Compressed dataset will be stored in `basepath / compressed-datasets`.
    exclude_dataset : Container[str]
        Datasets to exclude from compression.
    include_dataset : None | Container[str]
        Datasets to include in compression. If `None`, all datasets are included.
        If specified, only datasets in `include_dataset` will be compressed.
    exclude_compressor : Container[str]
        Compressors to exclude from compression.
    include_compressor : None | Container[str]
        Compressors to include in compression. If `None`, all compressors are included.
        If specified, only compressors in `include_compressor` will be used.
    data_loader_basepath : None | Path
        Base path for the data loader datasets. If `None`, defaults to `basepath / .. / data-loader`.
        Input datasets will be loaded from `data_loader_basepath / datasets`.
    progress : bool
        Whether to show a progress bar during compression.
    """
    datasets = (data_loader_basepath or basepath) / "datasets"
    compressed_datasets = basepath / "compressed-datasets"
    datasets_error_bounds = basepath / "datasets-error-bounds"

    for dataset in datasets.iterdir():
        if dataset.name == ".gitignore" or dataset.name in exclude_dataset:
            continue
        if include_dataset and dataset.name not in include_dataset:
            continue

        dataset /= "standardized.zarr"

        if not dataset.exists():
            print(f"No input dataset at {dataset}")
            continue

        ds = xr.open_dataset(dataset, chunks=dict(), engine="zarr")
        ds_dtypes: dict[str, np.dtype] = dict()
        ds_abs_mins: dict[str, float] = dict()
        ds_abs_maxs: dict[str, float] = dict()
        ds_mins: dict[str, float] = dict()
        ds_maxs: dict[str, float] = dict()
        for v in ds:
            vs: str = str(v)
            abs_vals = xr.ufuncs.abs(ds[v])
            ds_dtypes[vs] = ds[v].dtype
            # Take minimum of non-zero absolute values to avoid division by zero.
            ds_abs_mins[vs] = abs_vals.where(abs_vals > 0).min().values.item()
            ds_abs_maxs[vs] = abs_vals.max().values.item()
            ds_mins[vs] = ds[v].min().values.item()
            ds_maxs[vs] = ds[v].max().values.item()

        error_bounds = get_error_bounds(datasets_error_bounds, dataset.parent.name)
        registry: Mapping[str, type[Compressor]] = Compressor.registry  # type: ignore
        for compressor in registry.values():
            if compressor.name in exclude_compressor:
                continue
            if include_compressor and compressor.name not in include_compressor:
                continue

            compressor_variants: dict[str, list[NamedPerVariableCodec]] = (
                compressor.build(
                    ds_dtypes, ds_abs_mins, ds_abs_maxs, ds_mins, ds_maxs, error_bounds
                )
            )

            for compr_name, named_codecs in compressor_variants.items():
                for named_codec in named_codecs:
                    compressed_dataset = (
                        compressed_datasets
                        / dataset.parent.name
                        / named_codec.name
                        / compr_name
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
                            named_codec.codecs, ds
                        )
                    except Exception as e:
                        print(
                            f"Error compressing {dataset.parent.name} with {compressor.name}: {e}"
                        )
                        with (compressed_dataset / "error.out").open("w") as error_file:
                            error_file.write(traceback.format_exc())
                        print("Skipping...")
                        continue

                    with (compressed_dataset / "measurements.json").open("w") as f:
                        json.dump(measurements, f)

                    with progress_bar(progress):
                        ds_new.to_zarr(
                            compressed_dataset_path, encoding=dict(), compute=False
                        ).compute()


def compress_decompress(
    codecs: dict[str, Callable[[], Codec]],
    ds: xr.Dataset,
) -> tuple[xr.Dataset, dict]:
    variables = dict()
    measurements = dict()

    for v in ds:
        nbytes = BytesizeObserver()
        timing = WalltimeObserver()
        instructions = WasmCodecInstructionCounterObserver()

        codec: Codec = codecs[v]()  # type: ignore
        if not isinstance(codec, CodecStack):
            codec = CodecStack(codec)

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


def get_error_bounds(
    datasets_error_bounds: Path, dataset_name: str
) -> list[dict[str, ErrorBound]]:
    if not datasets_error_bounds.exists():
        raise FileNotFoundError(
            f"Expected error bounds to be defined in {datasets_error_bounds}. Run `scripts/create_error_bounds.py` to create them."
        )

    dataset_error_bounds = datasets_error_bounds / dataset_name
    with (dataset_error_bounds / "error_bounds.json").open() as f:
        error_bounds = json.load(f)
    return [
        {var_name: ErrorBound(**eb) for var_name, eb in eb_per_var.items()}
        for eb_per_var in error_bounds
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dataset", type=str, nargs="+", default=[])
    parser.add_argument("--include-dataset", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-compressor", type=str, nargs="+", default=[])
    parser.add_argument("--include-compressor", type=str, nargs="+", default=None)
    parser.add_argument("--basepath", type=Path, default=Path())
    parser.add_argument(
        "--data-loader-basepath", type=Path, default=Path() / ".." / "data-loader"
    )
    args = parser.parse_args()

    compress(
        basepath=args.basepath,
        exclude_dataset=args.exclude_dataset,
        include_dataset=args.include_dataset,
        exclude_compressor=args.exclude_compressor,
        include_compressor=args.include_compressor,
        data_loader_basepath=args.data_loader_basepath,
        progress=True,
    )
