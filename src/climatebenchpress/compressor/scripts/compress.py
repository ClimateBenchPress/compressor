__all__ = ["compress"]

import argparse
import json
import math
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

TARGET_CHUNK_SIZE = 4 * 1e6


def compress(
    basepath: Path = Path(),
    exclude_dataset: Container[str] = tuple(),
    include_dataset: None | Container[str] = None,
    exclude_compressor: Container[str] = tuple(),
    include_compressor: None | Container[str] = None,
    data_loader_basepath: None | Path = None,
    chunked: bool = False,
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
    chunked : bool
        Whether to chunk the input data.
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

        if chunked:
            for v in ds:
                word_size = ds[v].dtype.itemsize
                optimal_chunks = get_optimal_chunkshape(
                    ds[v], TARGET_CHUNK_SIZE, word_size=word_size
                )
                ds[v] = ds[v].chunk(optimal_chunks)
            # ds = ds.unify_chunks()

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
                    dataset_name = dataset.parent.name
                    if chunked:
                        dataset_name += "-chunked"
                    compressed_dataset = (
                        compressed_datasets
                        / dataset_name
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


def get_optimal_chunkshape(f, volume, word_size=4, logger=None):
    """
    Given a CF field, f get an optimal chunk shape using knowledge about the various dimensions.
    Our working assumption is that we want to have, for
     - hourly data, chunk shapes which are multiples of 12 in the time dimension
     - sub-daily data, chunk shapes which divide into a small multiple of 24
     - daily data, chunk shapes which are a multiple of 10
     - monthly data, chunk shapes which are a multiple of 12

    Function adapted from: https://github.com/NCAS-CMS/cfs3/blob/390ee593bfea1d926d6b814636b02fb4c430f91e/cfs3/cfchunking.py
    """

    default = get_chunkshape(np.array(f.data.shape), volume, word_size, logger)
    t_axis_name = f.cf.axes.get("T", None)
    if t_axis_name is None:
        raise ValueError(
            "Cannot identify a time axis, optimal chunk shape not possible"
        )
    t_data = f.cf["T"]
    interval = "u"
    if len(t_data) > 1:
        assert t_data.dtype == np.dtype("datetime64[ns]")
        # Input data is in ns. Convert delta unit to "day".
        t_delta = (t_data[1] - t_data[0]) / np.timedelta64(1, "D")
        t_delta = t_delta.item()

        if t_delta < 1:
            t_delta = round(t_delta * 24)
            if t_delta == 1:
                interval = "h"
            else:
                interval = int(24 / t_delta)
        elif t_delta == 1:
            interval = "d"
        else:
            interval = "m"

    try:
        index = f.dims.index(t_axis_name)
        guess = default[index]
        match interval:
            case "h":
                if guess < 3:
                    default[index] = 2
                elif guess < 6:
                    default[index] = 4
                elif guess < 12:
                    default[index] = 6
                elif guess < 19:
                    default[index] = 12
                else:
                    default[index] = round(guess / 24) * 24
            case "d":
                default[index] = round(guess / 10) * 10
            case "m":
                default[index] = round(guess / 12) * 12
            case "u":
                pass
            case _:
                default[index] = int(guess / interval) * interval
        if default[index] == 0:
            default[index] = guess  # well that clearly won't work so revert
        if guess != default[index] and logger:
            logger.info(f"Time chunk changed from {guess} to {default[index]}")
    except ValueError:
        pass
    return default


def get_chunkshape(shape, volume, word_size=4, logger=None, scale_tol=0.8):
    """
    Given a shape tuple, and byte size for the elements, calculate a suitable chunk shape
    for a given volume (in bytes). (We use word instead of dtype in case the user
    changes the data type within the writing operation.)

    Function adapted from: https://github.com/NCAS-CMS/cfs3/blob/390ee593bfea1d926d6b814636b02fb4c430f91e/cfs3/cfchunking.py
    """

    def constrained_largest_divisor(number, constraint):
        """
        Find the largest divisor of number which is less than the constraint
        """
        for i in range(int(constraint), 1, -1):
            if number % i == 0:
                return i
        return 1

    def revise(dimension, guess):
        """
        We need the largest integer (down) less than guess
        which is a factor of dimension, and we need
        to know how much smaller than guess it is,
        so that other dimensions can be scaled out.
        """
        old_guess = guess
        # there must be a more elegant way of doing this
        guess = constrained_largest_divisor(dimension, old_guess)
        scale_factor = old_guess / guess
        return scale_factor, guess

    v = volume / word_size
    size = np.prod(shape)

    n_chunks = int(size / v)
    root = v ** (1 / shape.size)

    # first get a scaled set of initial guess divisors
    initial_root = np.full(shape.size, root)
    ratios = [x / min(shape) for x in shape]
    other_root = 1.0 / (shape.size - 1)
    indices = list(range(shape.size))
    for i in indices:
        factor = ratios[i] ** other_root
        initial_root[i] = initial_root[i] * ratios[i]
        for j in indices:
            if j == i:
                continue
            initial_root[j] = initial_root[j] / factor

    weights_scaling = np.ones(shape.size)

    results = []
    remaining = 1
    for i in indices:
        # can't use zip because we are modifying weights in the loop
        d = shape[i]
        initial_guess = math.ceil(initial_root[i] * weights_scaling[i])
        if d % initial_guess == 0:
            results.append(initial_guess)
        else:
            scale_factor, next_guess = revise(d, initial_guess)
            results.append(next_guess)
            if remaining < shape.size:
                scale_factor = scale_factor ** (1 / (shape.size - remaining))
                weights_scaling[remaining:] = np.full(
                    shape.size - remaining, scale_factor
                )
        remaining += 1
        # fix up the last indice as we could have drifted quite small
        if i == indices[-1]:
            size_so_far = np.prod(np.array(results))
            scale_miss = size_so_far / v
            if scale_miss < scale_tol:
                constraint = results[-1] / (scale_miss)
                scaled_up = constrained_largest_divisor(shape[-1], constraint)
                results[-1] = scaled_up

    if logger:
        actual_n_chunks = int(np.prod(np.divide(shape, np.array(results))))
        cvolume = int(np.prod(np.array(results)) * 4)
        logger.info(
            f"Chunk size {results} - wanted {int(n_chunks)}/{int(volume)}B will get {actual_n_chunks}/{cvolume}B"
        )
    return results


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
    parser.add_argument("--chunked", action="store_true", default=False)
    args = parser.parse_args()

    compress(
        basepath=args.basepath,
        exclude_dataset=args.exclude_dataset,
        include_dataset=args.include_dataset,
        exclude_compressor=args.exclude_compressor,
        include_compressor=args.include_compressor,
        data_loader_basepath=args.data_loader_basepath,
        chunked=args.chunked,
        progress=True,
    )
