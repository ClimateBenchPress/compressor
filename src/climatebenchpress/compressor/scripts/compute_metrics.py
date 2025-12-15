__all__ = ["compute_metrics"]

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr

import climatebenchpress.compressor

EVALUATION_METRICS: dict[str, climatebenchpress.compressor.metrics.abc.Metric] = {
    "MAE": climatebenchpress.compressor.metrics.MAE(),
    "Max Absolute Error": climatebenchpress.compressor.metrics.MaxAbsError(),
    "Max Relative Error": climatebenchpress.compressor.metrics.MaxRelError(),
    "Spectral Error": climatebenchpress.compressor.metrics.SpectralError(),
    "DSSIM": climatebenchpress.compressor.metrics.DSSIM(),
    "PSNR": climatebenchpress.compressor.metrics.PSNR(),
}

PASSFAIL_TESTS: dict[str, climatebenchpress.compressor.tests.abc.Test] = {
    "Spatial Relative Error": climatebenchpress.compressor.tests.SRE(),
    "R^2 Correlation": climatebenchpress.compressor.tests.R2(),
}


def compute_metrics(
    basepath: Path = Path(),
    data_loader_basepath: None | Path = None,
    exclude_dataset: Iterable[str] = tuple(),
    include_dataset: None | Iterable[str] = None,
    exclude_compressor: Iterable[str] = tuple(),
    include_compressor: None | Iterable[str] = None,
):
    """Compute evaluation metrics for compressors.

    Parameters
    ----------
    basepath : Path
        Assumes the compressed datasets are stored in `basepath / compressed-datasets`.
        Computed metrics will be stored in `basepath / metrics`.
    data_loader_basepath : None | Path
        Base path for the data loader datasets. If `None`, defaults to `basepath / .. / data-loader`.
        Input datasets will be loaded from `data_loader_basepath / datasets`.
    exclude_dataset : Iterable[str]
        Datasets to exclude from evaluation.
    include_dataset : None | Iterable[str]
        Datasets to include in evaluation. If `None`, all datasets are included.
        If specified, only datasets in `include_dataset` will be evaluated.
    exclude_compressor : Iterable[str]
        Compressors to exclude from evaluation.
    include_compressor : None | Iterable[str]
        Compressors to include in evaluation. If `None`, all compressors are included.
        If specified, only compressors in `include_compressor` will be evaluated.
    """
    exclude_compressor = add_compressor_suffixes(exclude_compressor)
    include_compressor = add_compressor_suffixes(include_compressor)

    datasets = (data_loader_basepath or basepath) / "datasets"
    compressed_datasets = basepath / "compressed-datasets"
    metrics_dir = basepath / "metrics"

    for dataset in compressed_datasets.iterdir():
        if dataset.name == ".gitignore" or dataset.name in exclude_dataset:
            continue
        if include_dataset and dataset.name not in include_dataset:
            continue

        for error_bound in dataset.iterdir():
            variable2error_bound = parse_error_bounds(error_bound.name)

            for compressor in error_bound.iterdir():
                if compressor.stem in exclude_compressor:
                    continue
                if include_compressor and compressor.stem not in include_compressor:
                    continue
                print(f"Evaluating {compressor.stem} on {dataset.name}...")

                compressed_dataset = (
                    compressed_datasets
                    / dataset.name
                    / error_bound.name
                    / compressor.stem
                )
                compressed_dataset_path = compressed_dataset / "decompressed.zarr"
                uncompressed_dataset = datasets / dataset.name / "standardized.zarr"
                if not compressed_dataset_path.exists():
                    print(f"No compressed dataset at {compressed_dataset_path}")
                    continue
                if not uncompressed_dataset.exists():
                    print(f"No uncompressed dataset at {uncompressed_dataset}")
                    continue

                ds = xr.open_zarr(uncompressed_dataset, chunks=dict()).compute()
                ds_new = xr.open_zarr(compressed_dataset_path, chunks=dict()).compute()

                compressor_metrics = (
                    metrics_dir / dataset.name / error_bound.name / compressor.stem
                )
                compressor_metrics.mkdir(parents=True, exist_ok=True)

                compute_compressor_metrics(compressor_metrics, ds, ds_new)
                compute_tests(compressor_metrics, variable2error_bound, ds, ds_new)


def add_compressor_suffixes(compressors: None | Iterable[str]) -> list[str]:
    if compressors is None:
        return []

    extended_compressors = []
    for compressor in compressors:
        extended_compressors.append(compressor)
        extended_compressors.append(compressor + "-conservative-rel")
        extended_compressors.append(compressor + "-conservative-abs")

    return extended_compressors


def parse_error_bounds(error_bound_str: str) -> dict[str, tuple[str, float]]:
    """
    The error bound string is of the form
    "{variable_name1}-{error_type1}={error_bound1}_{variable_name2}-{error_type2}={error_bound2}".
    More than 2 variables are possible.
    Each variable name can itself contain an underscore.
    The error type is either "abs_error" or "rel_error".
    The error bound is a floating point number represented either in decimal or scientific notation.

    This function parses the string and returns a dictionary of the form
    {
        "variable_name1": (error_type1, error_bound1),
        "variable_name2": (error_type2, error_bound2),
    }

    For example, the string
    "pr-abs_error=3.108691982924938e-05_rlut-abs_error=0.2788982238769531"
    would be parsed as
    {
        "pr": ("abs_error", 3.108691982924938e-05),
        "rlut": ("abs_error", 0.2788982238769531),
    }
    """
    pattern = re.compile(
        r"(?:_?)"  # Underscore at the beginning separating the different variables.
        r"(?P<variable>[\w]+)"  # Variable name can any alphanumeric character.
        r"-(?P<error_type>abs_error|rel_error)="  # Error type is either "abs_error" or "rel_error".
        r"(?P<error_bound>\d+(\.\d+)?([eE][+-]?\d+)?)"  # Error bound is a floating point number.
    )
    result = {}
    for match in pattern.finditer(error_bound_str):
        try:
            error_bound = float(match["error_bound"])
        except ValueError:
            raise ValueError(
                f"Error bound '{match['error_bound']}' from '{error_bound_str}' is not a valid float"
            )

        result[match["variable"]] = (match["error_type"], error_bound)

    assert len(result) > 0, (
        f"Error bound string {error_bound_str} does not match expected format"
    )

    return result


def compute_compressor_metrics(
    compressor_metrics: Path, ds: xr.Dataset, ds_new: xr.Dataset
) -> pd.DataFrame:
    metrics_path = compressor_metrics / "metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)

    metric_list = []
    for name, metric in EVALUATION_METRICS.items():
        for v in ds_new:
            try:
                error = metric(ds[v], ds_new[v])
            except Exception as e:
                print(
                    f"Error computing metric {name} for variable {v} on "
                    f"{compressor_metrics.parent.name}: {e}"
                )
                error = float("nan")

            metric_list.append(
                {
                    "Metric": name,
                    "Variable": v,
                    "Error": error,
                }
            )
    metrics = pd.DataFrame(metric_list)
    metrics.to_csv(metrics_path, index=False)
    return metrics


def compute_tests(
    compressor_metrics: Path,
    variable2bound: dict[str, tuple[str, float]],
    ds: xr.Dataset,
    ds_new: xr.Dataset,
) -> pd.DataFrame:
    tests_path = compressor_metrics / "tests.csv"
    if tests_path.exists():
        return pd.read_csv(tests_path)

    test_list = []
    for name, test in PASSFAIL_TESTS.items():
        for v in ds_new:
            try:
                test_result, test_value = test(ds[v], ds_new[v])
            except Exception as e:
                print(
                    f"Error computing test {name} for variable {v} on "
                    f"{compressor_metrics.parent.name}: {e}"
                )
                test_result = False
                test_value = float("nan")

            test_list.append(
                {
                    "Test": name,
                    "Variable": v,
                    "Passed": test_result,
                    "Value": test_value,
                }
            )

    for v in ds_new:
        error_type, bound = variable2bound[str(v)]
        test = climatebenchpress.compressor.tests.ErrorBound(
            error_type=error_type, threshold=bound
        )
        test_result, test_value = test(ds[v], ds_new[v])
        test_list.append(
            {
                "Test": "Satisfies Bound",
                "Variable": v,
                "Passed": test_result,
                "Value": test_value,
            }
        )

    tests = pd.DataFrame(test_list)
    tests.to_csv(tests_path, index=False)
    return tests


def load_measurements(compressed_dataset: Path, compressor: Path) -> pd.DataFrame:
    with (compressed_dataset / "measurements.json").open() as f:
        measurements = json.load(f)

    rows = []
    for var, variable_measurements in measurements.items():
        rows.append(
            {
                "Compressor": compressor.stem,
                "Variable": var,
                "Compression Ratio [raw B / enc B]": variable_measurements[
                    "decoded_bytes"
                ]
                / variable_measurements["encoded_bytes"],
                "Encode Instructions [# / raw B]": (
                    None
                    if variable_measurements["encode_instructions"] is None
                    else (
                        variable_measurements["encode_instructions"]
                        / variable_measurements["decoded_bytes"]
                    )
                ),
                "Decode Instructions [# / raw B]": (
                    None
                    if variable_measurements["decode_instructions"] is None
                    else (
                        variable_measurements["decode_instructions"]
                        / variable_measurements["decoded_bytes"]
                    )
                ),
                "Encode Throughput [raw B / s]": (
                    None
                    if variable_measurements["encode_timing"] == 0
                    else variable_measurements["decoded_bytes"]
                    / variable_measurements["encode_timing"]
                ),
                "Decode Throughput [raw B / s]": (
                    None
                    if variable_measurements["decode_timing"] == 0
                    else variable_measurements["decoded_bytes"]
                    / variable_measurements["decode_timing"]
                ),
            }
        )
    return pd.DataFrame(rows)


def merge_metrics(
    measurements: pd.DataFrame, metrics: pd.DataFrame, tests: pd.DataFrame
) -> pd.DataFrame:
    # Turn each metric/test into a column. Merge on "variable" to avoid duplicating
    # the "variable" column.
    test_per_variable = tests.pivot(
        index="Variable", columns="Test", values=["Passed", "Value"]
    )
    # mypy cannot infer that test_per_variable.columns is a MultiIndex and therefore
    # gives spurious errors for this assignment.
    test_per_variable.columns = [  # type: ignore
        f"{metric_name} ({passed_or_val})"  # type: ignore
        for passed_or_val, metric_name in test_per_variable.columns  # type: ignore
    ]
    return pd.merge(
        measurements,
        metrics.pivot(index="Variable", columns="Metric", values="Error")
        .reset_index()
        .merge(
            test_per_variable.reset_index(),
            on="Variable",
        ),
        on="Variable",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=Path, default=Path())
    parser.add_argument(
        "--data-loader-basepath", type=Path, default=Path() / ".." / "data-loader"
    )
    parser.add_argument("--exclude-dataset", type=str, nargs="+", default=[])
    parser.add_argument("--include-dataset", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-compressor", type=str, nargs="+", default=[])
    parser.add_argument("--include-compressor", type=str, nargs="+", default=None)
    args = parser.parse_args()
    compute_metrics(
        basepath=args.basepath,
        data_loader_basepath=args.data_loader_basepath,
        exclude_dataset=args.exclude_dataset,
        include_dataset=args.include_dataset,
        exclude_compressor=args.exclude_compressor,
        include_compressor=args.include_compressor,
    )
