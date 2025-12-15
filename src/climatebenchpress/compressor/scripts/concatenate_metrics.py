__all__ = ["concatenate_metrics"]

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .compute_metrics import parse_error_bounds


def concatenate_metrics(basepath: Path = Path()):
    """Concatenate metrics from all datasets and compressors into a single CSV file.

    Parameters
    ----------
    basepath : Path
        Assumes that the metrics are stored in `basepath / metrics`. The script will
        create a `basepath / metrics / all_results.csv` file containing the concatenated results.
    """
    compressed_datasets = basepath / "compressed-datasets"
    error_bounds_dir = basepath / "datasets-error-bounds"
    metrics_dir = basepath / "metrics"

    all_results = []
    for dataset in metrics_dir.iterdir():
        if not dataset.is_dir():
            continue

        with (error_bounds_dir / dataset.name / "error_bounds.json").open() as f:
            error_bound_list = json.load(f)

        for error_bound in dataset.iterdir():
            variable2error_bound = parse_error_bounds(error_bound.name)
            error_bound_name = get_error_bound_name(
                variable2error_bound, error_bound_list
            )

            for compressor in error_bound.iterdir():
                metrics_csv = compressor / "metrics.csv"
                metrics = pd.read_csv(metrics_csv)
                tests_csv = compressor / "tests.csv"
                tests = pd.read_csv(tests_csv)
                compressed_dataset = (
                    compressed_datasets
                    / dataset.name
                    / error_bound.name
                    / compressor.stem
                )
                measurements = load_measurements(compressed_dataset, compressor)

                df = merge_metrics(measurements, metrics, tests)
                df["Dataset"] = dataset.name
                df["Error Bound"] = error_bound.name
                df["Error Bound Name"] = error_bound_name
                all_results.append(df)

    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(metrics_dir / "all_results.csv", index=False)


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


def get_error_bound_name(
    variable2bound: dict[str, tuple[str, float]],
    error_bound_list: list[dict[str, dict[str, Optional[float]]]],
    bound_names: list[str] = ["low", "mid", "high"],
) -> str:
    """The function returns either "low", "mid", or "high" depending on which error bound
    from the variable2bound dictionary matches the exact error bound in the error_bound_list.

    error_bound_list contains one dictionary for each error bound (low, mid, high).
    Each of these dictionaries contains the error bounds for
    each variable. The variable names in the dictionaries should exactly match the variable names
    in the variable2bound dictionary.

    Parameters
    ----------
    variable2bound : dict[str, tuple[str, float]]
        A dictionary representing a single error bound, mapping variable names to
        tuples of error type and error bound. The error type is either "abs_error"
        or "rel_error", and the error bound is a float.
    error_bound_list : list[dict[str, dict[str, Optional[float]]]]
        A list of dictionaries, each representing an error bound (low, mid, high).
        Each dictionary contains variable names as keys and a dictionary of error types
        and bounds as values.
    bound_names : list[str], optional
        A list of names for the error bounds, by default ["low", "mid", "high"].
    """

    # Convert the variable2bound dictionary to match the format of error_bound_list.
    new_bound_format = dict()
    for k in variable2bound.keys():
        new_bound_format[k] = {
            "abs_error": (
                variable2bound[k][1] if variable2bound[k][0] == "abs_error" else None
            ),
            "rel_error": (
                variable2bound[k][1] if variable2bound[k][0] == "rel_error" else None
            ),
        }

    # Return the name of the error bound that matches new_bound_format.
    for bound_name, error_bound in zip(bound_names, error_bound_list):
        if new_bound_format == error_bound:
            return bound_name

    raise ValueError(
        f"Error bounds {new_bound_format} do not match any of the error bounds "
        f"{error_bound_list}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basepath", type=Path, default=Path())
    args = parser.parse_args()

    concatenate_metrics(basepath=args.basepath)
