import json
from pathlib import Path

import climatebenchpress.compressor
import pandas as pd
import xarray as xr

REPO = Path(__file__).parent.parent

EVALUATION_METRICS: dict[str, climatebenchpress.compressor.metrics.abc.Metric] = {
    "MAE": climatebenchpress.compressor.metrics.MAE(),
    "Spectral Error": climatebenchpress.compressor.metrics.SpectralError(),
    "PSNR": climatebenchpress.compressor.metrics.PSNR(),
}

PASSFAIL_TESTS: dict[str, climatebenchpress.compressor.tests.abc.Test] = {
    "Spatial Relative Error": climatebenchpress.compressor.tests.SRE(),
    "R^2 Correlation": climatebenchpress.compressor.tests.R2(),
}


def main():
    datasets = REPO.parent / "data-loader" / "datasets"
    compressed_datasets = REPO / "compressed-datasets"
    metrics_dir = REPO / "metrics"

    all_results = []
    for dataset in compressed_datasets.iterdir():
        if dataset.name == ".gitignore":
            continue

        for compressor in dataset.iterdir():
            print(f"Evaluating {compressor.stem} on {dataset.name}...")

            compressed_dataset = compressed_datasets / dataset.name / compressor.stem
            compressed_dataset_path = compressed_dataset / "decompressed.zarr"
            uncompressed_dataset = datasets / dataset.name / "standardized.zarr"
            assert compressed_dataset_path.exists(), (
                f"No compressed dataset at {compressed_dataset_path}"
            )
            assert uncompressed_dataset.exists(), (
                f"No uncompressed dataset at {uncompressed_dataset}"
            )

            ds = xr.open_zarr(uncompressed_dataset, chunks=dict()).compute()
            ds_new = xr.open_zarr(compressed_dataset_path, chunks=dict()).compute()

            compressor_metrics = metrics_dir / dataset.name / compressor.stem
            compressor_metrics.mkdir(parents=True, exist_ok=True)

            metrics = compute_metrics(compressor_metrics, ds, ds_new)
            tests = compute_tests(compressor_metrics, ds, ds_new)
            measurements = load_measurements(compressed_datasets, dataset, compressor)

            df = merge_metrics(measurements, metrics, tests)
            df["Dataset"] = dataset.name
            all_results.append(df)

    all_results = pd.concat(all_results)
    all_results.to_csv(metrics_dir / "all_results.csv", index=False)


def compute_metrics(
    compressor_metrics: Path, ds: xr.Dataset, ds_new: xr.Dataset
) -> pd.DataFrame:
    metrics_path = compressor_metrics / "metrics.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)

    metric_list = []
    for name, metric in EVALUATION_METRICS.items():
        for v in ds_new:
            error = metric(ds[v], ds_new[v])
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
    compressor_metrics: Path, ds: xr.Dataset, ds_new: xr.Dataset
) -> pd.DataFrame:
    tests_path = compressor_metrics / "tests.csv"
    if tests_path.exists():
        return pd.read_csv(tests_path)

    test_list = []
    for name, test in PASSFAIL_TESTS.items():
        for v in ds_new:
            test_result, test_value = test(ds[v], ds_new[v])
            test_list.append(
                {
                    "Test": name,
                    "Variable": v,
                    "Passed": test_result,
                    "Value": test_value,
                }
            )
    tests = pd.DataFrame(test_list)
    tests.to_csv(tests_path, index=False)
    return tests


def load_measurements(
    compressed_datasets: Path, dataset: Path, compressor: Path
) -> pd.DataFrame:
    with open(
        compressed_datasets / dataset.name / compressor.stem / "measurements.json"
    ) as f:
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
            }
        )
    return pd.DataFrame(rows)


def merge_metrics(
    measurements: pd.DataFrame, metrics: pd.DataFrame, tests: pd.DataFrame
) -> pd.DataFrame:
    # Turn each metric into a column. Merge on "variable" to avoid duplicating
    # the "variable" column.
    return pd.merge(
        measurements,
        metrics.pivot(index="Variable", columns="Metric", values="Error")
        .reset_index()
        .merge(
            tests.pivot(
                index="Variable", columns="Test", values="Passed"
            ).reset_index(),
            on="Variable",
        ),
        on="Variable",
    )


if __name__ == "__main__":
    main()
