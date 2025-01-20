from pathlib import Path

import pandas as pd
import xarray as xr

import climatebenchpress.compressor

repo = Path(__file__).parent.parent

datasets = repo.parent / "data-loader" / "datasets"
compressed_datasets = repo / "compressed-datasets"
metrics_dir = repo / "metrics"

EVALUATION_METRICS: dict[str, climatebenchpress.compressor.metrics.abc.Metric] = {
    "MAE": climatebenchpress.compressor.metrics.MAE(),
    "Spectral Error": climatebenchpress.compressor.metrics.SpectralError(),
}

PASSFAIL_TESTS: dict[str, climatebenchpress.compressor.tests.abc.Test] = {
    "Spatial Relative Error": climatebenchpress.compressor.tests.SRE(),
    "R^2 Correlation": climatebenchpress.compressor.tests.R2(),
}


for dataset in compressed_datasets.iterdir():
    if dataset.name == ".gitignore":
        continue

    for compressor in dataset.iterdir():
        print(f"Evaluating {compressor.stem} on {dataset.name}...")

        compressed_dataset = compressed_datasets / dataset.name / compressor.stem
        compressed_dataset_path = compressed_dataset / "decompressed.zarr"

        uncompressed_dataset = datasets / dataset.name / "standardized.zarr"

        assert (
            compressed_dataset_path.exists()
        ), f"No compressed dataset at {compressed_dataset_path}"
        assert (
            uncompressed_dataset.exists()
        ), f"No uncompressed dataset at {uncompressed_dataset}"

        ds = xr.open_dataset(
            uncompressed_dataset, chunks=dict(), engine="zarr"
        ).compute()
        ds_new = xr.open_dataset(
            compressed_dataset_path, chunks=dict(), engine="zarr"
        ).compute()

        compressor_metrics = metrics_dir / dataset.name / compressor.stem
        compressor_metrics.mkdir(parents=True, exist_ok=True)

        metrics_path = compressor_metrics / "metrics.csv"
        if not metrics_path.exists():
            metrics = []
            for name, metric in EVALUATION_METRICS.items():
                for v in ds_new:
                    error = metric(ds[v], ds_new[v])
                    metrics.append(
                        {
                            "Metric": name,
                            "Variable": v,
                            "Error": error,
                        }
                    )
            pd.DataFrame(metrics).to_csv(metrics_path, index=False)

        tests_path = compressor_metrics / "tests.csv"
        if not tests_path.exists():
            tests = []
            for name, test in PASSFAIL_TESTS.items():
                for v in ds_new:
                    test_result, test_value = test(ds[v], ds_new[v])
                    tests.append(
                        {
                            "Test": name,
                            "Variable": v,
                            "Passed": test_result,
                            "Value": test_value,
                        }
                    )
            pd.DataFrame(tests).to_csv(tests_path, index=False)
