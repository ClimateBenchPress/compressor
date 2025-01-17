from pathlib import Path

import pandas as pd
import xarray as xr

import climatebenchpress.compressor

datasets = Path("..") / "data-loader" / "datasets"
compressed_datasets = Path("compressed-datasets")
metrics_path = Path("metrics")

EVALUATION_METRICS: dict[str, climatebenchpress.compressor.metrics.abc.Metric] = {
    "MAE": climatebenchpress.compressor.metrics.MAE(),
    "Spectral Error": climatebenchpress.compressor.metrics.SpectralError(),
}

PASSFAIL_TESTS: dict[str, climatebenchpress.compressor.tests.abc.Test] = {
    "Spatial Relative Error": climatebenchpress.compressor.tests.SRE(),
    "R^2 Correlation": climatebenchpress.compressor.tests.R2(),
}


for dataset in datasets.iterdir():
    if dataset.name == ".gitignore":
        continue

    dataset /= "standardized.zarr"

    for compressor in (compressed_datasets / dataset.parent.name).iterdir():
        print(f"Evaluating {compressor.stem} on {dataset.parent.name}...")
        compressed_dataset = compressed_datasets / dataset.parent.name / compressor.stem
        compressed_dataset.mkdir(parents=True, exist_ok=True)

        compressed_dataset_path = compressed_dataset / "decompressed.zarr"

        assert (
            compressed_dataset_path.exists()
        ), f"No compressed dataset at {compressed_dataset_path}"

        ds = xr.open_dataset(dataset, chunks=dict(), engine="zarr").compute()
        ds_new = xr.open_dataset(
            compressed_dataset_path, chunks=dict(), engine="zarr"
        ).compute()

        compressor_metrics = metrics_path / dataset.parent.name / compressor.stem
        compressor_metrics.mkdir(parents=True, exist_ok=True)

        metrics = []
        for name, metric in EVALUATION_METRICS.items():
            for v in ds_new:
                error = metric(ds[v], ds_new[v])
                metrics.append(
                    {
                        "metric": name,
                        "variable": v,
                        "error": error,
                    }
                )
        pd.DataFrame(metrics).to_csv(compressor_metrics / "metrics.csv", index=False)

        tests = []
        for name, test in PASSFAIL_TESTS.items():
            for v in ds_new:
                test_result, test_value = test(ds[v], ds_new[v])
                tests.append(
                    {
                        "test": name,
                        "variable": v,
                        "passed": test_result,
                        "value": test_value,
                    }
                )
        pd.DataFrame(tests).to_csv(compressor_metrics / "tests.csv", index=False)
