import json
from pathlib import Path

import pandas as pd

datasets = Path("..") / "data-loader" / "datasets"
compressors = Path("compressors")
compressed_datasets = Path("compressed-datasets")

for dataset in datasets.iterdir():
    if dataset.name == ".gitignore":
        continue

    dataset /= "standardized.zarr"

    print(50 * "=")
    print(f"Results on {dataset.parent.name}")
    print(50 * "=")
    data = []
    for compressor in compressors.iterdir():
        compressed_dataset = compressed_datasets / dataset.parent.name / compressor.stem
        compressed_dataset.mkdir(parents=True, exist_ok=True)

        compressed_dataset_path = compressed_dataset / "decompressed.zarr"

        metrics_path = compressed_dataset / "metrics.csv"
        metrics = pd.read_csv(metrics_path)

        tests_path = compressed_dataset / "tests.csv"
        tests = pd.read_csv(tests_path)

        with open(compressed_dataset / "measurements.json") as f:
            measurements = json.load(f)

        measurements = pd.DataFrame(
            {
                "Compressor": [compressor.stem],
                "Compression Ratio [raw B / enc B]": [
                    measurements[0]["decoded_bytes"] / measurements[-1]["encoded_bytes"]
                ],
            }
        )

        # Merge DataFrames column-wise
        data.append(
            pd.concat(
                [
                    measurements,
                    # Turn each metric into a column. Merge on "variable" to avoid duplicating
                    # the "variable" column.
                    metrics.pivot(index="variable", columns="metric", values="error")
                    .reset_index()
                    .merge(
                        tests.pivot(
                            index="variable", columns="test", values="passed"
                        ).reset_index(),
                        on="variable",
                    ),
                ],
                axis=1,
            )
        )

    df = pd.concat(data)
    print(df.set_index(["Compressor", "variable"]).to_markdown())
