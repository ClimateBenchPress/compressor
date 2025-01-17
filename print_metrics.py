import json
from pathlib import Path

import pandas as pd

compressed_datasets = Path("compressed-datasets")
metrics_path = Path("metrics")

for dataset in metrics_path.iterdir():
    if dataset.name == ".gitignore":
        continue

    print(100 * "=")
    print(f"Results on {dataset.name}")
    print(100 * "=" + "\n")
    data = []
    for compressor in (metrics_path / dataset.name).iterdir():
        compressor_metrics = metrics_path / dataset.name / compressor.stem

        metrics = pd.read_csv(compressor_metrics / "metrics.csv")

        tests = pd.read_csv(compressor_metrics / "tests.csv")

        with open(
            compressed_datasets / dataset.name / compressor.stem / "measurements.json"
        ) as f:
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
