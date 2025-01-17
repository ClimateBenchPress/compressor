import json
from pathlib import Path

import pandas as pd

repo = Path(__file__).parent.parent

compressed_datasets = repo / "compressed-datasets"
metrics_path = repo / "metrics"

for dataset in metrics_path.iterdir():
    if dataset.name == ".gitignore":
        continue

    print("\n" + 100 * "=")
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

        num_variables = len(measurements.keys())
        rows = []
        for var, variable_measurements in measurements.items():
            rows.append(
                {
                    "Compressor": compressor.stem,
                    "variable": var,
                    "Compression Ratio [raw B / enc B]": variable_measurements[0][
                        "decoded_bytes"
                    ]
                    / variable_measurements[-1]["encoded_bytes"],
                }
            )
        measurements = pd.DataFrame(rows)

        # Merge DataFrames column-wise
        data.append(
            pd.merge(
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
                on="variable",
            )
        )

    df = pd.concat(data)
    print(df.set_index(["Compressor", "variable"]).to_markdown())
