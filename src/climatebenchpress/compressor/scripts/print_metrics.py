__all__ = ["print_metrics"]

from pathlib import Path

import pandas as pd


def print_metrics(basepath: Path = Path()):
    all_results = pd.read_csv(basepath / "metrics" / "all_results.csv")
    for dataset in all_results["Dataset"].unique():
        print("\n" + 100 * "=")
        print(f"Results on {dataset}")
        print(100 * "=" + "\n")
        print(
            all_results[all_results["Dataset"] == dataset]
            .drop(columns=["Dataset"])
            .to_markdown(index=False)
        )


if __name__ == "__main__":
    print_metrics(basepath=Path())
