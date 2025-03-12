from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent.parent


def main():
    all_results = pd.read_csv(REPO / "metrics" / "all_results.csv")
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
    main()
