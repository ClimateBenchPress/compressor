import json
from pathlib import Path

REPO = Path(__file__).parent.parent

ERROR_BOUNDS = [
    {"abs_error": 0.01, "rel_error": None},
    {"abs_error": 0.1, "rel_error": None},
]


def main():
    datasets = REPO.parent / "data-loader" / "datasets"
    datasets_error_bounds = REPO / "datasets-error-bounds"

    for dataset in datasets.iterdir():
        if dataset.name == ".gitignore":
            continue

        dataset_error_bounds = datasets_error_bounds / dataset.name
        dataset_error_bounds.mkdir(parents=True, exist_ok=True)
        with open(dataset_error_bounds / "error_bounds.json", "w") as f:
            json.dump(ERROR_BOUNDS, f)


if __name__ == "__main__":
    main()
