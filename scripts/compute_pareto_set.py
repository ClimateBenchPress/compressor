import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import paretoset
import seaborn as sns

METRIC_OPTIMIZER = {
    "DSSIM": "max",
    "MAE": "min",
    "MaxAbsErr": "min",
    "CR": "max",
}

REPO = Path(__file__).parent.parent


def rename_error_bounds(df, bound_names):
    # Get unique variables
    variables = df["Variable"].unique()

    # Process each variable
    for variable in variables:
        var_selector = df["Variable"] == variable
        var_data = df[var_selector]

        error_bounds = sorted(
            var_data["Error Bound"].unique(),
            key=lambda x: float(x.split("=")[1].split("_")[0]),
        )

        assert len(error_bounds) == len(bound_names)
        for i in range(len(error_bounds)):
            bound_selector = var_data["Error Bound"] == error_bounds[i]
            df.loc[bound_selector & var_selector, "Error Bound"] = bound_names[i]

    return df


def plot_ranking_stats(df, bound_names, outfile):
    fig, axs = plt.subplots(1, 3, figsize=(len(bound_names) * 6, 6), sharey=True)

    number_first = df.pivot(
        index="Compressor", columns=["Metric", "Error Bound"], values="Variable"
    ).fillna(0.0)
    for i, bound_name in enumerate(bound_names):
        bound_columns = [col for col in number_first.columns if bound_name in col]
        sns.heatmap(
            number_first[bound_columns],
            annot=True,
            fmt=".0f",
            cbar=False,
            linewidths=0.5,
            ax=axs[i],
        )
        axs[i].set_title(f"Bound: {bound_name}")
        if i != 0:
            axs[i].set_ylabel("")

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close()


def print_pareto_set(df, outfile):
    df = pd.pivot_table(
        data=df,
        index="Compressor",
        columns=["Variable", "Error Bound"],
        values=[
            "DSSIM",
            "MAE",
            "MaxAbsErr",
            "CR",
        ],
    )
    # Flatten the MultiIndex columns
    min_or_max = [METRIC_OPTIMIZER[metric] for metric, _, _ in df.columns]
    # df.columns = [
    #     f"{error_bound} {var} {metric}" for metric, var, error_bound in df.columns
    # ]

    mask = paretoset.paretoset(df, sense=min_or_max)

    # pareto_compressors = {comp: [] for comp in df[mask].index}
    pareto_compressors = []
    for col, sense in zip(df.columns, min_or_max):
        best_compressor = (
            df.loc[mask, col].idxmin() if sense == "min" else df.loc[mask, col].idxmax()
        )
        # pareto_compressors[best_compressor].append(col)
        pareto_compressors.append((best_compressor, *col))
    pareto_compressors = pd.DataFrame.from_records(
        pareto_compressors,
        columns=["Compressor", "Metric", "Variable", "Error Bound"],
    )
    compressor_is_best = pareto_compressors.groupby(
        ["Compressor", "Metric", "Error Bound"]
    ).count()
    compressor_is_best = compressor_is_best.reset_index()
    plot_ranking_stats(compressor_is_best, ["low", "mid", "high"], outfile)
    # for compressor, best_columns in pareto_compressors.items():
    #     print(f"{compressor}: {best_columns}")


def main(csv_file):
    plots_path = REPO / "test-plots"

    df = pd.read_csv(csv_file)
    df = rename_error_bounds(df, ["low", "mid", "high"])
    df = df.rename(
        columns={
            "Compression Ratio [raw B / enc B]": "CR",
            "Max Absolute Error": "MaxAbsErr",
        }
    )
    print_pareto_set(df, plots_path / "pareto_ranking.png")

    df_passed_bound = df[df["Satisfies Bound (Passed)"]]
    print_pareto_set(df_passed_bound, plots_path / "filtered_pareto_ranking.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    args = parser.parse_args()
    main(args.csv_file)
