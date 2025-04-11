import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

COMPRESSOR2COLOR = {
    "jpeg2000": "blue",
    "zfp": "green",
    "sz3": "red",
    "bitround-pco-conservative-rel": "purple",
    "bitround-conservative-rel": "orange",
    "stochround": "brown",
    "tthresh": "pink",
}

REPO = Path(__file__).parent.parent


def rename_error_bounds(df, bound_names):
    # Get unique variables
    variables = df["Variable"].unique()

    # Initialize empty dataframes for each error bound category
    # low_bounds_df = pd.DataFrame(columns=df.columns)
    # medium_bounds_df = pd.DataFrame(columns=df.columns)
    # high_bounds_df = pd.DataFrame(columns=df.columns)
    # result_dfs = [
    #     low_bounds_df,
    #     medium_bounds_df,
    #     high_bounds_df,
    # ]

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


def normalize(data, bound_normalize="mid"):
    # Group by Variable and rank compressors within each variable
    ranked = data.copy()
    ranked = ranked[ranked["Error Bound"] == bound_normalize]
    ranked["CompRatio_Rank"] = ranked.groupby("Variable")[
        "Compression Ratio [raw B / enc B]"
    ].rank(ascending=False)

    # Calculate average rank for each compressor across all variables
    avg_ranks = ranked.groupby("Compressor")["CompRatio_Rank"].mean().reset_index()
    avg_ranks.columns = ["Compressor", "Average_Rank"]
    avg_ranks = avg_ranks.sort_values("Average_Rank")

    best_compressor = avg_ranks.iloc[0]["Compressor"]

    normalized = data.copy()
    normalize_vars = [
        ("Compression Ratio [raw B / enc B]", "Normalized_CR"),
        ("MAE", "Normalized_MAE"),
        ("Spatial Relative Error (Value)", "Normalized_SRE"),
        ("DSSIM", "Normalized_DSSIM"),
        ("Max Absolute Error", "Normalized_MaxAbsError"),
    ]
    # Avoid division by zero
    normalized["Spatial Relative Error (Value)"] = normalized[
        "Spatial Relative Error (Value)"
    ].replace(0.0, 1e-12)
    # Avoid negative values. By default, DSSIM is in the range [-1, 1].
    normalized["DSSIM"] = normalized["DSSIM"] + 1.0

    def get_normalizer(row):
        return normalized[
            (data["Compressor"] == best_compressor)
            & (data["Variable"] == row["Variable"])
            & (data["Error Bound"] == bound_normalize)
        ][col].item()

    for col, new_col in normalize_vars:
        normalized[new_col] = normalized.apply(
            lambda x: x[col] / get_normalizer(x),
            axis=1,
        )

    return normalized


def calculate_ranks(data):
    # Group by Variable and rank compressors within each variable
    ranked = data.copy()
    ranked["CompRatio_Rank"] = ranked.groupby("Variable")[
        "Compression Ratio [raw B / enc B]"
    ].rank(ascending=False)

    # Calculate average rank for each compressor across all variables
    avg_ranks = ranked.groupby("Compressor")["CompRatio_Rank"].mean().reset_index()
    avg_ranks.columns = ["Compressor", "Average_Rank"]
    avg_ranks = avg_ranks.sort_values("Average_Rank")

    best_compressor = avg_ranks.iloc[0]["Compressor"]

    normalized = data.copy()
    normalize_vars = [
        ("Compression Ratio [raw B / enc B]", "Normalized_CR"),
        ("MAE", "Normalized_MAE"),
        ("Spatial Relative Error (Value)", "Normalized_SRE"),
        ("Max Absolute Error", "Normalized_MaxAbsError"),
    ]
    normalized["Spatial Relative Error (Value)"] = normalized[
        "Spatial Relative Error (Value)"
    ].replace(0.0, 1e-12)

    def get_normalizer(row):
        return normalized[
            (data["Compressor"] == best_compressor)
            & (data["Variable"] == row["Variable"])
        ][col].item()

    for col, new_col in normalize_vars:
        normalized[new_col] = normalized.apply(
            lambda x: x[col] / get_normalizer(x),
            axis=1,
        )

    return normalized, avg_ranks


def plot_compression_data(data, outfile):
    # Calculate ranks
    data, avg_ranks = calculate_ranks(data)
    # Invert so that lower is better
    data["Normalized_CR"] = 1 / data["Normalized_CR"]

    # Get ordered list of compressors by average rank
    ordered_compressors = avg_ranks["Compressor"].tolist()

    plt.figure(figsize=(15, 8))

    # Variables for plotting
    variables = data["Variable"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(variables)))
    markers = ["o", "^", "s", "D", "d", "*", "P", "v", ">"]

    metrics_to_plot = [
        ("Normalized_CR", 0.3, colors[0]),
        ("Normalized_MAE", 0.0, colors[1]),
        ("Normalized_SRE", -0.3, colors[2]),
    ]

    for metric_name, offset, color in metrics_to_plot:
        for i, variable in enumerate(variables):
            var_data = data[data["Variable"] == variable]
            for j, comp in enumerate(ordered_compressors):
                comp_data = var_data[var_data["Compressor"] == comp]
                if not comp_data.empty:
                    plt.scatter(
                        j - offset,
                        comp_data[metric_name].iloc[0],
                        marker=markers[i],
                        s=100,
                        color=color,
                        alpha=0.7,
                    )
                if j != len(variables) - 1:
                    # Draw a vertical line at the end of each variable group
                    plt.axvline(
                        x=j + 0.5,
                        color="black",
                        alpha=0.5,
                        linewidth=0.5,
                    )

    # Configure plot
    plt.xlabel("Compressors (ordered by average rank)")
    plt.ylabel("Normalised Value")
    plt.yscale("log")
    plt.title("Compression Ratio and MAE by Compressor")
    plt.xticks(
        range(len(ordered_compressors)), ordered_compressors, rotation=45, ha="right"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xlim(-0.5, len(ordered_compressors) - 0.5)
    plt.ylim(1e-3, 1e2)

    # Create custom legends
    variable_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[i],
            color="w",
            markerfacecolor="black",
            markersize=10,
            label=variable,
        )
        for i, variable in enumerate(variables)
    ]

    variable_legend = plt.legend(
        handles=variable_handles,
        title="Variables",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    plt.gca().add_artist(variable_legend)

    metric_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=colors[0],
            linestyle="None",
            markersize=10,
            label="Compression Ratio",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=colors[1],
            linestyle="None",
            markersize=10,
            label="MAE",
        ),
    ]

    plt.legend(
        handles=metric_handles,
        title="Metrics",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.6),
    )

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_variables(data, outfile):
    variables = data["Variable"].unique()
    compressors = data["Compressor"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(variables)))
    markers = ["o", "^", "s", "D", "d", "*", "P", "H"]

    offset = 0.2
    # Plot compression ratios with circles
    # comp_ratio_marker = "o"
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Plot compression ratios on the left y-axis
    comp_ratio_color = colors[0]
    for i, compressor in enumerate(compressors):
        comp_data = data[data["Compressor"] == compressor]
        for j, variable in enumerate(variables):
            var_data = comp_data[comp_data["Variable"] == variable]
            if not var_data.empty:
                ax1.scatter(
                    j - offset,
                    var_data["Compression Ratio [raw B / enc B]"].iloc[0],
                    marker=markers[i],
                    s=100,
                    color=comp_ratio_color,
                    alpha=0.7,
                )

    ax1.set_ylabel("Compression Ratio [raw B / enc B]")
    ax1.set_yscale("log")

    # Create a second y-axis for MAE
    ax2 = ax1.twinx()
    mae_color = colors[1]
    for i, compressor in enumerate(compressors):
        comp_data = data[data["Compressor"] == compressor]
        for j, variable in enumerate(variables):
            var_data = comp_data[comp_data["Variable"] == variable]
            if not var_data.empty:
                ax2.scatter(
                    j + offset,
                    var_data["MAE"].iloc[0],
                    marker=markers[i],
                    s=100,
                    color=mae_color,
                    alpha=0.7,
                )
            if j != len(variables) - 1:
                # Draw a vertical line at the end of each variable group
                ax2.axvline(
                    x=j + 0.5,
                    color="black",
                    alpha=0.5,
                    linewidth=0.5,
                )

    ax2.set_ylabel("MAE")
    ax2.set_yscale("log")

    # Configure plot
    ax1.set_xlabel("Variables")
    plt.title("Compression Ratio and MAE by Variable")
    ax1.set_xticks(range(len(variables)))
    ax1.set_xticklabels(variables, rotation=45, ha="right")
    ax1.set_xlim(-0.5, len(variables) - 0.5)

    compressor_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=markers[i],
            color="w",
            markerfacecolor="black",
            markersize=10,
            label=compressor,
        )
        for i, compressor in enumerate(compressors)
    ]
    compressor_legend = ax1.legend(
        handles=compressor_handles,
        title="Compressors",
        loc="upper left",
        bbox_to_anchor=(1.1, 1),
    )
    ax1.add_artist(compressor_legend)  # Ensure the first legend is added to the plot

    metric_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=comp_ratio_color,
            linestyle="None",
            markersize=10,
            label="Compression Ratio",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=mae_color,
            linestyle="None",
            markersize=10,
            label="MAE",
        ),
    ]

    ax1.legend(
        handles=metric_handles,
        title="Metrics",
        loc="upper left",
        bbox_to_anchor=(1.1, 0.7),
    )

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(outfile, dpi=300)
    plt.close()


def plot_rd_curve(
    normalized_df,
    outfile,
    compression_metric="Compression Ratio [raw B / enc B]",
    distortion_metric="PSNR",
    agg="median",
    bound_names=["low", "mid", "high"],
):
    plt.figure(figsize=(10, 6))
    compressors = normalized_df["Compressor"].unique()
    agg_distortion = normalized_df.groupby(["Error Bound", "Compressor"])[
        [compression_metric, distortion_metric]
    ].agg(agg)
    for comp in compressors:
        compr_ratio = [
            agg_distortion.loc[(bound, comp), compression_metric]
            for bound in bound_names
        ]
        psnr = [
            agg_distortion.loc[(bound, comp), distortion_metric]
            for bound in bound_names
        ]
        plt.plot(
            compr_ratio,
            psnr,
            label=comp,
            marker="x",
            color=COMPRESSOR2COLOR[comp],
        )

    plt.title(f"{compression_metric} vs {distortion_metric}")
    plt.xlabel(compression_metric)
    plt.xscale("log")
    if "PSNR" not in distortion_metric:
        # PSNR is already on log scale.
        plt.yscale("log")
    plt.ylabel(distortion_metric)
    plt.legend(title="Compressor")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_bound_violations(df, bound_names, outfile):
    fig, axs = plt.subplots(1, 3, figsize=(len(bound_names) * 6, 6), sharey=True)

    for i, bound_name in enumerate(bound_names):
        df_bound = df[df["Error Bound"] == bound_name]
        pass_fail = df_bound.pivot(
            index="Compressor", columns="Variable", values="Satisfies Bound (Passed)"
        )
        pass_fail = pass_fail.astype(np.float32)
        fraction_fail = df_bound.pivot(
            index="Compressor", columns="Variable", values="Satisfies Bound (Value)"
        )
        annotations = fraction_fail.map(
            lambda x: "{:.2f}".format(x * 100) if x * 100 >= 0.01 else "<0.01"
        )
        annotations[fraction_fail == 0.0] = ""
        sns.heatmap(
            pass_fail,
            cbar=False,
            cmap="vlag_r",
            annot=annotations,
            fmt="s",
            linewidths=0.5,
            ax=axs[i],
        )
        axs[i].set_title(f"Bound: {bound_name}")
        if i != 0:
            axs[i].set_ylabel("")

    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close()


def main(csv_file):
    plots_path = REPO / "test-plots"

    df = pd.read_csv(csv_file)
    bound_names = ["low", "mid", "high"]
    df = rename_error_bounds(df, bound_names)
    normalized_df = normalize(df, bound_normalize="mid")

    plot_bound_violations(
        normalized_df, bound_names, plots_path / "bound_violations.png"
    )

    for metric in ["Normalized_MAE", "Normalized_DSSIM", "Normalized_MaxAbsError"]:
        plot_rd_curve(
            normalized_df,
            plots_path / f"rd_curve_{metric.lower()}.png",
            compression_metric="Normalized_CR",
            distortion_metric=metric,
            agg="median",
        )

    for name in bound_names:
        error_df = df[df["Error Bound"] == name]
        plot_compression_data(
            error_df, plots_path / f"metrics_per_compressor_{name}_bounds.png"
        )
        plot_variables(error_df, plots_path / f"metrics_per_variable_{name}_bound.png")

        # Create a pivot table with Compressors as rows and Variables as columns
        pivot_table = error_df.pivot(
            index="Compressor",
            columns="Variable",
            values="Compression Ratio [raw B / enc B]",
        )

        # Compute rankings for each column (highest compression ratio gets rank 1)
        rank_table = pivot_table.rank(axis=0, ascending=False, method="min")

        pivot_table["Mean CR"] = pivot_table.mean(axis=1)
        pivot_table["Median CR"] = pivot_table.median(axis=1)
        pivot_table["Mean Rank"] = rank_table.mean(axis=1)
        pivot_table = pivot_table.sort_values(by="Mean Rank")

        print(pivot_table.to_markdown(index=True))
        print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    args = parser.parse_args()
    main(args.csv_file)
