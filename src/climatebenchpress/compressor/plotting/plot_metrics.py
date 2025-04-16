from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from .variable_plotters import PLOTTERS

COMPRESSOR2COLOR = {
    "jpeg2000": "#EE7733",
    "zfp": "#EE3377",
    "sz3": "#CC3311",
    "bitround-pco-conservative-rel": "#0077BB",
    "bitround-conservative-rel": "#33BBEE",
    "stochround": "#009988",
    "tthresh": "#BBBBBB",
}

COMPRESSOR2LEGEND_NAME = {
    "jpeg2000": "JPEG2000",
    "zfp": "ZFP",
    "sz3": "SZ3",
    "bitround-pco-conservative-rel": "BitRound + PCO",
    "bitround-conservative-rel": "BitRound + Zlib",
    "stochround": "StochRound",
    "tthresh": "TTHRESH",
}


def plot_metrics(
    basepath: Path = Path(), bound_names: list[str] = ["low", "mid", "high"]
):
    metrics_path = basepath / "metrics"
    plots_path = basepath / "plots"

    df = pd.read_csv(metrics_path / "all_results.csv")
    plot_per_variable_metrics(
        basepath=basepath,
        plots_path=plots_path,
        all_results=df,
    )

    df = rename_error_bounds(df, bound_names)
    normalized_df = normalize(df, bound_normalize="mid")

    plot_bound_violations(
        normalized_df, bound_names, plots_path / "bound_violations.png"
    )

    for metric in ["Normalized_MAE", "Normalized_DSSIM", "Normalized_MaxAbsError"]:
        plot_aggregated_rd_curve(
            normalized_df,
            plots_path / f"rd_curve_{metric.lower()}.png",
            compression_metric="Normalized_CR",
            distortion_metric=metric,
            agg="median",
            bound_names=bound_names,
        )


def rename_error_bounds(df, bound_names):
    """Give error bound consistent names between variables. By default the error bounds
    have the pattern {variable_name}-{bound_type}={bound_value}."""
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


def normalize(data, bound_normalize="mid"):
    """Generate normalized metrics for each compressor and variable. The normalization
    first computes the 'best compressor' with the highest average rank over all variables (ranked by
    compression ratio).

    For each metric, the normalization is done by dividing the metric by the value of the
    'best compressor' for the same variable and error bound, i.e.:
    normalized_metric = metric[compressor, variable] / metric[best_compressor, variable].
    """
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
        ("DSSIM", "Normalized_DSSIM"),
        ("Max Absolute Error", "Normalized_MaxAbsError"),
    ]
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


def plot_per_variable_metrics(
    basepath: Path, plots_path: Path, all_results: pd.DataFrame
):
    """Creates all the plots which only depend on a single variable."""
    for dataset in all_results["Dataset"].unique():
        df = all_results[all_results["Dataset"] == dataset]
        dataset_plots_path = plots_path / dataset
        dataset_plots_path.mkdir(parents=True, exist_ok=True)

        # For each variable and compressor, plot the input, output, and error fields.
        variables = df["Variable"].unique()
        for var in variables:
            for dist_metric in ["Max Absolute Error", "MAE"]:
                metric_name = dist_metric.lower().replace(" ", "_")
                if df[df["Variable"] == var][dist_metric].isnull().all():
                    continue
                plot_variable_rd_curve(
                    df[df["Variable"] == var],
                    dataset_plots_path / f"{var}_compression_ratio_{metric_name}.png",
                    distortion_metric=dist_metric,
                )

            error_bounds = df[df["Variable"] == var]["Error Bound"].unique()
            for err_bound in error_bounds:
                compressors = df[
                    (df["Variable"] == var) & (df["Error Bound"] == err_bound)
                ]["Compressor"].unique()

                err_bound_path = dataset_plots_path / err_bound
                err_bound_path.mkdir(parents=True, exist_ok=True)
                for comp in compressors:
                    print(f"Plotting {var} error for {comp}...")
                    plot_variable_error(
                        basepath,
                        dataset,
                        err_bound,
                        comp,
                        var,
                        err_bound_path / f"{var}_{comp}.png",
                    )


def plot_variable_error(basepath, dataset_name, error_bound, compressor, var, outfile):
    if outfile.exists():
        # These plots can be quite expensive to generate, so we skip if they already exist.
        return

    compressed = (
        basepath
        / ".."
        / "compressor"
        / "compressed-datasets"
        / dataset_name
        / error_bound
        / compressor
        / "decompressed.zarr"
    )
    input = (
        basepath
        / ".."
        / "data-loader"
        / "datasets"
        / dataset_name
        / "standardized.zarr"
    )

    ds = xr.open_dataset(input, chunks=dict(), engine="zarr").compute()
    ds_new = xr.open_dataset(compressed, chunks=dict(), engine="zarr").compute()
    ds, ds_new = ds[var], ds_new[var]

    plotter = PLOTTERS.get(dataset_name, None)
    if plotter:
        plotter().plot(ds, ds_new, dataset_name, compressor, var, outfile)
    else:
        print(f"No plotter found for dataset {dataset_name}")


def plot_variable_rd_curve(df, outfile, distortion_metric):
    plt.figure(figsize=(8, 6))
    compressors = df["Compressor"].unique()
    for comp in compressors:
        compressor_data = df[df["Compressor"] == comp]
        sorting_ixs = np.argsort(compressor_data["Compression Ratio [raw B / enc B]"])
        compr_ratio = [
            compressor_data["Compression Ratio [raw B / enc B]"].iloc[i]
            for i in sorting_ixs
        ]
        distortion = [compressor_data[distortion_metric].iloc[i] for i in sorting_ixs]
        plt.plot(
            compr_ratio,
            distortion,
            label=COMPRESSOR2LEGEND_NAME[comp],
            marker="s",
            color=COMPRESSOR2COLOR[comp],
            linewidth=4,
            markersize=8,
        )

    plt.xlabel("Compression Ratio [raw B / enc B]", fontsize=14)
    plt.xscale("log")
    if distortion_metric != "PSNR":
        # PSNR is already on log scale.
        plt.yscale("log")
    plt.ylabel(distortion_metric, fontsize=14)

    plt.legend(
        title="Compressor",
        fontsize=10,
        title_fontsize=12,
    )
    plt.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        length=12,
        direction="in",
        top=True,
        right=True,
    )
    plt.tick_params(
        axis="both",
        which="minor",
        length=6,
        direction="in",
        top=True,
        right=True,
    )

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()


def plot_aggregated_rd_curve(
    normalized_df,
    outfile,
    compression_metric,
    distortion_metric,
    agg="median",
    bound_names=["low", "mid", "high"],
):
    plt.figure(figsize=(8, 6))
    compressors = normalized_df["Compressor"].unique()
    agg_distortion = normalized_df.groupby(["Error Bound", "Compressor"])[
        [compression_metric, distortion_metric]
    ].agg(agg)
    for comp in compressors:
        compr_ratio = [
            agg_distortion.loc[(bound, comp), compression_metric]
            for bound in bound_names
        ]
        distortion = [
            agg_distortion.loc[(bound, comp), distortion_metric]
            for bound in bound_names
        ]
        plt.plot(
            compr_ratio,
            distortion,
            label=COMPRESSOR2LEGEND_NAME[comp],
            marker="s",
            color=COMPRESSOR2COLOR[comp],
            linewidth=4,
            markersize=8,
        )

    plt.xlabel(compression_metric, fontsize=14)
    plt.xscale("log")
    if "PSNR" not in distortion_metric:
        # PSNR is already on log scale.
        plt.yscale("log")
    plt.ylabel(distortion_metric, fontsize=14)

    plt.legend(
        title="Compressor",
        fontsize=10,
        title_fontsize=12,
    )
    plt.tick_params(
        axis="both",
        which="major",
        labelsize=14,
        length=12,
        direction="in",
        top=True,
        right=True,
    )
    plt.tick_params(
        axis="both",
        which="minor",
        length=6,
        direction="in",
        top=True,
        right=True,
    )

    if "MAE" in distortion_metric:
        plt.legend(
            title="Compressor",
            loc="upper right",
            bbox_to_anchor=(0.95, 0.6),
            fontsize=10,
            title_fontsize=12,
        )
        plt.xlabel("Normalized Compression Ratio", fontsize=14)
        plt.ylabel("Normalized Mean Absolute Error", fontsize=14)
        # Add an arrow pointing into the lower right corner
        plt.annotate(
            "",
            xy=(0.97, 0.05),
            xycoords="axes fraction",
            xytext=(-60, 50),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="-|>", color="grey", lw=2),
        )
        plt.text(
            0.85,
            0.08,
            "Better",
            transform=plt.gca().transAxes,
            fontsize=14,
            fontweight="bold",
            color="grey",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
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


if __name__ == "__main__":
    plot_metrics(basepath=Path())
