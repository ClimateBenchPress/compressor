import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from ..scripts.collect_metrics import parse_error_bounds
from .error_dist_plotter import ErrorDistPlotter
from .variable_plotters import PLOTTERS

COMPRESSOR2LINEINFO = {
    "jpeg2000": ("#EE7733", "-"),
    "sperr": ("#000000", ":"),
    "zfp": ("#EE3377", "--"),
    "zfp-round": ("#DDAA33", "--"),
    "sz3": ("#CC3311", "-."),
    "bitround-pco-conservative-rel": ("#0077BB", ":"),
    "bitround-conservative-rel": ("#33BBEE", "-"),
    "stochround": ("#009988", "--"),
    "stochround-pco": ("#BBBBBB", "--"),
    "tthresh": ("#882255", "-."),
}

COMPRESSOR2LEGEND_NAME = {
    "jpeg2000": "JPEG2000",
    "sperr": "SPERR",
    "zfp": "ZFP",
    "zfp-round": "ZFP-ROUND",
    "sz3": "SZ3",
    "bitround-pco-conservative-rel": "BitRound + PCO",
    "bitround-conservative-rel": "BitRound + Zstd",
    "stochround": "StochRound + Zstd",
    "stochround-pco": "StochRound + PCO",
    "tthresh": "TTHRESH",
}


def plot_metrics(
    basepath: Path = Path(),
    data_loader_base_path: None | Path = None,
    bound_names: list[str] = ["low", "mid", "high"],
    normalizer: str = "sz3",
    exclude_dataset: list[str] = [],
    exclude_compressor: list[str] = [],
    tiny_datasets: bool = False,
):
    metrics_path = basepath / "metrics"
    plots_path = basepath / "plots"
    datasets = (data_loader_base_path or basepath) / "datasets"
    compressed_datasets = basepath / "compressed-datasets"

    df = pd.read_csv(metrics_path / "all_results.csv")

    # Filter out excluded datasets and compressors
    df = df[~df["Compressor"].isin(exclude_compressor)]
    df = df[~df["Dataset"].isin(exclude_dataset)]
    is_tiny = df["Dataset"].str.endswith("-tiny")
    filter_tiny = is_tiny if tiny_datasets else ~is_tiny
    df = df[filter_tiny]

    plot_per_variable_metrics(
        datasets=datasets,
        compressed_datasets=compressed_datasets,
        plots_path=plots_path,
        all_results=df,
    )

    df = rename_error_bounds(df, bound_names)
    plot_throughput(df, plots_path / "throughput.pdf")

    normalized_df = normalize(df, bound_normalize="mid", normalizer=normalizer)
    plot_bound_violations(
        normalized_df, bound_names, plots_path / "bound_violations.pdf"
    )

    for metric in ["Relative MAE", "Relative DSSIM", "Relative MaxAbsError"]:
        with plt.rc_context(rc={"text.usetex": True}):
            plot_aggregated_rd_curve(
                normalized_df,
                normalizer=normalizer,
                compression_metric="Relative CR",
                distortion_metric=metric,
                outfile=plots_path / f"rd_curve_{metric.lower().replace(' ', '_')}.pdf",
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

        assert len(error_bounds) == len(bound_names), (
            f"Number of error bounds {len(error_bounds)} does not match number of bound names {len(bound_names)} for {variable}."
        )

        for i in range(len(error_bounds)):
            bound_selector = var_data["Error Bound"] == error_bounds[i]
            df.loc[bound_selector & var_selector, "Error Bound"] = bound_names[i]

    return df


def normalize(data, bound_normalize="mid", normalizer=None):
    """Generate normalized metrics for each compressor and variable. The normalization
    is done either with respect to either a user provided compressor or the
    compressor with the highest average rank over all variables (ranked by
    compression ratio).

    For each metric, the normalization is done by dividing the metric by the value of the
    normalizer for the same variable and error bound, i.e.:
    normalized_metric = metric[compressor, variable] / metric[normalizer, variable].
    """
    if normalizer is None:
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

        normalizer = avg_ranks.iloc[0]["Compressor"]

    normalized = data.copy()
    normalize_vars = [
        ("Compression Ratio [raw B / enc B]", "Relative CR"),
        ("MAE", "Relative MAE"),
        ("DSSIM", "Relative DSSIM"),
        ("Max Absolute Error", "Relative MaxAbsError"),
    ]
    # Avoid negative values. By default, DSSIM is in the range [-1, 1].
    normalized["DSSIM"] = normalized["DSSIM"] + 1.0

    def get_normalizer(row):
        return normalized[
            (data["Compressor"] == normalizer)
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
    datasets: Path,
    compressed_datasets: Path,
    plots_path: Path,
    all_results: pd.DataFrame,
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
                    distortion_metric=dist_metric,
                    outfile=dataset_plots_path
                    / f"{var}_compression_ratio_{metric_name}.pdf",
                )

        error_bounds = df[df["Dataset"] == dataset]["Error Bound"].unique()
        error_dist_plotter = ErrorDistPlotter(
            dataset=dataset,
            variables=variables,
            error_bounds=error_bounds,
        )
        for i, err_bound in enumerate(error_bounds):
            compressors = df[(df["Error Bound"] == err_bound)]["Compressor"].unique()

            err_bound_path = dataset_plots_path / err_bound
            err_bound_path.mkdir(parents=True, exist_ok=True)

            error_bound_vals = parse_error_bounds(err_bound)
            for comp in compressors:
                compressed = (
                    compressed_datasets
                    / dataset
                    / err_bound
                    / comp
                    / "decompressed.zarr"
                )
                input = datasets / dataset / "standardized.zarr"

                ds = xr.open_dataset(input, chunks=dict(), engine="zarr")
                ds_new = xr.open_dataset(compressed, chunks=dict(), engine="zarr")

                for var in variables:
                    print(f"Plotting {var} error for {comp}...")
                    error_dist_plotter.compute_errors(
                        comp,
                        ds,
                        ds_new,
                        var,
                        error_bound_vals[var][0],
                    )

                    plot_variable_error(
                        ds[var],
                        ds_new[var],
                        dataset,
                        comp,
                        var,
                        outfile=err_bound_path / f"{var}_{comp}.png",
                    )

            error_dist_plotter.plot_error_bound_histograms(
                i,
                variables,
                compressors,
                error_bound_vals,
                COMPRESSOR2LEGEND_NAME,
                COMPRESSOR2LINEINFO,
            )

        fig, _ = error_dist_plotter.get_final_figure()
        savefig(dataset_plots_path / f"error_histograms_{dataset}.pdf")
        plt.close(fig)


def plot_variable_error(
    uncompressed_data: xr.DataArray,
    compressed_data: xr.DataArray,
    dataset_name: str,
    compressor: str,
    var: str,
    outfile: None | Path = None,
):
    if outfile is not None and outfile.exists():
        # These plots can be quite expensive to generate, so we skip if they already exist.
        return

    plotter = PLOTTERS.get(dataset_name, None)
    if plotter:
        plotter().plot(
            uncompressed_data, compressed_data, dataset_name, compressor, var, outfile
        )
    else:
        print(f"No plotter found for dataset {dataset_name}")


def plot_variable_rd_curve(df, distortion_metric, outfile: None | Path = None):
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
        color, linestyle = COMPRESSOR2LINEINFO[comp]
        plt.plot(
            compr_ratio,
            distortion,
            label=COMPRESSOR2LEGEND_NAME[comp],
            marker="s",
            color=color,
            linestyle=linestyle,
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
    if outfile is not None:
        savefig(outfile)
    plt.close()


def plot_aggregated_rd_curve(
    normalized_df,
    normalizer,
    compression_metric,
    distortion_metric,
    outfile: None | Path = None,
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
        color, linestyle = COMPRESSOR2LINEINFO[comp]
        plt.plot(
            compr_ratio,
            distortion,
            label=COMPRESSOR2LEGEND_NAME[comp],
            marker="s",
            color=color,
            linestyle=linestyle,
            linewidth=4,
            markersize=8,
        )

    plt.xlabel(f"{agg.title()} {compression_metric}", fontsize=14)
    plt.xscale("log")
    if "PSNR" not in distortion_metric:
        # PSNR is already on log scale.
        plt.yscale("log")
    plt.ylabel(f"{agg.title()} {distortion_metric}", fontsize=14)

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
            bbox_to_anchor=(0.95, 0.7),
            fontsize=12,
            title_fontsize=14,
        )
        normalizer_label = COMPRESSOR2LEGEND_NAME.get(normalizer, normalizer)
        plt.xlabel(
            rf"Median Compression Ratio Relative to {normalizer_label} ($\uparrow$)",
            fontsize=16,
        )
        plt.ylabel(
            rf"Median Mean Absolute Error Relative to {normalizer_label} ($\downarrow$)",
            fontsize=16,
        )
        arrow_color = "black"
        # Add an arrow pointing into the lower right corner
        plt.annotate(
            "",
            xy=(0.95, 0.05),
            xycoords="axes fraction",
            xytext=(-60, 50),
            textcoords="offset points",
            arrowprops=dict(
                arrowstyle="-|>, head_length=0.5, head_width=0.5",
                color=arrow_color,
                lw=5,
            ),
        )
        plt.text(
            0.83,
            0.08,
            "Better",
            transform=plt.gca().transAxes,
            fontsize=16,
            fontweight="bold",
            color=arrow_color,
            ha="center",
        )

    plt.tight_layout()
    if outfile is not None:
        savefig(outfile)
    plt.close()


def plot_throughput(df, outfile: None | Path = None):
    grouped_df = df.groupby(["Compressor", "Error Bound"])[
        ["Encode Throughput [raw B / s]", "Decode Throughput [raw B / s]"]
    ].agg(["mean", "std"])
    grouped_df["Encode Throughput [raw B / s]"] /= 1e6
    grouped_df["Decode Throughput [raw B / s]"] /= 1e6
    grouped_df.rename(
        columns={
            "Encode Throughput [raw B / s]": "Encode Throughput [MB / s]",
            "Decode Throughput [raw B / s]": "Decode Throughput [MB / s]",
        },
        inplace=True,
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 18), sharex=True, sharey=True)

    # Bar width
    bar_width = 0.35
    compressors = grouped_df.index.levels[0].tolist()
    x_labels = [COMPRESSOR2LEGEND_NAME[c] for c in compressors]
    x_positions = range(len(x_labels))

    error_bounds = ["low", "mid", "high"]

    for i, error_bound in enumerate(error_bounds):
        ax = axes[i]
        bound_data = grouped_df.xs(error_bound, level="Error Bound")

        # Plot encode throughput
        ax.bar(
            x_positions,
            bound_data[("Encode Throughput [MB / s]", "mean")],
            bar_width,
            label="Encoding",
            color=[COMPRESSOR2LINEINFO[comp][0] for comp in compressors],
        )

        # Plot decode throughput
        ax.bar(
            [p + bar_width for p in x_positions],
            bound_data[("Decode Throughput [MB / s]", "mean")],
            bar_width,
            label="Decoding",
            edgecolor=[COMPRESSOR2LINEINFO[comp][0] for comp in compressors],
            fill=False,
            linewidth=4,
        )

        # Add labels and title
        ax.set_xticks([p + bar_width / 2 for p in x_positions])
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("Throughput (MB/s)")
        ax.set_title(
            f"Mean Encode and Decode Throughput ({error_bound.capitalize()} Error Bound)"
        )
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if i == 0:
            ax.legend()

    fig.tight_layout()
    if outfile is not None:
        savefig(outfile)
    plt.close()


def plot_bound_violations(df, bound_names, outfile: None | Path = None):
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
    if outfile is not None:
        savefig(outfile)
    plt.close()


def savefig(outfile: Path):
    ispdf = outfile.suffix == ".pdf"
    if ispdf:
        # Saving a PDF with the alternative code below leads to a corrupted file.
        # Hence, we use the default savefig method.
        # NOTE: This means passing a virtual UPath is only supported for non-PDF files.
        plt.savefig(outfile, dpi=300)
    else:
        with outfile.open("wb") as f:
            plt.savefig(f, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dataset", type=str, nargs="+", default=[])
    parser.add_argument("--exclude-compressor", type=str, nargs="+", default=[])
    parser.add_argument("--tiny-datasets", action="store_true", default=False)
    args = parser.parse_args()

    plot_metrics(
        basepath=Path(),
        data_loader_base_path=Path() / ".." / "data-loader",
        exclude_compressor=args.exclude_compressor,
        exclude_dataset=args.exclude_dataset,
        tiny_datasets=args.tiny_datasets,
    )
