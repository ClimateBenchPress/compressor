import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from ..scripts.compute_metrics import parse_error_bounds
from .error_dist_plotter import ErrorDistPlotter
from .variable_plotters import PLOTTERS

_COMPRESSOR2LINEINFO = [
    ("jpeg2000", ("#EE7733", "-")),
    ("sperr", ("#117733", "-")),
    ("zfp-round", ("#DDAA33", "-")),
    ("zfp", ("#EE3377", "--")),
    ("sz3", ("#CC3311", "-")),
    ("bitround-pco", ("#0077BB", "-")),
    ("bitround", ("#33BBEE", "-")),
    ("stochround-pco", ("#BBBBBB", "--")),
    ("stochround", ("#009988", "--")),
    ("tthresh", ("#882255", "-.")),
    ("safeguarded-sperr", ("#117733", ":")),
    ("safeguarded-zfp-round", ("#DDAA33", ":")),
    ("safeguarded-sz3", ("#CC3311", ":")),
    ("safeguarded-zero-dssim", ("#9467BD", "--")),
    ("safeguarded-zero", ("#9467BD", ":")),
    ("safeguarded-bitround-pco", ("#0077BB", ":")),
]


def _get_lineinfo(compressor: str) -> tuple[str, str]:
    """Get the line color and style for a given compressor."""
    for comp, (color, linestyle) in _COMPRESSOR2LINEINFO:
        if compressor.startswith(comp):
            return color, linestyle
    raise ValueError(f"Unknown compressor: {compressor}")


_COMPRESSOR2LEGEND_NAME = [
    ("jpeg2000", "JPEG2000"),
    ("sperr", "SPERR"),
    ("zfp-round", "ZFP"),
    ("zfp", "ZFP"),
    ("sz3", "SZ3[v3.2]"),
    ("bitround-pco", "BitRound"),
    ("bitround", "BitRound + Zstd"),
    ("stochround-pco", "StochRound + PCO"),
    ("stochround", "StochRound + Zstd"),
    ("tthresh", "TTHRESH"),
    ("safeguarded-sperr", "Safeguarded(SPERR)"),
    ("safeguarded-zfp-round", "Safeguarded(ZFP)"),
    ("safeguarded-sz3", "Safeguarded(SZ3[v3.2])"),
    ("safeguarded-zero-dssim", "Safeguarded(0, dSSIM)"),
    ("safeguarded-zero", "Safeguarded(0)"),
    ("safeguarded-bitround-pco", "Safeguarded(BitRound)"),
]

_COMPRESSOR_ORDER = [
    "BitRound",
    "Safeguarded(BitRound)",
    "ZFP",
    "Safeguarded(ZFP)",
    "SZ3[v3.2]",
    "Safeguarded(SZ3[v3.2])",
    "SPERR",
    "Safeguarded(SPERR)",
    "Safeguarded(0)",
    "Safeguarded(0, dSSIM)",
]

DISTORTION2LEGEND_NAME = {
    "Relative MAE": "Mean Absolute Error",
    "Relative DSSIM": "DSSIM",
    "Relative MaxAbsError": "Max Absolute Error",
    "Spectral Error": "Spectral Error",
}


def _get_legend_name(compressor: str) -> str:
    """Get the legend name for a given compressor."""
    for comp, name in _COMPRESSOR2LEGEND_NAME:
        if compressor.startswith(comp):
            return name

    return compressor  # Fallback to the compressor name if not found in the mapping.


def plot_metrics(
    basepath: Path = Path(),
    data_loader_basepath: None | Path = None,
    bound_names: list[str] = ["low", "mid", "high"],
    exclude_dataset: list[str] = [],
    exclude_compressor: list[str] = [],
    tiny_datasets: bool = False,
    chunked_datasets: bool = False,
    use_latex: bool = True,
):
    """Create diagnostic plots for the metrics computed by the compressors.

    Parameters
    ----------
    basepath: Path
        Assumes compressed datasets are stored in `basepath / compressed-datasets`
        and metrics in `basepath / metrics`.
    data_loader_basepath: Path | None
        Assumes datasets are stored in `data_loader_basepath / datasets`.
        If None, uses `basepath / datasets`.
    bound_names: list[str]
        Names of the error bounds to use for plotting.
    exclude_dataset: list[str]
        List of dataset names to exclude from the plotting.
    exclude_compressor: list[str]
        List of compressor names to exclude from the plotting.
    tiny_datasets: bool
        If True, only plot the tiny datasets. Defaults to False.
    use_latex: bool
        If True, use LaTeX for rendering text in the plots. Defaults to True.
    """
    metrics_path = basepath / "metrics"
    plots_path = basepath / "plots"
    datasets = (data_loader_basepath or basepath) / "datasets"
    compressed_datasets = basepath / "compressed-datasets"

    df = pd.read_csv(metrics_path / "all_results.csv")

    # Filter out excluded datasets and compressors
    # bitround jpeg2000-conservative-abs stochround-conservative-abs stochround-pco-conservative-abs zfp-conservative-abs bitround-conservative-rel stochround-pco stochround zfp jpeg2000
    df = df[~df["Compressor"].isin(exclude_compressor)]
    df = df[~df["Dataset"].isin(exclude_dataset)]
    is_tiny = df["Dataset"].str.endswith("-tiny")
    filter_tiny = is_tiny if tiny_datasets else ~is_tiny
    df = df[filter_tiny]
    is_chunked = df["Dataset"].str.endswith("-chunked")
    filter_chunked = is_chunked if chunked_datasets else ~is_chunked
    df = df[filter_chunked]

    # _plot_per_variable_metrics(
    #     datasets=datasets,
    #     compressed_datasets=compressed_datasets,
    #     plots_path=plots_path,
    #     all_results=df,
    #     rd_curves_metrics=["Max Absolute Error", "MAE", "DSSIM", "Spectral Error"],
    # )

    df = _rename_compressors(df)
    normalized_df = _normalize(df)
    _plot_bound_violations(
        normalized_df, bound_names, plots_path / "bound_violations.pdf"
    )
    _plot_throughput(df, plots_path / "throughput.pdf")
    _plot_instruction_count(df, plots_path / "instruction_count.pdf")

    for metric in [
        "Relative MAE",
        "Relative DSSIM",
        "Relative MaxAbsError",
        "Relative SpectralError",
    ]:
        with plt.rc_context(rc={"text.usetex": use_latex}):
            _plot_aggregated_rd_curve(
                normalized_df,
                compression_metric="Relative CR",
                distortion_metric=metric,
                outfile=plots_path / f"rd_curve_{metric.lower().replace(' ', '_')}.pdf",
                agg="mean",
                bound_names=bound_names,
            )

            _plot_aggregated_rd_curve(
                normalized_df,
                compression_metric="Relative CR",
                distortion_metric=metric,
                outfile=plots_path
                / f"full_rd_curve_{metric.lower().replace(' ', '_')}.pdf",
                agg="mean",
                bound_names=bound_names,
                remove_outliers=False,
            )


def _rename_compressors(df):
    """Give compressors consistent names. They sometimes have suffixes if they are
    applied on a converted error bound. The three patterns are:
    - {compressor_name}
    - {compressor_name}-conservative-abs
    - {compressor_name}-conservative-rel
    """
    df = df.copy()
    df["Compressor"] = df["Compressor"].str.replace(
        r"-(conservative-(abs|rel))$", "", regex=True
    )
    return df


def _sort_error_bounds(error_bounds: list[str]) -> list[str]:
    """Each error bound has the format

    {variable_name}-{bound_type}={bound_value}_{variable_name2}-{bound_type2}={bound_value2}

    for 1 or more variables. This function takes a list of error bounds and sorts them
    according to their {bound_value} i.e. in ascending order of the first bound value.
    """
    return sorted(
        error_bounds,
        key=lambda x: float(x.split("=")[1].split("_")[0]),
    )


def _normalize(data):
    normalized = data.copy()
    normalize_vars = [
        ("Compression Ratio [raw B / enc B]", "Relative CR"),
        ("MAE", "Relative MAE"),
        ("DSSIM", "Relative DSSIM"),
        ("Max Absolute Error", "Relative MaxAbsError"),
        ("Spectral Error", "Relative SpectralError"),
    ]

    variables = normalized["Variable"].unique()

    dssim_unreliable = normalized["Variable"].isin(["ta", "tos"])
    normalized.loc[dssim_unreliable, "DSSIM"] = np.nan

    for col, new_col in normalize_vars:
        mean_std = dict()
        for var in variables:
            mean = normalized[normalized["Variable"] == var][col].mean()
            std = normalized[normalized["Variable"] == var][col].std()
            mean_std[var] = (mean, std)

        # Normalize each variable by its mean and std
        normalized[new_col] = normalized.apply(
            lambda x: (
                (x[col] - mean_std[x["Variable"]][0]) / mean_std[x["Variable"]][1]
            ),
            axis=1,
        )

    return normalized


def _plot_per_variable_metrics(
    datasets: Path,
    compressed_datasets: Path,
    plots_path: Path,
    all_results: pd.DataFrame,
    rd_curves_metrics: list[str] = ["Max Absolute Error", "MAE"],
):
    """Creates all the plots which only depend on a single variable."""
    for dataset in all_results["Dataset"].unique():
        df = all_results[all_results["Dataset"] == dataset]
        dataset_plots_path = plots_path / dataset
        dataset_plots_path.mkdir(parents=True, exist_ok=True)

        # For each variable and compressor, plot the input, output, and error fields.
        variables = df["Variable"].unique()
        for var in variables:
            for dist_metric in rd_curves_metrics:
                metric_name = dist_metric.lower().replace(" ", "_")
                if df[df["Variable"] == var][dist_metric].isnull().all():
                    continue
                _plot_variable_rd_curve(
                    df[df["Variable"] == var],
                    distortion_metric=dist_metric,
                    outfile=dataset_plots_path
                    / f"{var}_compression_ratio_{metric_name}.pdf",
                )

        error_bounds = df[df["Dataset"] == dataset]["Error Bound"].unique()
        error_bounds = _sort_error_bounds(error_bounds)

        error_dist_plotter = ErrorDistPlotter(
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
                input_dataset_name = dataset
                if dataset.endswith("-chunked"):
                    input_dataset_name = dataset.removesuffix("-chunked")
                input = datasets / input_dataset_name / "standardized.zarr"

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

                    _plot_variable_error(
                        ds[var],
                        ds_new[var],
                        dataset,
                        comp,
                        var,
                        error_bound_vals[var],
                        outfile=err_bound_path / f"{var}_{comp}.png",
                    )

            error_dist_plotter.plot_error_bound_histograms(
                i,
                variables,
                compressors,
                error_bound_vals,
                _get_legend_name,
                _get_lineinfo,
            )

        figs, _ = error_dist_plotter.get_final_figure()
        for var, fig in figs.items():
            _savefig(
                dataset_plots_path / f"error_histograms_{dataset}_{var}.pdf", fig=fig
            )
            plt.close(fig)


def _plot_variable_error(
    uncompressed_data: xr.DataArray,
    compressed_data: xr.DataArray,
    dataset_name: str,
    compressor: str,
    var: str,
    err_bound: tuple[str, float],
    outfile: None | Path = None,
):
    if outfile is not None and outfile.exists():
        # These plots can be quite expensive to generate, so we skip if they already exist.
        return

    dataset_name = dataset_name.removesuffix("-chunked")
    plotter = PLOTTERS.get(dataset_name, None)
    if plotter:
        plotter().plot(
            uncompressed_data,
            compressed_data,
            dataset_name,
            compressor,
            var,
            err_bound,
            outfile,
        )
    else:
        print(f"No plotter found for dataset {dataset_name}")


def _plot_variable_rd_curve(
    df, distortion_metric, bounds=["low", "mid", "high"], outfile: None | Path = None
):
    plt.figure(figsize=(8, 6))
    compressors = df["Compressor"].unique()
    for comp in compressors:
        compressor_data = df[df["Compressor"] == comp]
        assert len(compressor_data) == len(bounds)
        bound_ixs = [
            compressor_data[compressor_data["Error Bound Name"] == bound].index[0]
            for bound in bounds
        ]
        compr_ratio = [
            compressor_data["Compression Ratio [raw B / enc B]"].loc[i]
            for i in bound_ixs
        ]
        distortion = [compressor_data[distortion_metric].loc[i] for i in bound_ixs]
        color, linestyle = _get_lineinfo(comp)
        plt.plot(
            compr_ratio,
            distortion,
            label=_get_legend_name(comp),
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
        _savefig(outfile)
    plt.close()


def _plot_aggregated_rd_curve(
    normalized_df,
    compression_metric,
    distortion_metric,
    outfile: None | Path = None,
    agg="median",
    bound_names=["low", "mid", "high"],
    exclude_vars=None,
    remove_outliers=True,
):
    plt.figure(figsize=(8, 6))
    if exclude_vars:
        # Exclude variables that are not relevant for the distortion metric.
        normalized_df = normalized_df[~normalized_df["Variable"].isin(exclude_vars)]

    compressors = sorted(
        normalized_df["Compressor"].unique(),
        key=lambda k: _COMPRESSOR_ORDER.index(_get_legend_name(k)),
    )
    agg_distortion = normalized_df.groupby(["Error Bound Name", "Compressor"])[
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
        color, linestyle = _get_lineinfo(comp)
        plt.plot(
            compr_ratio,
            distortion,
            label=_get_legend_name(comp),
            marker="s",
            color=color,
            linestyle=linestyle,
            linewidth=4,
            markersize=8,
        )

    if remove_outliers:
        # SZ3 and JPEG2000 often give outlier values and violate the bounds.
        exclude_compressors = [
            "sz3",
            "jpeg2000",
            "safeguarded-zero-dssim",
            "safeguarded-zero",
            "safeguarded-sz3",
        ]
        filtered_agg = agg_distortion[
            ~agg_distortion.index.get_level_values("Compressor").isin(
                exclude_compressors
            )
        ]
        cr_mean, cr_std = (
            filtered_agg[compression_metric].mean(),
            filtered_agg[compression_metric].std(),
        )
        distortion_mean, distortion_std = (
            filtered_agg[distortion_metric].mean(),
            filtered_agg[distortion_metric].std(),
        )

        # Adjust the plot limits
        xlims = plt.xlim()
        xlims_min = max(xlims[0], cr_mean - 4 * cr_std)
        xlims_max = min(xlims[1], cr_mean + 4 * cr_std)
        plt.xlim(xlims_min, xlims_max)
        ylims = plt.ylim()
        ylims_min = max(ylims[0], distortion_mean - 4 * distortion_std)
        ylims_max = min(ylims[1], distortion_mean + 4 * distortion_std)
        plt.ylim(ylims_min, ylims_max)

    plt.xlabel(f"{agg.title()} {compression_metric}", fontsize=14)
    plt.ylabel(f"{agg.title()} {distortion_metric}", fontsize=14)

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
    plt.xlabel(
        r"Mean Normalized Compression Ratio ($\uparrow$)",
        fontsize=16,
    )
    metric_name = DISTORTION2LEGEND_NAME.get(distortion_metric, distortion_metric)
    plt.ylabel(
        rf"Mean Normalized {metric_name} ($\downarrow$)",
        fontsize=16,
    )
    plt.legend(
        title="Compressor",
        loc="upper left",
        # bbox_to_anchor=(0.8, 0.99),
        fontsize=12,
        title_fontsize=14,
    )

    arrow_color = "black"
    if "DSSIM" in distortion_metric:
        # Add an arrow pointing into the top right corner
        plt.annotate(
            "",
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            xytext=(-60, -50),
            textcoords="offset points",
            arrowprops=dict(
                arrowstyle="-|>, head_length=0.5, head_width=0.5",
                color=arrow_color,
                lw=5,
            ),
        )
        # Attach the text to the lower left of the arrow
        plt.text(
            0.83,
            0.92,
            "Better",
            transform=plt.gca().transAxes,
            fontsize=16,
            fontweight="bold",
            color=arrow_color,
            ha="center",
            va="center",
        )
        # Correct the y-label to point upwards
        plt.ylabel(
            rf"Mean Normalized {metric_name} ($\uparrow$)",
            fontsize=16,
        )
    else:
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
    if (
        "DSSIM" in distortion_metric
        or "MaxAbsError" in distortion_metric
        or "SpectralError" in distortion_metric
    ):
        plt.legend().remove()

    plt.tight_layout()
    if outfile is not None:
        _savefig(outfile)
    plt.close()


def _plot_throughput(df, outfile: None | Path = None):
    # Transform throughput measurements from raw B/s to s/MB for better comparison
    # with instruction count measurements.
    encode_col = "Encode Throughput [raw B / s]"
    decode_col = "Decode Throughput [raw B / s]"
    new_df = df[["Compressor", "Error Bound Name", encode_col, decode_col]].copy()
    transformed_encode_col = "Encode Throughput [s / MB]"
    transformed_decode_col = "Decode Throughput [s / MB]"
    new_df[transformed_encode_col] = 1e6 / new_df[encode_col]
    new_df[transformed_decode_col] = 1e6 / new_df[decode_col]
    encode_col, decode_col = transformed_encode_col, transformed_decode_col

    grouped_df = _get_median_and_quantiles(new_df, encode_col, decode_col)
    _plot_grouped_df(
        grouped_df,
        title="",
        ylabel="Throughput [s / MB]",
        logy=True,
        outfile=outfile,
    )


def _plot_instruction_count(df, outfile: None | Path = None):
    encode_col = "Encode Instructions [# / raw B]"
    decode_col = "Decode Instructions [# / raw B]"
    grouped_df = _get_median_and_quantiles(df, encode_col, decode_col)
    _plot_grouped_df(
        grouped_df,
        title="",
        ylabel="Instructions [# / raw B]",
        logy=True,
        outfile=outfile,
    )


def _get_median_and_quantiles(df, encode_column, decode_column):
    return (
        df.groupby(["Compressor", "Error Bound Name"])[[encode_column, decode_column]]
        .agg(
            encode_median=pd.NamedAgg(
                column=encode_column, aggfunc=lambda x: x.quantile(0.5)
            ),
            encode_lower_quantile=pd.NamedAgg(
                column=encode_column, aggfunc=lambda x: x.quantile(0.25)
            ),
            encode_upper_quantile=pd.NamedAgg(
                column=encode_column, aggfunc=lambda x: x.quantile(0.75)
            ),
            decode_median=pd.NamedAgg(
                column=decode_column, aggfunc=lambda x: x.quantile(0.5)
            ),
            decode_lower_quantile=pd.NamedAgg(
                column=decode_column, aggfunc=lambda x: x.quantile(0.25)
            ),
            decode_upper_quantile=pd.NamedAgg(
                column=decode_column, aggfunc=lambda x: x.quantile(0.75)
            ),
        )
        .sort_index(
            level=0,
            key=lambda ks: [_COMPRESSOR_ORDER.index(_get_legend_name(k)) for k in ks],
        )
    )


def _plot_grouped_df(
    grouped_df, title, ylabel, outfile: None | Path = None, logy=False
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # Bar width
    bar_width = 0.35
    compressors = sorted(
        grouped_df.index.levels[0].tolist(),
        key=lambda k: _COMPRESSOR_ORDER.index(_get_legend_name(k)),
    )
    x_labels = [_get_legend_name(c) for c in compressors]
    x_positions = range(len(x_labels))

    error_bounds = ["low", "mid", "high"]

    for i, error_bound in enumerate(error_bounds):
        ax = axes[i]
        bound_data = grouped_df.xs(error_bound, level="Error Bound Name").sort_index(
            level=0,
            key=lambda ks: [_COMPRESSOR_ORDER.index(_get_legend_name(k)) for k in ks],
        )

        # Plot encode throughput
        ax.bar(
            x_positions,
            bound_data["encode_median"],
            bar_width,
            yerr=[
                bound_data["encode_lower_quantile"],
                bound_data["encode_upper_quantile"],
            ],
            label="Encoding",
            color=[_get_lineinfo(comp)[0] for comp in compressors],
        )

        # Plot decode throughput
        ax.bar(
            [p + bar_width for p in x_positions],
            bound_data["decode_median"],
            bar_width,
            yerr=[
                bound_data["decode_lower_quantile"],
                bound_data["decode_upper_quantile"],
            ],
            label="Decoding",
            edgecolor=[_get_lineinfo(comp)[0] for comp in compressors],
            fill=False,
            linewidth=4,
        )

        # Add labels and title
        ax.set_xticks([p + bar_width / 2 for p in x_positions])
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=14)
        ax.set_yscale("log" if logy else "linear")
        ax.set_title(f"{error_bound.capitalize()} Error Bound", fontsize=14)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        if i == 0:
            ax.legend(fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.annotate(
                "Better",
                xy=(0.1, 0.8),
                xycoords="axes fraction",
                xytext=(0.1, 0.95),
                textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=3, color="black"),
                fontsize=14,
                ha="center",
                va="bottom",
            )

    fig.suptitle(title)

    fig.tight_layout()
    if outfile is not None:
        _savefig(outfile)
    plt.close()


def _plot_bound_violations(df, bound_names, outfile: None | Path = None):
    fig, axs = plt.subplots(1, 3, figsize=(len(bound_names) * 6, 6), sharey=True)

    for i, bound_name in enumerate(bound_names):
        df_bound = df[df["Error Bound Name"] == bound_name].copy()
        df_bound["Compressor"] = df_bound["Compressor"].map(_get_legend_name)
        pass_fail = df_bound.pivot(
            index="Compressor", columns="Variable", values="Satisfies Bound (Passed)"
        ).sort_index(key=lambda ks: [_COMPRESSOR_ORDER.index(k) for k in ks])
        pass_fail = pass_fail.astype(np.float32)
        fraction_fail = df_bound.pivot(
            index="Compressor", columns="Variable", values="Satisfies Bound (Value)"
        ).sort_index(key=lambda ks: [_COMPRESSOR_ORDER.index(k) for k in ks])
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
            alpha=0.8,
            ax=axs[i],
            annot_kws={"size": 12},  # Adjust annotation font size
        )
        axs[i].set_xticklabels(
            axs[i].get_xticklabels(), rotation=45, ha="right", fontsize=12
        )
        axs[i].tick_params(axis="y", labelsize=12)  # Adjust y-axis label font size
        axs[i].set_title(f"Bound: {bound_name}", fontsize=14)  # Adjust title font size
        if i != 0:
            axs[i].set_ylabel("")

    fig.tight_layout()
    if outfile is not None:
        _savefig(outfile)
    plt.close()


def _savefig(outfile: Path, fig=None):
    ispdf = outfile.suffix == ".pdf"
    fig = fig if fig is not None else plt.gcf()
    if ispdf:
        # Saving a PDF with the alternative code below leads to a corrupted file.
        # Hence, we use the default savefig method.
        # NOTE: This means passing a virtual UPath is only supported for non-PDF files.
        fig.savefig(outfile, dpi=300)
    else:
        with outfile.open("wb") as f:
            fig.savefig(f, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-dataset", type=str, nargs="+", default=[])
    parser.add_argument("--exclude-compressor", type=str, nargs="+", default=[])
    parser.add_argument("--tiny-datasets", action="store_true", default=False)
    parser.add_argument("--avoid-latex", action="store_true", default=False)
    parser.add_argument("--basepath", type=Path, default=Path())
    parser.add_argument(
        "--data-loader-basepath",
        type=Path,
        default=Path() / ".." / "data-loader",
    )
    args = parser.parse_args()

    plot_metrics(
        basepath=args.basepath,
        data_loader_basepath=args.data_loader_basepath,
        exclude_compressor=args.exclude_compressor,
        exclude_dataset=args.exclude_dataset,
        tiny_datasets=args.tiny_datasets,
        use_latex=(not args.avoid_latex),
    )
