"""Scorecard figures summarizing the benchmark results.

For each error bound level one scorecard is emitted, split into two rows of
metrics. Each cell shows the raw metric value, coloured by its relative
difference to a reference compressor.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .constants import _get_compressor_legend_name

METRICS2NAME = {
    "MAE": "Mean Absolute Error",
    "Spatial Relative Error (Value)": "SRE",
    "Compression Ratio [raw B / enc B]": "Compression Ratio",
    "Satisfies Bound (Value)": r"% of Pixels Exceeding Error Bound",
}

VARIABLE2NAME = {
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "msl",
}

HIGHER_BETTER_METRICS = ["DSSIM", "Compression Ratio [raw B / enc B]"]

# DSSIM and Spectral Error are unreliable for variables with large NaN regions.
UNRELIABLE_NAN_METRICS = {"DSSIM", "Spectral Error"}
UNRELIABLE_NAN_VARIABLES = {"ta", "tos"}

_CONSERVATIVE_SUFFIXES = ("-conservative-abs", "-conservative-rel")


def converted_bound_cells(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Find the (compressor, variable) pairs whose error bound had to be converted.

    Must be called on the raw results, i.e. before `_rename_compressors` strips the
    `-conservative-abs` / `-conservative-rel` suffixes which encode the conversion.
    The returned compressor names are the stripped ones, so they match the renamed
    frame the scorecards are built from.
    """
    converted = set()
    for compressor, variable in zip(df["Compressor"], df["Variable"]):
        for suffix in _CONSERVATIVE_SUFFIXES:
            if compressor.endswith(suffix):
                converted.add((compressor.removesuffix(suffix), variable))
    return converted


def _create_data_matrix(
    df: pd.DataFrame,
    error_bound: str,
    metrics: list[str],
    ref_compressor: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    df_filtered = df[df["Error Bound Name"] == error_bound].copy()
    # Convert to percentage.
    df_filtered["Satisfies Bound (Value)"] = (
        df_filtered["Satisfies Bound (Value)"] * 100
    )

    variables = sorted(df_filtered["Variable"].unique())
    compressors = sorted(df_filtered["Compressor"].unique())
    compressors = [ref_compressor] + [c for c in compressors if c != ref_compressor]

    column_labels = [f"{v}\n{m}" for m in metrics for v in variables]
    data_matrix = np.full((len(compressors), len(column_labels)), np.nan)

    for i, compressor in enumerate(compressors):
        for j, metric in enumerate(metrics):
            for k, variable in enumerate(variables):
                subset = df_filtered[
                    (df_filtered["Compressor"] == compressor)
                    & (df_filtered["Variable"] == variable)
                ]
                if subset.empty:
                    print(f"No data for Compressor: {compressor}, Variable: {variable}")
                    continue
                if (
                    metric in UNRELIABLE_NAN_METRICS
                    and variable in UNRELIABLE_NAN_VARIABLES
                ):
                    continue

                col_idx = j * len(variables) + k
                if metric in subset.columns:
                    values = subset[metric]
                    if len(values) == 1:
                        data_matrix[i, col_idx] = values.iloc[0]

    return data_matrix, compressors, variables


def _create_compression_scorecard(
    data_matrix: np.ndarray,
    compressors: list[str],
    variables: list[str],
    metrics: list[str],
    converted_cells: set[tuple[str, str]],
    cbar: bool = True,
    ref_compressor: str = "bitround",
    higher_better_metrics: list[str] = HIGHER_BETTER_METRICS,
    save_fn: str | Path | None = None,
):
    """Create a scorecard plot of relative metric differences vs a reference."""
    ref_idx = compressors.index(ref_compressor)
    ref_values = data_matrix[ref_idx, :]

    relative_matrix = np.full_like(data_matrix, np.nan)
    for i in range(len(compressors)):
        for j in range(data_matrix.shape[1]):
            if np.isnan(data_matrix[i, j]) or np.isnan(ref_values[j]):
                continue
            ref_val = np.abs(ref_values[j])
            if ref_val == 0.0:
                ref_val = 1e-10
            metric = metrics[j // len(variables)]
            if metric in higher_better_metrics:
                relative_matrix[i, j] = (
                    (ref_values[j] - data_matrix[i, j]) / ref_val * 100
                )
            elif metric == "Satisfies Bound (Value)":
                relative_matrix[i, j] = 100 if data_matrix[i, j] != 0 else 0
            else:
                relative_matrix[i, j] = (
                    (data_matrix[i, j] - ref_values[j]) / ref_val * 100
                )

    reds = sns.color_palette("Reds", 6)
    blues = sns.color_palette("Blues_r", 6)
    cmap = mpl.colors.ListedColormap(blues + [(0.95, 0.95, 0.95)] + reds)
    cb_levels = [-100, -75, -50, -25, -10, -1, 1, 10, 25, 50, 75, 100]
    norm = mpl.colors.BoundaryNorm(cb_levels, cmap.N, extend="both")

    ncompressors = len(compressors)
    nvariables = len(variables)
    nmetrics = len(metrics)

    panel_width = (2.5 / 5) * nvariables
    label_width = 1.5 * panel_width
    padding_right = 0.1
    panel_height = panel_width / nvariables

    title_height = panel_height * 1.25
    cbar_height = panel_height * 2
    spacing_height = panel_height * 0.1
    spacing_width = panel_height * 0.2

    total_width = (
        label_width
        + nmetrics * panel_width
        + (nmetrics - 1) * spacing_width
        + padding_right
    )
    total_height = (
        title_height
        + cbar_height
        + ncompressors * panel_height
        + (ncompressors - 1) * spacing_height
    )

    fig = plt.figure(figsize=(total_width, total_height))
    gs = mpl.gridspec.GridSpec(
        ncompressors,
        nmetrics,
        figure=fig,
        left=label_width / total_width,
        right=1 - padding_right / total_width,
        top=1 - (title_height / total_height),
        bottom=cbar_height / total_height,
        hspace=spacing_height / panel_height,
        wspace=spacing_width / panel_width,
    )

    img = None
    border_targets: list[tuple[mpl.axes.Axes, int]] = []
    for row, compressor in enumerate(compressors):
        for col, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[row, col])

            start_col = col * nvariables
            end_col = start_col + nvariables
            rel_values = relative_matrix[row, start_col:end_col].reshape(1, -1)
            abs_values = data_matrix[row, start_col:end_col]

            img = ax.imshow(rel_values, aspect="auto", cmap=cmap, norm=norm)

            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])

            for i in range(nvariables):
                rect = mpl.patches.Rectangle(
                    (i - 0.5, -0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="white",
                    facecolor="none",
                )
                ax.add_patch(rect)

                if (compressor, variables[i]) in converted_cells:
                    border_targets.append((ax, i))

            for i, val in enumerate(abs_values):
                color = "black" if abs(rel_values[0, i]) < 75 else "white"
                fontsize = 10
                if (
                    metric in UNRELIABLE_NAN_METRICS
                    and variables[i] in UNRELIABLE_NAN_VARIABLES
                ):
                    text = "N/A"
                    color = "black"
                elif np.isnan(val):
                    text = "Fail"
                    color = "black"
                elif abs(val) > 10_000:
                    text = f"{val:.1e}"
                    fontsize = 8
                elif abs(val) > 10:
                    text = f"{val:.0f}"
                elif abs(val) > 1:
                    text = f"{val:.1f}"
                elif val == 0:
                    text = "0.0"
                elif abs(val) < 0.01:
                    text = f"{val:.1e}"
                    fontsize = 8
                else:
                    text = f"{val:.2f}"
                ax.text(
                    i,
                    0,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    color=color,
                )

            if col == 0:
                ax.set_ylabel(
                    _get_compressor_legend_name(compressor),
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=10,
                    fontsize=14,
                )

            if row == 0:
                ax.set_title(METRICS2NAME.get(metric, metric), fontsize=16, pad=10)
                ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
                ax.set_xticks(range(nvariables))
                ax.set_xticklabels(
                    [VARIABLE2NAME.get(v, v) for v in variables],
                    rotation=45,
                    ha="left",
                    fontsize=12,
                )

            for spine in ax.spines.values():
                spine.set_color("0.7")

    # Mark converted cells with a small black triangle in the upper right corner.
    # Drawn last so they sit on top of the white grid rectangles and the axes spines.
    triangle_size = 0.3
    for ax, i in border_targets:
        x_right = i + 0.5
        y_top = -0.5
        triangle = mpl.patches.Polygon(
            [
                (x_right - triangle_size, y_top),
                (x_right, y_top),
                (x_right, y_top + triangle_size),
            ],
            closed=True,
            facecolor="black",
            edgecolor="none",
            zorder=10,
            clip_on=False,
        )
        ax.add_patch(triangle)

    if cbar and img is not None:
        rel_cbar_height = cbar_height / total_height
        cax = fig.add_axes((0.4, rel_cbar_height * 0.3, 0.5, rel_cbar_height * 0.2))
        cb = fig.colorbar(img, cax=cax, orientation="horizontal")
        cb.ax.set_xticks(cb_levels)
        cb.ax.set_xlabel(
            f"Better ← % difference vs {_get_compressor_legend_name(ref_compressor)} → Worse",
            fontsize=16,
        )

    plt.tight_layout()

    if save_fn:
        # bbox_inches="tight" is needed here because the row labels and the rotated
        # column labels stick out of the figure box.
        plt.savefig(save_fn, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_scorecards(
    df: pd.DataFrame,
    plots_path: Path,
    converted_cells: set[tuple[str, str]],
    bound_names: list[str] = ["low", "mid", "high"],
    ref_compressor: str = "bitround",
    metrics: list[str] = [
        "DSSIM",
        "MAE",
        "Max Absolute Error",
        "Spectral Error",
        "Compression Ratio [raw B / enc B]",
        "Satisfies Bound (Value)",
    ],
):
    """Create one scorecard per error bound, each split into two rows of metrics.

    Parameters
    ----------
    df: pd.DataFrame
        Results with the compressor names already normalized by
        `plot_metrics._rename_compressors`.
    plots_path: Path
        Directory the scorecards are written to. Created if it does not exist.
    converted_cells: set[tuple[str, str]]
        (compressor, variable) pairs to flag as having a converted error bound, as
        returned by `converted_bound_cells` on the raw results.
    """
    if ref_compressor not in df["Compressor"].values:
        print(
            f"Reference compressor {ref_compressor} is missing from the results, "
            "skipping the scorecards."
        )
        return

    plots_path.mkdir(parents=True, exist_ok=True)

    nrow1 = len(metrics) // 2
    for bound in bound_names:
        if df[df["Error Bound Name"] == bound].empty:
            print(f"No results for the {bound} error bound, skipping its scorecard.")
            continue

        print(f"Creating scorecard for {bound} bound...")
        data_matrix, compressors, variables = _create_data_matrix(
            df, bound, metrics, ref_compressor
        )
        split = nrow1 * len(variables)
        _create_compression_scorecard(
            data_matrix[:, :split],
            compressors,
            variables,
            metrics[:nrow1],
            converted_cells,
            ref_compressor=ref_compressor,
            cbar=False,
            save_fn=plots_path / f"{bound}_scorecard_row1.pdf",
        )
        _create_compression_scorecard(
            data_matrix[:, split:],
            compressors,
            variables,
            metrics[nrow1:],
            converted_cells,
            ref_compressor=ref_compressor,
            save_fn=plots_path / f"{bound}_scorecard_row2.pdf",
        )
