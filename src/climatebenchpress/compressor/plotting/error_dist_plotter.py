import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class ErrorDistPlotter:
    def __init__(self, variables, error_bounds):
        self.variables = variables
        self.error_bounds = error_bounds
        self.figs = {}
        self.axes = {}
        # Create a separate figure for each variable
        for var in variables:
            self.figs[var], self.axes[var] = plt.subplots(
                1,
                len(error_bounds),
                figsize=(17, 5),
                squeeze=False,
            )
            # Remove the first dimension since we only have 1 row
            self.axes[var] = self.axes[var][0]

        self.errors = {var: dict() for var in variables}

    def compute_errors(self, compressor, ds, ds_new, var, err_bound_type):
        if "-pco" in compressor:
            return

        error = robust_error(ds[var], ds_new[var])
        if err_bound_type == "abs_error":
            error = error.compute().values
        elif err_bound_type == "rel_error":
            # Relative error calculation with avoiding division by zero.
            error = (
                xr.where(
                    (ds[var] == 0) & (ds[var] == ds_new[var]),
                    0.0,
                    error / np.abs(ds[var]),
                )
                .compute()
                .values
            )
        else:
            raise ValueError(f"Unknown error bound type: {err_bound_type}")
        self.errors[var][compressor] = error.flatten()

    def plot_error_bound_histograms(
        self,
        col_index,
        variables,
        compressors,
        error_bound_vals,
        get_legend_name,
        get_line_info,
    ):
        """
        Plot error histograms for a single error bound across all variables in that
        dataset.
        """
        # We only plot bitround and stochround once because the lossless compressor
        # does not change the error plot distribution. Hence, we ignore the PCO
        # compressors here.
        compressors = [comp for comp in compressors if "-pco" not in comp]
        for var in variables:
            for comp in compressors:
                color, linestyle = get_line_info(comp)
                label = get_legend_name(comp)
                # Don't state the lossless compressor in the legend.
                if label.startswith("BitRound"):
                    label = "BitRound"
                elif label.startswith("StochRound"):
                    label = "StochRound"
                # Filter out inf values
                error_data = self.errors[var][comp]
                error_data = error_data[~np.isinf(error_data) & ~np.isnan(error_data)]
                self.axes[var][col_index].hist(
                    np.float64(error_data),
                    bins=100,
                    density=True,
                    histtype="step",
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=4,
                    alpha=0.6,
                )

            error_bound_name, error_bound_value = error_bound_vals[var]
            self.axes[var][col_index].set_xlabel("Error Value", fontsize=14)
            self.axes[var][col_index].set_ylabel("Log Probability Density", fontsize=14)
            self.axes[var][col_index].set_yscale("log")

            xticks = np.linspace(-2 * error_bound_value, 2 * error_bound_value, num=5)
            xlabels = [_format_number(x) for x in xticks]
            self.axes[var][col_index].set_xticks(xticks, labels=xlabels, fontsize=12)
            self.axes[var][col_index].set_xlim(
                -2 * error_bound_value, 2 * error_bound_value
            )

            self.axes[var][col_index].axvline(
                -error_bound_value,
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
            )
            self.axes[var][col_index].axvline(
                error_bound_value, color="black", linestyle="--", linewidth=2, alpha=0.7
            )

            err_bound_type = (
                "Abs. Error" if error_bound_name == "abs_error" else "Rel. Error"
            )
            self.axes[var][col_index].set_title(
                f"{err_bound_type} = {_format_number(error_bound_value)}", fontsize=14
            )

        # Reset errors for the next iteration. Ensures we don't plot the wrong errors
        # for the next error bound.
        self.errors = {var: dict() for var in variables}

    def get_final_figure(self):
        for var in self.variables:
            self.axes[var][-1].legend(
                loc="center left", bbox_to_anchor=(1, 0.5), fontsize=14
            )
            self.figs[var].tight_layout()
        return self.figs, self.axes


def robust_error(x, y):
    """
    Compute the difference between two arrays with NaN and inf handling. Ensures
    that the difference is 0 when both x and y are NaN or inf, and returns NaN
    when one is NaN or inf and the other is not.
    """
    x_nan = np.isnan(x)
    y_nan = np.isnan(y)
    x_inf = np.isinf(x)
    y_inf = np.isinf(y)

    # Check if infinities have mismatched signs
    inf_sign_mismatch = (x_inf & y_inf) & (np.sign(x) != np.sign(y))
    both_nan = x_nan & y_nan
    both_inf = x_inf & y_inf

    # Hard check: If both are NaN or inf with matching signs, return 0.
    # If one is NaN or inf and the other is not, or infinities have mismatched signs, then
    # the result will already evaluate to NaN.
    result = xr.where(both_nan | (both_inf & ~inf_sign_mismatch), 0.0, x - y)

    return result


def _format_number(value: float) -> str:
    if abs(value) > 10_000:
        text = f"{value:.1e}"
    elif abs(value) > 10:
        text = f"{value:.0f}"
    elif abs(value) > 1:
        text = f"{value:.1f}"
    elif value == 0:
        text = "0.0"
    elif abs(value) < 0.001:
        text = f"{value:.1e}"
    else:
        # 0.001 <= abs(value) < 1
        text = f"{value:.3f}"

    return text
