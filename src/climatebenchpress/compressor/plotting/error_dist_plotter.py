import matplotlib.pyplot as plt
import numpy as np


class ErrorDistPlotter:
    def __init__(self, dataset, variables, error_bounds):
        self.fig, self.axes = plt.subplots(
            len(variables),
            len(error_bounds),
            figsize=(17, 5 * len(variables)),
            squeeze=False,
        )
        self.fig.suptitle(f"Error Histograms for {dataset}", fontsize=16)

        self.errors = {var: dict() for var in variables}

    def compute_errors(self, compressor, ds, ds_new, var, err_bound_type):
        if "-pco" in compressor:
            return

        if err_bound_type == "abs_error":
            error = (ds_new[var] - ds[var]).compute().values
        elif err_bound_type == "rel_error":
            error = ((ds_new[var] - ds[var]) / ds[var]).compute().values
        else:
            raise ValueError(f"Unknown error bound type: {err_bound_type}")
        self.errors[var][compressor] = error.flatten()

    def plot_error_bound_histograms(
        self,
        col_index,
        variables,
        compressors,
        error_bound_vals,
        compressor2legendname,
        compressor2lineinfo,
    ):
        """
        Plot error histograms for a single error bound across all variables in that
        dataset.
        """
        # We only plot bitround and stochround once because the lossless compressor
        # does not change the error plot distribution.
        legend_names = compressor2legendname.copy()
        legend_names.update(
            {
                "bitround-conservative-rel": "BitRound",
                "stochround": "StochRound",
            }
        )
        compressors = [comp for comp in compressors if "-pco" not in comp]

        for j, var in enumerate(variables):
            for comp in compressors:
                self.axes[j, col_index].hist(
                    self.errors[var][comp],
                    bins=100,
                    density=True,
                    histtype="step",
                    label=legend_names.get(comp, comp),
                    color=compressor2lineinfo.get(comp, ("#000000", "-"))[0],
                    linestyle=compressor2lineinfo.get(comp, ("#000000", "-"))[1],
                    linewidth=2,
                    alpha=0.8,
                )

            error_bound_name, error_bound_value = error_bound_vals[var]
            self.axes[j, col_index].set_xlabel("Error Value")
            self.axes[j, col_index].set_ylabel("Log Probability Density")
            self.axes[j, col_index].set_yscale("log")

            xticks = np.linspace(-2 * error_bound_value, 2 * error_bound_value, num=5)
            xlabels = ["0.0" if x == 0.0 else f"{x:.2e}" for x in xticks]
            self.axes[j, col_index].set_xticks(xticks, labels=xlabels)
            self.axes[j, col_index].set_xlim(
                -2 * error_bound_value, 2 * error_bound_value
            )

            self.axes[j, col_index].set_title(
                f"{var}\n{error_bound_name} = {error_bound_value:.2e}"
                if col_index == 1
                else f"{error_bound_name} = {error_bound_value:.2e}"
            )

        # Reset errors for the next iteration. Ensures we don't plot the wrong errors
        # for the next error bound.
        self.errors = {var: dict() for var in variables}

    def get_final_figure(self):
        self.axes[0, -1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        self.fig.tight_layout()
        return self.fig, self.axes
