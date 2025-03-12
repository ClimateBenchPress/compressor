from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).parent.parent


def main():
    metrics_path = REPO / "metrics"
    plots_path = REPO / "plots"

    all_results = pd.read_csv(metrics_path / "all_results.csv")
    for dataset in all_results["Dataset"].unique():
        df = all_results[all_results["Dataset"] == dataset]
        dataset_plots_path = plots_path / dataset

        # For each variable and compressor, plot the input, output, and error fields.
        variables = df["Variable"].unique()
        compressors = df["Compressor"].unique()
        for var in variables:
            for comp in compressors:
                print(f"Plotting {var} error for {comp}...")
                plot_variable_error(
                    REPO,
                    dataset,
                    comp,
                    var,
                    dataset_plots_path / f"{var}_{comp}.png",
                )

        plot_rd_curve(
            df,
            compressors,
            dataset_plots_path / "compression_ratio_vs_psnr.png",
        )

    plot_metrics(plots_path, all_results)


def plot_variable_error(repo, dataset_name, compressor, var, outfile):
    if outfile.exists():
        # These plots can be quite expensive to generate, so we skip if they already exist.
        return

    compressed = (
        repo
        / ".."
        / "compressor"
        / "compressed-datasets"
        / dataset_name
        / compressor
        / "decompressed.zarr"
    )
    input = (
        repo / ".." / "data-loader" / "datasets" / dataset_name / "standardized.zarr"
    )

    ds = xr.open_dataset(input, chunks=dict(), engine="zarr").compute()
    ds_new = xr.open_dataset(compressed, chunks=dict(), engine="zarr").compute()
    ds, ds_new = ds[var], ds_new[var]

    if dataset_name.startswith("esa-biomass"):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
        selector = dict(time=0)
        ds.isel(**selector).plot(ax=ax[0])
        ds_new.isel(**selector).plot(ax=ax[1])
        error = ds.isel(**selector) - ds_new.isel(**selector)
        error.plot(ax=ax[2])
        ax[0].set_title("Original Dataset")
        ax[1].set_title("Compressed Dataset")
        ax[2].set_title("Error")
        fig.suptitle(f"{var} Error for {dataset_name} ({compressor})")
        plt.tight_layout()
        plt.savefig(outfile)
        plt.close()
    else:
        plot_global_variable(dataset_name, compressor, var, outfile, ds, ds_new)


def plot_global_variable(dataset_name, compressor, var, outfile, ds, ds_new):
    projection = ccrs.Robinson()
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(20, 7), subplot_kw={"projection": projection}
    )
    if dataset_name.startswith("cmip6") and var == "tos":
        pcm0 = ax[0].pcolormesh(
            ds.longitude.values,
            ds.latitude.values,
            ds.isel(time=0).values.squeeze(),
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap="coolwarm",
        )
        fig.colorbar(
            pcm0, ax=ax[0], orientation="vertical", fraction=0.046, pad=0.04
        ).set_label("degC")

        pcm1 = ax[1].pcolormesh(
            ds_new.longitude.values,
            ds_new.latitude.values,
            ds_new.isel(time=0).values.squeeze(),
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap="coolwarm",
        )
        fig.colorbar(
            pcm1, ax=ax[1], orientation="vertical", fraction=0.046, pad=0.04
        ).set_label("degC")

        error = ds.isel(time=0) - ds_new.isel(time=0)
        pcm2 = ax[2].pcolormesh(
            ds.longitude.values,
            ds.latitude.values,
            error.values.squeeze(),
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap="coolwarm",
        )
        fig.colorbar(
            pcm2, ax=ax[2], orientation="vertical", fraction=0.046, pad=0.04
        ).set_label("degC")
    elif dataset_name.startswith("cmip6") and var == "ta":
        selector = dict(time=0, plev=3)
        ds.isel(**selector).plot(ax=ax[0], transform=ccrs.PlateCarree())
        ds_new.isel(**selector).plot(
            ax=ax[1], transform=ccrs.PlateCarree(), robust=True
        )
        error = ds.isel(**selector) - ds_new.isel(**selector)
        error.plot(ax=ax[2], transform=ccrs.PlateCarree())
    elif dataset_name.startswith("cams"):
        selector = dict(valid_time=0, pressure_level=3)
        ds.isel(**selector).plot(ax=ax[0], transform=ccrs.PlateCarree())
        ds_new.isel(**selector).plot(ax=ax[1], transform=ccrs.PlateCarree())
        error = ds.isel(**selector) - ds_new.isel(**selector)
        error.plot(ax=ax[2], transform=ccrs.PlateCarree())
    elif dataset_name.startswith("era5"):
        selector = dict(time=0)
        error = ds.isel(**selector) - ds_new.isel(**selector)

        # Instead of using the inbuilt xarray plot method, we are manually doing
        # the projection and calling pcolormesh. By doing so we can avoid having
        # to do the projection three times and only have to do it once and re-use
        # it between plots.
        lons = ds.isel(**selector).longitude.values
        lats = ds.isel(**selector).latitude.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        xys = projection.transform_points(ccrs.PlateCarree(), lon_grid, lat_grid)
        x, y = xys[..., 0], xys[..., 1]
        # Wind variable plots coolwarm because they lie around 0 and change in sign
        # signifies change in wind direction.
        cmap = "coolwarm" if var.startswith("10m") else "viridis"
        c1 = ax[0].pcolormesh(x, y, ds.isel(**selector).values.squeeze(), cmap=cmap)
        c2 = ax[1].pcolormesh(
            x,
            y,
            ds_new.isel(**selector).values.squeeze(),
            cmap=cmap,
        )
        c3 = ax[2].pcolormesh(x, y, error.values.squeeze(), cmap="coolwarm")
        for i, c in enumerate([c1, c2, c3]):
            fig.colorbar(c, ax=ax[i], shrink=0.6)

    ax[0].coastlines()
    ax[1].coastlines()
    ax[2].coastlines()
    ax[0].set_title("Original Dataset")
    ax[1].set_title("Compressed Dataset")
    ax[2].set_title("Error")
    fig.suptitle(f"{var} Error for {dataset_name} ({compressor})")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_metrics(
    plots_path,
    all_results,
    metrics=[
        "Compression Ratio [raw B / enc B]",
        "Encode Instructions [# / raw B]",
        "Decode Instructions [# / raw B]",
        "PSNR",
        "Spectral Error",
        "MAE",
    ],
):
    markers = [
        "o",
        "s",
        "D",
        "^",
        "p",
        "P",
        "v",
        "<",
        ">",
    ]
    compressors = all_results["Compressor"].unique()

    for metric in metrics:
        plt.figure(figsize=(12, 8))
        plt.grid(True, which="major", axis="y")

        for i, comp in enumerate(compressors):
            compressor_data = all_results[all_results["Compressor"] == comp]
            plt.scatter(
                compressor_data["Dataset"] + " / " + compressor_data["Variable"],
                compressor_data[metric],
                label=comp,
                marker=markers[i % len(markers)],
                s=100,  # Increase the size of the markers
                alpha=0.8,
            )

        plt.title(f"{metric} across Datasets and Variables")
        plt.xlabel("Dataset / Variable")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.legend(title="Compressor")
        plt.yscale("log")
        plt.tight_layout()

        metric_fname = " ".join(metric.split()[:2]).replace(" ", "_").lower()
        outfile = plots_path / f"{metric_fname}.png"
        plt.savefig(outfile)
        plt.close()


def plot_rd_curve(df, compressors, outfile):
    plt.figure(figsize=(10, 6))
    for comp in compressors:
        compressor_data = df[df["Compressor"] == comp]
        plt.scatter(
            compressor_data["Compression Ratio [raw B / enc B]"],
            compressor_data["PSNR"],
            label=comp,
            s=100,
        )

    plt.title("Compression Ratio vs PSNR")
    plt.xlabel("Compression Ratio [raw B / enc B]")
    plt.ylabel("PSNR")
    plt.legend(title="Compressor")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


if __name__ == "__main__":
    main()
