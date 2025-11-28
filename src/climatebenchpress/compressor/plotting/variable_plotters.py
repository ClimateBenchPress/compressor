from abc import ABC, abstractmethod
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xarray.plot.utils as xplot_utils


class Plotter(ABC):
    datasets: list[str]
    title_fontsize = 22

    def __init__(self):
        self.projection = ccrs.Robinson()
        self.error_title = "Error"

    @abstractmethod
    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var, err_bound):
        pass

    def plot(
        self,
        ds,
        ds_new,
        dataset_name,
        compressor,
        var,
        err_bound,
        outfile: None | Path = None,
    ):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(18, 6),
            subplot_kw={"projection": self.projection},
        )
        self.plot_fields(fig, ax, ds, ds_new, dataset_name, var, err_bound)
        ax[0].coastlines()
        ax[1].coastlines()
        ax[2].coastlines()
        ax[0].set_title("Original Dataset", fontsize=self.title_fontsize)
        ax[1].set_title("Compressed Dataset", fontsize=self.title_fontsize)
        ax[2].set_title(self.error_title, fontsize=self.title_fontsize)
        # fig.suptitle(f"{var} Error for {dataset_name} ({compressor})")
        fig.tight_layout()
        if outfile is not None:
            with outfile.open("wb") as f:
                fig.savefig(f, dpi=300)
        plt.close()


class CmipAtmosPlotter(Plotter):
    datasets = ["cmip6-access-ta-tiny", "cmip6-access-ta"]

    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var, err_bound):
        selector = dict(time=0, plev=3)
        # Calculate shared vmin and vmax for consistent color ranges
        data_orig = ds.isel(**selector)
        data_new = ds_new.isel(**selector)
        vmin = np.nanmin(data_orig.values.squeeze())
        vmax = np.nanmax(data_orig.values.squeeze())

        data_orig.plot(ax=ax[0], transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
        data_new.plot(
            ax=ax[1], transform=ccrs.PlateCarree(), robust=True, vmin=vmin, vmax=vmax
        )
        error = data_orig - data_new
        error.attrs["long_name"] = data_orig.attrs.get("long_name", "")
        error.attrs["units"] = data_orig.attrs.get("units", "")

        _, bound_value = err_bound
        vmin_error, vmax_error = -bound_value, bound_value
        error.plot(
            ax=ax[2],
            transform=ccrs.PlateCarree(),
            rasterized=True,
            vmin=vmin_error,
            vmax=vmax_error,
            cbar_kwargs={"ticks": [-bound_value, 0, bound_value]},
            cmap="seismic",
        )
        self.error_title = "Absolute Error"


class CmipOceanPlotter(Plotter):
    datasets = ["cmip6-access-tos-tiny", "cmip6-access-tos"]

    cbar_label_fontsize = 20
    cbar_tick_fontsize = 16

    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var, err_bound):
        # Calculate shared vmin and vmax for consistent color ranges
        data_orig = ds.isel(time=0).values.squeeze()
        vmin, vmax = np.nanmin(data_orig), np.nanmax(data_orig)

        ds.isel(time=0).plot(
            ax=ax[0],
            x="longitude",
            y="latitude",
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            cbar_kwargs={
                "orientation": "vertical",
                "fraction": 0.046,
                "pad": 0.04,
                "label": "degC",
                "shrink": 0.6,
            },
        )

        ds_new.isel(time=0).plot(
            ax=ax[1],
            x="longitude",
            y="latitude",
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
            cbar_kwargs={
                "orientation": "vertical",
                "fraction": 0.046,
                "pad": 0.04,
                "label": "degC",
                "shrink": 0.6,
            },
        )

        error = ds.isel(time=0) - ds_new.isel(time=0)
        _, bound_value = err_bound
        vmin_error, vmax_error = -bound_value, bound_value
        error.plot(
            ax=ax[2],
            x="longitude",
            y="latitude",
            transform=ccrs.PlateCarree(),
            vmin=vmin_error,
            vmax=vmax_error,
            cmap="seismic",
            rasterized=True,
            cbar_kwargs={
                "orientation": "vertical",
                "fraction": 0.046,
                "pad": 0.04,
                "label": "degC",
                "shrink": 0.6,
                "ticks": [-bound_value, 0, bound_value],
            },
        )
        for a in ax:
            a.collections[0].colorbar.ax.set_ylabel(
                "degC", fontsize=self.cbar_label_fontsize
            )
            a.collections[0].colorbar.ax.tick_params(labelsize=self.cbar_tick_fontsize)
        ax[2].collections[0].colorbar.ax.yaxis.get_offset_text().set(
            size=self.cbar_tick_fontsize
        )
        self.error_title = "Absolute Error"


class Era5Plotter(Plotter):
    datasets = ["era5-tiny", "era5", "ifs-uncompressed"]

    cbar_label_fontsize = 18
    cbar_tick_fontsize = 14

    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var, err_bound):
        selector = dict(time=0)
        # Calculate shared vmin and vmax for consistent color ranges
        data_orig = ds.isel(**selector).values.squeeze()
        vmin, vmax = np.nanmin(data_orig), np.nanmax(data_orig)

        # Instead of using the inbuilt xarray plot method, we are manually doing
        # the projection and calling pcolormesh. By doing so we can avoid having
        # to do the projection three times and only have to do it once and re-use
        # it between plots.
        lons = ds.isel(**selector).longitude.values
        lats = ds.isel(**selector).latitude.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        xys = self.projection.transform_points(ccrs.PlateCarree(), lon_grid, lat_grid)
        x, y = xys[..., 0], xys[..., 1]
        # Wind variable plots coolwarm because they lie around 0 and change in sign
        # signifies change in wind direction.
        cmap = "coolwarm" if var.startswith("10m") else "viridis"
        c1 = ax[0].pcolormesh(
            x, y, ds.isel(**selector).values.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax
        )
        c2 = ax[1].pcolormesh(
            x,
            y,
            ds_new.isel(**selector).values.squeeze(),
            cmap=cmap,
            rasterized=True,
            vmin=vmin,
            vmax=vmax,
        )

        error = ds.isel(**selector) - ds_new.isel(**selector)
        error.attrs["long_name"] = ds.isel(**selector).attrs.get("long_name", "")
        error.attrs["units"] = ds.isel(**selector).attrs.get("units", "")

        _, bound_value = err_bound
        c3 = ax[2].pcolormesh(
            x,
            y,
            error.values.squeeze(),
            cmap="seismic",
            vmin=-bound_value,
            vmax=bound_value,
        )
        self.error_title = "Absolute Error"
        for i, c in enumerate([c1, c2, c3]):
            if i == 0:
                extend = xplot_utils._determine_extend(ds.isel(**selector), vmin, vmax)
                label = xplot_utils.label_from_attrs(ds)
            elif i == 1:
                extend = xplot_utils._determine_extend(
                    ds_new.isel(**selector), vmin, vmax
                )
                label = xplot_utils.label_from_attrs(ds_new)
            else:
                extend = xplot_utils._determine_extend(error, -bound_value, bound_value)
                label = xplot_utils.label_from_attrs(error)
            cbar = fig.colorbar(c, ax=ax[i], shrink=0.6, extend=extend, label=label)
            if i == 2:
                cbar.ax.set_yticks([-bound_value, 0, bound_value])
            cbar.ax.tick_params(labelsize=self.cbar_tick_fontsize)
            cbar.ax.set_ylabel(label, fontsize=self.cbar_label_fontsize)


class NextGEMSPlotter(Plotter):
    datasets = ["nextgems-icon-tiny", "nextgems-icon"]

    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var, err_bound):
        selector = dict(time=0)
        data_orig = ds.isel(**selector).values.squeeze()
        vmin, vmax = np.nanmin(data_orig), np.nanmax(data_orig)

        lons = ds.isel(**selector).lon.values
        lats = ds.isel(**selector).lat.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        xys = self.projection.transform_points(ccrs.PlateCarree(), lon_grid, lat_grid)
        x, y = xys[..., 0], xys[..., 1]

        cmap = "Blues"
        # Avoid zero values for log transformation for precipitation
        offset = 1e-12 if var == "pr" else 0
        color_norm = (
            mcolors.LogNorm(vmin=1e-12, vmax=vmax + offset)
            if var == "pr"
            else mcolors.Normalize(vmin=vmin + offset, vmax=vmax + offset)
        )
        c1 = ax[0].pcolormesh(
            x,
            y,
            ds.isel(**selector).values.squeeze() + offset,
            norm=color_norm,
            cmap=cmap,
            rasterized=True,
        )
        c2 = ax[1].pcolormesh(
            x,
            y,
            ds_new.isel(**selector).values.squeeze() + offset,
            norm=color_norm,
            cmap=cmap,
            rasterized=True,
        )

        bound_type, bound_value = err_bound
        error = ds.isel(**selector) - ds_new.isel(**selector)
        self.error_title = "Absolute Error"
        if bound_type == "rel_error":
            error = error / np.abs(ds.isel(**selector))
            self.error_title = "Relative Error"

        c3 = ax[2].pcolormesh(
            x,
            y,
            error.values.squeeze(),
            cmap="seismic",
            vmin=-bound_value,
            vmax=bound_value,
        )
        for i, c in enumerate([c1, c2, c3]):
            if i == 0:
                extend = xplot_utils._determine_extend(
                    ds.isel(**selector), vmin + offset, vmax + offset
                )
                label = xplot_utils.label_from_attrs(ds)
            elif i == 1:
                extend = xplot_utils._determine_extend(
                    ds_new.isel(**selector), vmin + offset, vmax + offset
                )
                label = xplot_utils.label_from_attrs(ds_new)
            elif i == 2 and bound_type == "rel_error":
                extend = xplot_utils._determine_extend(error, -bound_value, bound_value)
                label = ""
            elif i == 2 and bound_type == "abs_error":
                extend = xplot_utils._determine_extend(error, -bound_value, bound_value)
                # Error has same label as original dataset.
                label = xplot_utils.label_from_attrs(ds)
            cbar = fig.colorbar(c, ax=ax[i], shrink=0.6, label=label, extend=extend)
            if i == 2:
                cbar.ax.set_yticks([-bound_value, 0, bound_value])


class CamsPlotter(Plotter):
    datasets = ["cams-nitrogen-dioxide-tiny", "cams-nitrogen-dioxide"]

    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var, err_bound):
        selector = dict(valid_time=0, hybrid=3)
        in_min = ds.isel(**selector).min().values.item()
        in_max = ds.isel(**selector).max().values.item()
        out_min = ds_new.isel(**selector).min().values.item()
        out_max = ds_new.isel(**selector).max().values.item()
        vmin, vmax = min(in_min, out_min), max(in_max, out_max)
        vmin = max(vmin, 1e-14)  # Avoid zero values for log transformation
        color_norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        ds.isel(**selector).plot(
            ax=ax[0],
            transform=ccrs.PlateCarree(),
            norm=color_norm,
            cmap="gist_earth",
            rasterized=True,
        )
        ds_new.isel(**selector).plot(
            ax=ax[1],
            transform=ccrs.PlateCarree(),
            norm=color_norm,
            cmap="gist_earth",
            rasterized=True,
        )

        error = ds.isel(**selector) - ds_new.isel(**selector)
        rel_error = error / np.abs(ds.isel(**selector))
        # Sets the colorbar label.
        rel_error.attrs["long_name"] = ""

        _, bound_value = err_bound
        vmin_error, vmax_error = -bound_value, bound_value
        rel_error.plot(
            ax=ax[2],
            transform=ccrs.PlateCarree(),
            vmin=vmin_error,
            vmax=vmax_error,
            cmap="seismic",
            cbar_kwargs={"ticks": [-bound_value, 0, bound_value]},
        )
        self.error_title = "Relative Error"


class EsaBiomassPlotter(Plotter):
    datasets = ["esa-biomass-cci-tiny", "esa-biomass-cci"]

    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var, err_bound):
        selector = dict(time=0)
        data_orig = ds.isel(**selector).values.squeeze()
        vmin, vmax = np.nanmin(data_orig), np.nanmax(data_orig)

        ds.isel(**selector).plot(ax=ax[0], cmap="Greens", vmin=vmin, vmax=vmax)
        ds_new.isel(**selector).plot(ax=ax[1], cmap="Greens", vmin=vmin, vmax=vmax)

        _, bound_value = err_bound
        error = ds.isel(**selector) - ds_new.isel(**selector)
        non_zero_mask = np.abs(ds.isel(**selector)) > 0.0
        # Check where both original and new data are zero
        both_zero_mask = (np.abs(ds.isel(**selector)) == 0.0) & (
            np.abs(ds_new.isel(**selector)) == 0.0
        )
        rel_error = xr.where(
            both_zero_mask,
            0.0,
            xr.where(non_zero_mask, error / np.abs(ds.isel(**selector)), 1e12),
        )
        rel_error.attrs["long_name"] = ""
        rel_error.plot(
            ax=ax[2],
            rasterized=True,
            cmap="seismic",
            vmin=-bound_value,
            vmax=bound_value,
            cbar_kwargs={"ticks": [-bound_value, 0, bound_value]},
        )
        self.error_title = "Relative Error"


plotter_clss: list[type[Plotter]] = [
    CamsPlotter,
    CmipAtmosPlotter,
    CmipOceanPlotter,
    Era5Plotter,
    EsaBiomassPlotter,
    NextGEMSPlotter,
]
PLOTTERS: dict[str, type[Plotter]] = dict()
for plotter_cls in plotter_clss:
    for dataset in plotter_cls.datasets:
        assert dataset not in PLOTTERS, f"Duplicate dataset found: {dataset}"
        PLOTTERS[dataset] = plotter_cls
