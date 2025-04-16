from abc import ABC, abstractmethod

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


class Plotter(ABC):
    def __init__(self):
        self.projection = ccrs.Robinson()

    @abstractmethod
    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var):
        pass

    def plot(self, ds, ds_new, dataset_name, compressor, var, outfile):
        fig, ax = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(20, 7),
            subplot_kw={"projection": self.projection},
        )
        self.plot_fields(fig, ax, ds, ds_new, dataset_name, var)
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


class CmipAtmosPlotter(Plotter):
    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var):
        selector = dict(time=0, plev=3)
        ds.isel(**selector).plot(ax=ax[0], transform=ccrs.PlateCarree())
        ds_new.isel(**selector).plot(
            ax=ax[1], transform=ccrs.PlateCarree(), robust=True
        )
        error = ds.isel(**selector) - ds_new.isel(**selector)
        error.plot(ax=ax[2], transform=ccrs.PlateCarree())


class CmipOceanPlotter(Plotter):
    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var):
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


class Era5Plotter(Plotter):
    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var):
        selector = dict(time=0)
        error = ds.isel(**selector) - ds_new.isel(**selector)

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


class NextGEMSPlotter(Plotter):
    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var):
        selector = dict(time=0)
        error = ds.isel(**selector) - ds_new.isel(**selector)

        lons = ds.isel(**selector).lon.values
        lats = ds.isel(**selector).lat.values
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        xys = self.projection.transform_points(ccrs.PlateCarree(), lon_grid, lat_grid)
        x, y = xys[..., 0], xys[..., 1]

        cmap = "Blues"
        max_val = max(
            ds.isel(**selector).max().values.item(),
            ds_new.isel(**selector).max().values.item(),
        )
        color_norm = mcolors.LogNorm(vmin=1e-12, vmax=max_val) if var == "pr" else None
        # Avoid zero values for log transformation for precipitation
        offset = 1e-12 if var == "pr" else 0
        c1 = ax[0].pcolormesh(
            x,
            y,
            ds.isel(**selector).values.squeeze() + offset,
            norm=color_norm,
            cmap=cmap,
        )
        c2 = ax[1].pcolormesh(
            x,
            y,
            ds_new.isel(**selector).values.squeeze() + offset,
            norm=color_norm,
            cmap=cmap,
        )
        c3 = ax[2].pcolormesh(x, y, error.values.squeeze(), cmap="coolwarm")
        for i, c in enumerate([c1, c2, c3]):
            fig.colorbar(c, ax=ax[i], shrink=0.6)


class CamsPlotter(Plotter):
    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var):
        selector = dict(valid_time=0, pressure_level=3)
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
        )
        ds_new.isel(**selector).plot(
            ax=ax[1],
            transform=ccrs.PlateCarree(),
            norm=color_norm,
            cmap="gist_earth",
        )
        error = ds.isel(**selector) - ds_new.isel(**selector)
        error.plot(ax=ax[2], transform=ccrs.PlateCarree())


class EsaBiomassPlotter(Plotter):
    def plot_fields(self, fig, ax, ds, ds_new, dataset_name, var):
        selector = dict(time=0)
        ds.isel(**selector).plot(ax=ax[0])
        ds_new.isel(**selector).plot(ax=ax[1])
        error = ds.isel(**selector) - ds_new.isel(**selector)
        error.plot(ax=ax[2])
        ax[0].set_title("Original Dataset")
        ax[1].set_title("Compressed Dataset")
        ax[2].set_title("Error")


PLOTTERS = {
    "cams-nitrogen-dioxide-tiny": CamsPlotter,
    "cmip6-access-ta-tiny": CmipAtmosPlotter,
    "cmip6-access-tos-tiny": CmipOceanPlotter,
    "era5-tiny": Era5Plotter,
    "esa-biomass-cci-tiny": EsaBiomassPlotter,
    "nextgems-icon-tiny": NextGEMSPlotter,
}
