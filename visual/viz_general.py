from __future__ import annotations

from typing import Any, no_type_check

from matplotlib import typing
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs

from src.utils import DataCompose
from src.utils.colorbar import make_cmap, qw1000_levels

class VizGeneral():
    def __init__(self):
        self.windspeed_cmap = make_cmap("clist_WS")

        self.qw1000_cmap = make_cmap("clist_prec")
        self.qw1000_cticks = qw1000_levels[::2]
        self.qw1000_cticks.pop(1)
        self.vort_cmap = make_cmap("clist_vort")

    # -------------------------
    # Small utilities
    # -------------------------
    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert torch/xarray/numpy to numpy array."""
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return np.asarray(x.numpy())
        if hasattr(x, "values"):
            return np.asarray(x.values)
        return np.asarray(x)


    def _as_2d_lonlat(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lon = self._to_numpy(lon)
        lat = self._to_numpy(lat)
        if lon.ndim == 1 and lat.ndim == 1:
            Lon, Lat = np.meshgrid(lon, lat)
            return Lon, Lat
        if lon.ndim == 2 and lat.ndim == 2:
            return lon, lat
        raise ValueError(f"lon/lat dims not supported: lon={lon.shape}, lat={lat.shape}")


    def _as_1d_from_2d_grid(self, grid2d: np.ndarray, axis: int) -> np.ndarray:
        """
        For regular lat/lon grid stored as 2D (ny,nx):
        axis=0 -> return x (lon) as grid2d[0,:]
        axis=1 -> return y (lat) as grid2d[:,0]
        """
        if grid2d.ndim == 1:
            return grid2d
        if grid2d.ndim != 2:
            raise ValueError(f"grid must be 1D/2D, got {grid2d.ndim}D")
        return grid2d[0, :] if axis == 0 else grid2d[:, 0]


    def _safe_squeeze_tyx(self, arr: np.ndarray) -> np.ndarray:
        """
        Make sure array is (T, Y, X). If it is (Y,X) -> add T=1.
        If it is (T,1,Y,X) -> squeeze channel.
        """
        a = self._to_numpy(arr)
        a = np.squeeze(a)
        if a.ndim == 2:
            a = a[None, ...]
        elif a.ndim == 3:
            pass
        elif a.ndim == 4:
            # assume (T,C,Y,X) or (C,T,Y,X) - try common case
            if a.shape[1] in (1,):
                a = a[:, 0, :, :]
            elif a.shape[0] in (1,):
                a = a[0, :, :, :][None, ...]
            else:
                # last resort: flatten first dim as T if looks like T
                a = a.reshape(a.shape[0], a.shape[-2], a.shape[-1])
        else:
            raise ValueError(f"Unexpected array shape after squeeze: {a.shape}")
        return a


    def _calc_rel_vorticity(self, u: np.ndarray, v: np.ndarray, Lon: np.ndarray, Lat: np.ndarray) -> np.ndarray:
        """
        Relative vorticity on a lat/lon grid:
        zeta = (1/(a cosφ)) ∂v/∂λ - (1/a) ∂u/∂φ
        u,v: (Y,X), Lon/Lat: degrees (Y,X)
        """
        a = 6371000.0
        lon_r = np.deg2rad(Lon)
        lat_r = np.deg2rad(Lat)

        # assume regular grid: use 1D spacing from edges
        lam_1d = self._as_1d_from_2d_grid(lon_r, axis=0)
        phi_1d = self._as_1d_from_2d_grid(lat_r, axis=1)

        dlam = np.gradient(lam_1d)  # (X,)
        dphi = np.gradient(phi_1d)  # (Y,)

        dv_dlam = np.gradient(v, axis=1) / dlam[None, :]
        du_dphi = np.gradient(u, axis=0) / dphi[:, None]

        cosphi = np.cos(lat_r)
        cosphi = np.where(np.abs(cosphi) < 1e-6, np.sign(cosphi) * 1e-6, cosphi)

        zeta = (1.0 / (a * cosphi)) * dv_dlam - (1.0 / a) * du_dphi
        return zeta


    # -------------------------
    # Plotting
    # -------------------------
    def _add_cbar(self, fig, ax, mappable, ticks=None, extend="neither"):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05, axes_class=Axes)
        cb = fig.colorbar(mappable, cax=cax, extend=extend, ticks=ticks)
        cb.ax.tick_params(labelsize=9)
        return cb

    @no_type_check
    def _plot_row(
        self,
        fig: Figure,
        Lon, Lat,
        z1000, u10, v10, ws10,
        qw1000,
        u850, v850,
        w700, u700, v700,
        qv500, u500, v500,
        quiver_skip=(slice(None, None, 10), slice(None, None, 10)),
    ) -> None:
        proj = ccrs.PlateCarree()
        extent = [float(np.nanmin(Lon)), float(np.nanmax(Lon)), float(np.nanmin(Lat)), float(np.nanmax(Lat))]

        lon1d = self._as_1d_from_2d_grid(Lon, axis=0)
        lat1d = self._as_1d_from_2d_grid(Lat, axis=1)

        # --- Col 1: ws10 + z1000 + streamplot
        ax = fig.add_subplot(1, 5, 1, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.coastlines(linewidth=0.8)
        m0 = ax.pcolormesh(Lon, Lat, ws10, transform=proj, cmap=self.windspeed_cmap, vmin=0, vmax=40)
        self._add_cbar(fig, ax, m0, extend="max")
        ax.contour(Lon, Lat, z1000, transform=proj, levels=np.arange(0, 1000, 5), colors="b", linewidths=0.3)
        ax.contour(Lon, Lat, z1000, transform=proj, levels=np.arange(0, 1000, 20), colors="b", linewidths=1)
        ax.streamplot(lon1d, lat1d, u10, v10, transform=proj, color="k", linewidth=0.4, density=1.2)
        ax.set_title("WS10 + Z1000", fontsize=11)

        # --- Col 2: qw1000 + quiver(10m)
        ax = fig.add_subplot(1, 5, 2, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.coastlines(linewidth=0.8)
        m1 = ax.pcolormesh(Lon, Lat, qw1000, transform=proj, cmap=self.qw1000_cmap, vmin=0, vmax=2)
        self._add_cbar(fig, ax, m1, extend="max")
        ax.quiver(
            Lon[quiver_skip], Lat[quiver_skip],
            u10[quiver_skip], v10[quiver_skip],
            transform=proj, color="k", linewidth=0.3, scale=500
        )
        ax.set_title("Qw1000 + 10m wind", fontsize=11)

        # --- Col 3: vorticity(850) + quiver(850)
        ax = fig.add_subplot(1, 5, 3, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.coastlines(linewidth=0.8)
        vort = self._calc_rel_vorticity(u850, v850, Lon, Lat) * 1e5  # to 1e-5 s^-1 scale like old script
        m2 = ax.pcolormesh(Lon, Lat, vort, transform=proj, cmap=self.vort_cmap, vmin=-65, vmax=65)
        self._add_cbar(fig, ax, m2, extend="both")
        ax.quiver(
            Lon[quiver_skip], Lat[quiver_skip],
            u850[quiver_skip], v850[quiver_skip],
            transform=proj, color="k", linewidth=0.3, scale=500
        )
        ax.set_title("Vort(850) + wind", fontsize=11)

        # --- Col 4: w(700) + streamplot(700)
        ax = fig.add_subplot(1, 5, 4, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.coastlines(linewidth=0.8)
        m3 = ax.pcolormesh(Lon, Lat, w700, transform=proj, cmap="bwr", vmin=-1.5, vmax=1.5)
        self._add_cbar(fig, ax, m3, extend="both")
        ax.streamplot(lon1d, lat1d, u700, v700, transform=proj, color="k", linewidth=0.4, density=1.2)
        ax.set_title("W(700) + wind", fontsize=11)

        # --- Col 5: qv500 + streamplot(500)
        ax = fig.add_subplot(1, 5, 5, projection=proj)
        ax.set_extent(extent, crs=proj)
        ax.coastlines(linewidth=0.8)
        m4 = ax.pcolormesh(Lon, Lat, qv500, transform=proj, cmap="Spectral", vmin=0.001, vmax=0.008)
        self._add_cbar(fig, ax, m4, extend="both")
        ax.streamplot(lon1d, lat1d, u500, v500, transform=proj, color="k", linewidth=0.4, density=1.2)
        ax.set_title("Qv500 + wind(500)", fontsize=11)

    def _get_var_data(self, var: DataCompose) -> np.ndarray:
        lev = 0
        if var.level in self.p_levs:
            lev = self.p_levs.index(var.level)
            var_idx = self.v_upper.index(var.var_name)
            oup = self.d_upper
        else:
            var_idx = self.v_sfc.index(var.var_name)
            oup = self.d_sfc
        return oup[lev, :, :, var_idx]

    # -------------------------
    # Main
    # -------------------------
    def plot(self, lon: np.ndarray, lat: np.ndarray, d_upp: np.ndarray, d_sfc: np.ndarray, p_levels: list, upp_vars: list, sfc_vars: list, epoch: int = -1) -> tuple[Figure, Axes]:
        plt.close()
        self.p_levs = p_levels
        self.d_upper = d_upp
        if self.d_upper.ndim == 5:
            self.d_upper = self.d_upper[0, ...]  # squeeze T dim if exists
        self.d_sfc = d_sfc
        if self.d_sfc.ndim == 5:
            self.d_sfc = self.d_sfc[0, ...]  # squeeze T dim if exists
        self.v_upper = upp_vars
        self.v_sfc = sfc_vars
        fig = plt.figure(1, figsize=(20, 3.5), dpi=150, layout="constrained")

        Lon, Lat = self._as_2d_lonlat(lon, lat)

        # DataCompose definitions (new project naming)
        u10, v10 = DataCompose.from_config({"U": ["Meter10"], "V": ["Meter10"]})
        (z1000,) = DataCompose.from_config({"Z": ["Hpa1000"]})
        (qw1000,) = DataCompose.from_config({"Qw": ["Hpa1000"]})
        u850, v850 = DataCompose.from_config({"U": ["Hpa850"], "V": ["Hpa850"]})
        (w700,) = DataCompose.from_config({"W": ["Hpa700"]})
        u700, v700 = DataCompose.from_config({"U": ["Hpa700"], "V": ["Hpa700"]})
        (qv500,) = DataCompose.from_config({"Qv": ["Hpa500"]})
        u500, v500 = DataCompose.from_config({"U": ["Hpa500"], "V": ["Hpa500"]})

        # --- Model rows
        data_u10 =    self._get_var_data(u10)
        data_v10 =    self._get_var_data(v10)
        data_z1000 =  self._get_var_data(z1000)
        data_qw1000 = self._get_var_data(qw1000)
        data_u850 =   self._get_var_data(u850)
        data_v850 =   self._get_var_data(v850)
        data_w700 =   self._get_var_data(w700)
        data_u700 =   self._get_var_data(u700)
        data_v700 =   self._get_var_data(v700)
        data_qv500 =  self._get_var_data(qv500)
        data_u500 =   self._get_var_data(u500)
        data_v500 =   self._get_var_data(v500)

        ws10 = np.sqrt(data_u10 ** 2 + data_v10 ** 2)

        self._plot_row(
            fig, Lon, Lat,
            z1000=data_z1000, u10=data_u10, v10=data_v10, ws10=ws10,
            qw1000=data_qw1000,
            u850=data_u850, v850=data_v850,
            w700=data_w700, u700=data_u700, v700=data_v700,
            qv500=data_qv500, u500=data_u500, v500=data_v500,
        )

        if epoch >= 0:
            fig.suptitle(f"Epoch {epoch}")

        return fig, fig.get_axes()[0]
