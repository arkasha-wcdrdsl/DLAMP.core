from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.const import FIGURE_PATH, WSP_COLOR, WSP_LV
from src.utils import DataCompose, DataType, Level, gen_data

from .tw_background import TwBackground


class VizWind(TwBackground):
    def __init__(self, pressure_level: str | None = None):
        super().__init__()
        self.press_lv = pressure_level
        self.title_suffix = f"Wind@{self.press_lv}" if pressure_level else ""

    # ==== Added: check whether coordinates are evenly spaced. ====
    @staticmethod
    def _is_evenly_spaced(coord: np.ndarray, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
        """Check whether a 1D coordinate array is strictly increasing and evenly spaced."""
        coord = np.asarray(coord)
        if coord.ndim != 1 or coord.size < 2:
            return True  # Treat very short arrays as valid.
        diffs = np.diff(coord)
        # Coordinates must be strictly increasing with nearly constant spacing.
        if not np.all(diffs > 0):
            return False
        return np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)

    # ==== Added: normalize 1D/2D lon/lat to 1D axes. ====
    @staticmethod
    def _extract_1d_axes(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize lon and lat to 1D axes:
        - Return them directly when they are already 1D.
        - For a 2D meshgrid, use lon[0, :] as x and lat[:, 0] as y.
        """
        lon = np.asarray(lon)
        lat = np.asarray(lat)

        if lon.ndim == 1 and lat.ndim == 1:
            return lon.copy(), lat.copy()
        if lon.ndim == 2 and lat.ndim == 2:
            # Assume a regular grid (rectilinear grid).
            x = lon[0, :].copy()
            y = lat[:, 0].copy()
            return x, y

        raise ValueError(f"Unsupported lon/lat shape: lon{lon.shape}, lat{lat.shape}")

    # ==== Added: resample u and v onto an evenly spaced grid. ====
    @staticmethod
    def _resample_to_regular_grid(
        x: np.ndarray,
        y: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample (x, y, u, v) onto an evenly spaced grid within the original range.
        x and y are 1D arrays with lengths nx and ny; u and v are 2D arrays (ny, nx).
        Apply two 1D linear interpolations (x first, then y) to avoid a scipy dependency.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        u = np.asarray(u)
        v = np.asarray(v)

        ny, nx = u.shape
        assert ny == y.size and nx == x.size, "u/v and lon/lat dimensions do not match"

        # Ensure x and y are increasing, as required by streamplot.
        x_order = np.argsort(x)
        y_order = np.argsort(y)

        if not np.all(x_order == np.arange(x.size)):
            x = x[x_order]
            u = u[:, x_order]
            v = v[:, x_order]

        if not np.all(y_order == np.arange(y.size)):
            y = y[y_order]
            u = u[y_order, :]
            v = v[y_order, :]

        # Build evenly spaced axes while preserving the original grid counts.
        new_x = np.linspace(x[0], x[-1], x.size)
        new_y = np.linspace(y[0], y[-1], y.size)

        # Interpolate along x first.
        u_x = np.empty((ny, new_x.size), dtype=u.dtype)
        v_x = np.empty((ny, new_x.size), dtype=v.dtype)
        for j in range(ny):
            u_x[j, :] = np.interp(new_x, x, u[j, :])
            v_x[j, :] = np.interp(new_x, x, v[j, :])

        # Then interpolate along y.
        u_new = np.empty((new_y.size, new_x.size), dtype=u.dtype)
        v_new = np.empty((new_y.size, new_x.size), dtype=v.dtype)
        for i in range(new_x.size):
            u_new[:, i] = np.interp(new_y, y, u_x[:, i])
            v_new[:, i] = np.interp(new_y, y, v_x[:, i])

        return new_x, new_y, u_new, v_new

    def plot_mxn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        ground_truth_u: np.ndarray,
        ground_truth_v: np.ndarray,
        prediction_u: np.ndarray,
        prediction_v: np.ndarray,
        all_init_times: list[datetime] = [],
    ) -> tuple[Figure, Axes]:
        assert len(ground_truth_u.shape) == 3
        assert ground_truth_u.shape[-2:] == lat.shape

        rows = 2  # gt/pred
        columns = ground_truth_u.shape[0]
        plt.close()
        fig, ax = plt.subplots(rows, columns, figsize=(20, 7), dpi=200, facecolor="w")

        # ground truth
        for j in range(columns):
            tmp_ax = ax[0, j]
            time_title = (
                all_init_times[j].strftime("%Y%m%d_%H%M") if all_init_times else ""
            )
            fig, tmp_ax = self.plot_bg(fig, tmp_ax)
            fig, tmp_ax = self._plot_wind(
                fig, tmp_ax, lon, lat, ground_truth_u[j], ground_truth_v[j], time_title
            )

        # prediction
        for j in range(columns):
            tmp_ax = ax[1, j]
            time_title = (
                all_init_times[j].strftime("%Y%m%d_%H%M") if all_init_times else ""
            )
            fig, tmp_ax = self.plot_bg(fig, tmp_ax)
            fig, tmp_ax = self._plot_wind(
                fig, tmp_ax, lon, lat, prediction_u[j], prediction_v[j], time_title
            )

        return fig, ax

    def plot_1x1(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        title: str = "",
    ) -> tuple[Figure, Axes]:

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=200, facecolor="w")
        fig, ax = super().plot_bg(fig, ax)
        fig, ax = self._plot_wind(fig, ax, lon, lat, u_wind, v_wind, title)

        return fig, ax

    def _plot_wind(
        self,
        fig: Figure,
        ax: Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        title: str = "",
        quiver_only: bool = False,
    ) -> tuple[Figure, Axes]:
        """
        Resample all input coordinates to an evenly spaced grid within the original range
        before plotting. If matplotlib.streamplot still reports nonuniform spacing,
        fall back to quiver to avoid a ValueError.
        """

        lon = np.asarray(lon)
        lat = np.asarray(lat)
        u_wind = np.asarray(u_wind, dtype=float)
        v_wind = np.asarray(v_wind, dtype=float)

        # ---- 1. Normalize lon/lat to 1D axes. ----
        if lon.ndim == 2 and lat.ndim == 2:
            x_raw = lon[0, :].copy()
            y_raw = lat[:, 0].copy()
        elif lon.ndim == 1 and lat.ndim == 1:
            x_raw = lon.copy()
            y_raw = lat.copy()
        else:
            raise ValueError(f"Unsupported lon/lat shape: lon{lon.shape}, lat{lat.shape}")

        if u_wind.ndim != 2:
            raise ValueError(f"u_wind should be 2D (ny, nx), got shape {u_wind.shape}")
        if u_wind.shape != v_wind.shape:
            raise ValueError("u_wind and v_wind must have the same shape")

        ny, nx = u_wind.shape
        if nx != x_raw.size or ny != y_raw.size:
            raise ValueError(
                f"u/v shape {u_wind.shape} not consistent with lon/lat "
                f"({y_raw.size}, {x_raw.size})"
            )

        # ---- 2. Ensure the original axes are increasing; sort and reorder u/v if needed. ----
        x_order = np.argsort(x_raw)
        if not np.all(x_order == np.arange(x_raw.size)):
            x_raw = x_raw[x_order]
            u_wind = u_wind[:, x_order]
            v_wind = v_wind[:, x_order]

        y_order = np.argsort(y_raw)
        if not np.all(y_order == np.arange(y_raw.size)):
            y_raw = y_raw[y_order]
            u_wind = u_wind[y_order, :]
            v_wind = v_wind[y_order, :]

        # ---- 3. Build evenly spaced axes manually using a fixed step. ----
        if nx > 1:
            dx = (float(x_raw[-1]) - float(x_raw[0])) / float(nx - 1)
        else:
            dx = 1.0
        if ny > 1:
            dy = (float(y_raw[-1]) - float(y_raw[0])) / float(ny - 1)
        else:
            dy = 1.0

        new_x = float(x_raw[0]) + dx * np.arange(nx, dtype=float)
        new_y = float(y_raw[0]) + dy * np.arange(ny, dtype=float)

        # Interpolate along x first.
        u_x = np.empty((ny, nx), dtype=u_wind.dtype)
        v_x = np.empty((ny, nx), dtype=v_wind.dtype)
        for j in range(ny):
            u_x[j, :] = np.interp(new_x, x_raw, u_wind[j, :])
            v_x[j, :] = np.interp(new_x, x_raw, v_wind[j, :])

        # Then interpolate along y.
        u_new = np.empty((ny, nx), dtype=u_wind.dtype)
        v_new = np.empty((ny, nx), dtype=v_wind.dtype)
        for i in range(nx):
            u_new[:, i] = np.interp(new_y, y_raw, u_x[:, i])
            v_new[:, i] = np.interp(new_y, y_raw, v_x[:, i])

        # ---- 4. Wind speed magnitude from the resampled field. ----
        scalar = np.hypot(u_new, v_new)

        # ---- 5. Prefer streamplot; fall back to quiver if spacing is still rejected. ----
        try:
            ax.streamplot(
                new_x,
                new_y,
                u_new,
                v_new,
                zorder=0,
                color="C0",
                linewidth=0.5,
                arrowsize=0.6,
            )
        except ValueError as e:
            # Catch only spacing-related errors; re-raise all other errors.
            if "equally spaced" in str(e) or "must be equally spaced" in str(e):
                # Fallback to quiver so plotting can continue.
                Xg, Yg = np.meshgrid(new_x, new_y)
                ax.quiver(
                    Xg,
                    Yg,
                    u_new,
                    v_new,
                    zorder=0,
                    angles="xy",
                    scale_units="xy",
                    scale=1.0,
                    width=0.001,
                )
            else:
                raise

        if title:
            ax.set_title(f"{title} {self.title_suffix}")

        # ---- 6. Use the original coordinates for the contourf below. ----
        if not quiver_only:
            # contourf works fine on irregular lon/lat
            conf = ax.contourf(
                x_raw,
                y_raw,
                scalar,
                levels=WSP_LV,
                colors=WSP_COLOR,
                zorder=-1,
            )

            # create an axes on the right side of ax. The width of cax will be 5%
            # of ax and the padding between cax and ax will be fixed at 0.05 inch.
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # colorbar
            cbar = fig.colorbar(conf, cax=cax)
            cbar.ax.set_title("$\\frac{m}{s}$")

        return fig, ax

    def plot_1xn(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        u_wind_list: np.ndarray,
        v_wind_list: np.ndarray,
        titles: list[str] = [],
        grid_on: bool = False,
    ) -> tuple[Figure, Axes]:

        """
        1 x N wind-field plots (for example, multiple time steps).
        u_list, v_list shape: (N, H, W)
        """

        assert len(u_wind_list.shape) == 3  # (N, H, W)
        assert u_wind_list.shape[-2:] == lat.shape[-2:]

        cols = u_wind_list.shape[0]
        plt.close()
        fig, ax = plt.subplots(1, cols, figsize=(2.57143 * cols, 2.5), dpi=200, facecolor="w")

        # Matplotlib returns a single Axes for one subplot; convert it to an array.
        if cols == 1:
            ax = np.array([ax])

        for j in range(cols):
            tmp_ax = ax[j]
            title = titles[j] if titles else ""

            # Draw the background.
            fig, tmp_ax = self.plot_bg(fig, tmp_ax, grid_on)

            # Draw the wind field; the current implementation checks whether resampling is needed.
            fig, tmp_ax = self._plot_wind(
                fig,
                tmp_ax,
                lon,
                lat,
                u_wind_list[j],
                v_wind_list[j],
                title,
            )

        return fig, ax


if __name__ == "__main__":
    target_time = datetime(2022, 10, 16, 0)
    u850 = gen_data(target_time, DataCompose(DataType.U, Level.Hpa850))
    v850 = gen_data(target_time, DataCompose(DataType.V, Level.Hpa850))
    data_lat = gen_data(target_time, DataCompose(DataType.Lat, Level.Surface))
    data_lon = gen_data(target_time, DataCompose(DataType.Lon, Level.Surface))

    viz = VizWind("Hpa850")
    fig, ax = viz.plot_1x1(data_lon, data_lat, u850, v850)
    fig.savefig(
        f"{FIGURE_PATH}/{target_time.strftime('%Y%m%d_%H%M')}_wind.png",
        transparent=False,
    )
    plt.close()
