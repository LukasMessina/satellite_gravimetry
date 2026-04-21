import numpy as np
import json


from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from helpers import transform_vector_history_inertial_to_satellite_frame, transform_vector_history_inertial_to_rtn
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection


import matplotlib as mpl
mpl.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Define global color palette
RED = "#92354C"
ORANGE = "#D78643"
BLUE = "#1E5AA8"
TEAL = "#2CB7B2"
BLACK = "#000000"
PRIMARY_COLORS = (
    RED,
    TEAL,
    BLUE,
    ORANGE,
    BLACK,
)
GRID_COLOR = "#C8CDD7"


def apply_publication_style():
    mpl.rcParams.update({
        # Fonts
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12,

        # Axes
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 1.2,
        "axes.grid": True,

        # Grid
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,

        # Ticks
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,

        # Legend
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "black",

    })


apply_publication_style()

class Plotter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        output_path.mkdir(parents=True, exist_ok=True)

    def plot_orbits(self, no_satellites, position_data, title, sat_labels, file_name):
        """Plot the 3D orbits of the satellites."""

        fig = plt.figure(figsize=(7.4, 6.8), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        # ax.set_title(title)

        earth_texture_path = Path("data/earth_texture.jpg")

        self._add_earth(ax, texture_path=earth_texture_path)

        linestyles = ["-", "--", "-.", ":"]
        linewidth = [4, 2]
        colors = self._get_colors_palette(no_satellites)


        for i in range(no_satellites):
            reference_color = colors[i]
            point_color = colors[i]
                
            ax.plot(position_data[i][:, 0], position_data[i][:, 1], position_data[i][:, 2], label=sat_labels[i],
                     linewidth=linewidth[i % len(linewidth)], color=reference_color, linestyle=linestyles[i % len(linestyles)])
            ax.scatter(
                position_data[i][0, 0], position_data[i][0, 1], position_data[i][0, 2],
                marker="o", s=80, color=point_color, edgecolor="k", linewidth=0.8,
                label=f"{sat_labels[i]} @ $t_0$"
            )
            ax.scatter(
                position_data[i][-1, 0], position_data[i][-1, 1], position_data[i][-1, 2],
                marker="x", s=110, color=point_color, linewidth=1.8,
                label=f"{sat_labels[i]} @ $t_f$"
            )

        data = np.vstack([position_data[i] for i in range(no_satellites)])
        self._set_equal_3d_axes(ax, data)

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.tick_params(axis="z", which="major", labelsize=12)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.zaxis.label.set_size(14)
        ax.legend(loc="upper right", fontsize=11, frameon=True)
        ax.grid(True)

        plt.tight_layout()
        fig.savefig(self.output_path / file_name, dpi=720)
        plt.close(fig)

    def plot_attitude_triads_orientation(
        self,
        dependent_variables_array: np.ndarray,
        grace_fo_position_data: list[np.ndarray],
        epoch_idx: int,
        file_name: str,
    ) -> None:
        """
        Plot GRACE C/B body-frame attitude triads at a selected epoch.

        The view is centered at the midpoint between both satellites and uses
        a plotting box to keep all axes equally scaled.
        """


        dependent_variables_array = np.asarray(dependent_variables_array, dtype=float)
        if dependent_variables_array.ndim != 2:
            raise ValueError("dependent_variables_array must be a 2D array.")
        if dependent_variables_array.shape[1] < 35:
            raise ValueError(
                "dependent_variables_array must include rotation matrices for both satellites "
            )
        if epoch_idx < 0 or epoch_idx >= dependent_variables_array.shape[0]:
            raise IndexError(
                f"epoch_idx={epoch_idx} is out of bounds for dependent_variables_array "
                f"with {dependent_variables_array.shape[0]} rows."
            )

        if len(grace_fo_position_data) != 2:
            raise ValueError("grace_fo_position_data must contain two arrays: GRACE C and GRACE D.")

        grace_c_position_history = np.asarray(grace_fo_position_data[0], dtype=float)
        grace_d_position_history = np.asarray(grace_fo_position_data[1], dtype=float)
        if grace_c_position_history.ndim != 2 or grace_d_position_history.ndim != 2:
            raise ValueError("Satellite position arrays must be 2D with shape (N, 3).")
        if grace_c_position_history.shape[1] != 3 or grace_d_position_history.shape[1] != 3:
            raise ValueError("Satellite position arrays must have 3 columns (x, y, z).")
        if epoch_idx >= grace_c_position_history.shape[0] or epoch_idx >= grace_d_position_history.shape[0]:
            raise IndexError(
                f"epoch_idx={epoch_idx} is out of bounds for provided position history lengths "
            )

        grace_c_position = grace_c_position_history[epoch_idx]
        grace_d_position = grace_d_position_history[epoch_idx]

        rotation_inertial_to_body_grace_c = dependent_variables_array[epoch_idx, 17:26].reshape(3, 3)
        rotation_inertial_to_body_grace_d = dependent_variables_array[epoch_idx, 26:35].reshape(3, 3)
        rotation_body_to_inertial_grace_c = rotation_inertial_to_body_grace_c.T
        rotation_body_to_inertial_grace_d = rotation_inertial_to_body_grace_d.T

        midpoint = 0.5 * (grace_c_position + grace_d_position)
        separation = np.linalg.norm(grace_d_position - grace_c_position)
        # Set triad scale and plotting box size based on satellite separation to ensure good visibility and equal axis scaling
        triad_scale = max(1.0, 0.2 * separation)
        box_half_extent = 0.5 * separation + 2.0 * triad_scale

        fig = plt.figure(figsize=(6.2, 5.8), dpi=300)
        ax = fig.add_subplot(111, projection="3d", proj_type="ortho")

        colors = (RED, BLUE, TEAL)
        labels = (r"$X_{SF}$", r"$Y_{SF}$", r"$Z_{SF}$")

        def _plot_triad(center: np.ndarray, rotation_body_to_inertial: np.ndarray, name: str) -> None:
            for i, (label, color) in enumerate(zip(labels, colors)):
                axis_direction = rotation_body_to_inertial[:, i]
                line_start = center
                line_end = center + triad_scale * axis_direction
                ax.plot(
                    [line_start[0], line_end[0]],
                    [line_start[1], line_end[1]],
                    [line_start[2], line_end[2]],
                    color=color,
                    linewidth=2.0,
                )
                text_point = center + 1.8 * triad_scale * axis_direction
                ax.text(*text_point, label, color=color, va="center", ha="center")

            ax.text(
                *center,
                name,
                color="k",
                va="center",
                ha="center",
                fontsize = 9,
                bbox={"fc": "w", "alpha": 0.8, "boxstyle": "circle,pad=0.25"},
            )

        _plot_triad(grace_c_position, rotation_body_to_inertial_grace_c, "C")
        _plot_triad(grace_d_position, rotation_body_to_inertial_grace_d, "D")

        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="None", color="k",
                markersize=6, label="C: GRACE C"),
            Line2D([0], [0], marker="o", linestyle="None", color="k",
                markersize=6, label="D: GRACE D"),
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper left",          # similar placement to your annotation
            frameon=True,
            borderaxespad=0.5,
            fontsize=9,
        )

        ax.set_xlim(midpoint[0] - box_half_extent, midpoint[0] + box_half_extent)
        ax.set_ylim(midpoint[1] - box_half_extent, midpoint[1] + box_half_extent)
        ax.set_zlim(midpoint[2] - box_half_extent, midpoint[2] + box_half_extent)
        ax.tick_params(axis="both", which="major", labelsize=10)
        ax.tick_params(axis="z", which="major", labelsize=10)
        ax.set_box_aspect([1.0, 1.0, 1.0])

        ax.set_xlabel("x [m]", fontsize=11)
        ax.set_ylabel("y [m]", fontsize=11)
        ax.set_zlabel("z [m]", fontsize=11)
        # ax.set_title(f"GRACE Attitude triads - epoch index {epoch_idx}", fontsize=10)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))   # always use scientific notation
            axis.set_major_formatter(formatter)

        ax.xaxis.get_offset_text().set_fontsize(9)
        ax.yaxis.get_offset_text().set_fontsize(9)
        ax.zaxis.get_offset_text().set_fontsize(9)

        fig.subplots_adjust(left=0.06, right=0.84, bottom=0.08, top=0.90)
        fig.savefig(self.output_path / file_name, bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)
    
    @staticmethod
    def _set_equal_3d_axes(
        ax, 
        data: np.ndarray, 
        additional_plane_padding: bool = False
        ) -> None:
        """Set 3D plot axes to equal scale."""
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        max_range = np.array([x.max() - x.min(),
                            y.max() - y.min(),
                            z.max() - z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) / 2.0
        mid_y = (y.max() + y.min()) / 2.0
        mid_z = (z.max() + z.min()) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        if additional_plane_padding:
            padding = 0.05 * max_range
            ax.set_xlim(mid_x - max_range - padding, mid_x + max_range + padding)
            ax.set_ylim(mid_y - max_range - padding, mid_y + max_range + padding)
            ax.set_zlim(mid_z - max_range - 0.20*max_range, mid_z + max_range + padding)
        ax.set_box_aspect([1, 1, 1])

    @staticmethod
    def _plot_3d_box_projections(
        ax,
        x_data: np.ndarray,
        y_data: np.ndarray,
        z_data: np.ndarray,
        color: str = "#4A4A4A",
        linewidth: float = 1.2,
        alpha: float = 0.85,
    ) -> None:
        """Project a 3D trajectory onto three bounding planes of the current axes box."""

        x_min, _ = ax.get_xlim3d()
        _, y_max = ax.get_ylim3d()
        z_min, _ = ax.get_zlim3d()

        ax.plot(
            np.full_like(x_data, x_min, dtype=float),
            y_data,
            z_data,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )
        ax.plot(
            x_data,
            np.full_like(y_data, y_max, dtype=float),
            z_data,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )
        ax.plot(
            x_data,
            y_data,
            np.full_like(z_data, z_min, dtype=float),
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1,
        )

    @staticmethod
    def _add_earth(ax, texture_path: Path | None) -> None:
        """
        Add an Earth sphere centered at the origin.
        """
        # Sphere parameterization
        n_lon, n_lat = 720, 720  # increase for smoother sphere, decrease for speed
        lon = np.linspace(-np.pi, np.pi, n_lon)
        lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
        lon2, lat2 = np.meshgrid(lon, lat)
        radius_m = 6378137.0 - 1100000.0

        x = radius_m * np.cos(lat2) * np.cos(lon2)
        y = radius_m * np.cos(lat2) * np.sin(lon2)
        z = radius_m * np.sin(lat2)

        # Read texture (expects equirectangular: width=2*height typically)
        img = mpimg.imread(str(texture_path))
        if img.dtype != np.float32 and img.dtype != np.float64:
            img = img.astype(np.float32) / 255.0

        # Map lon/lat -> texture coordinates (u in [0,1], v in [0,1])
        u = (lon2 + np.pi) / (2 * np.pi)
        v = (lat2 + np.pi / 2) / np.pi

        # Convert to pixel indices
        h, w = img.shape[0], img.shape[1]
        px = np.clip((u * (w - 1)).astype(int), 0, w - 1)
        py = np.clip(((1 - v) * (h - 1)).astype(int), 0, h - 1)  # flip vertical for image coords

        facecolors = img[py, px]

        ax.plot_surface(
            x, y, z,
            facecolors=facecolors,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=True,
            shade=True,   # IMPORTANT: keep False so texture colors are not altered
            alpha=1.0
        )

    def plot_relative_position(
            self,
            time_data,
            position_data,
            velocity_data, 
            first_figure_title, 
            first_file_name, 
            second_figure_title, 
            second_file_name
            ) -> np.ndarray:
        """
        Plot the relative position components in the RTN frame between the satellites pair over time and
        the 3 dimensional relative position time history in the RTN refernce frame of the chaser satellite.
        """

        t_days = (time_data - time_data[0]) / 86400.0
        # Along-track distance computation (RTN of chaser satellite)
        # Along-track unit vector 
        v_chaser_norm = np.linalg.norm(velocity_data[1], axis=1)
  
        r_hat = position_data[1] / np.linalg.norm(position_data[1], axis=1)[:, None]  
        h_vector = np.cross(position_data[1], velocity_data[1])
        h_hat = h_vector / np.linalg.norm(h_vector, axis=1)[:, None]
        t_hat = np.cross(h_hat, r_hat)/ np.linalg.norm(np.cross(h_hat, r_hat), axis=1)[:, None]
        

        # Relative position (target relative to chaser)
        relative_position = position_data[0] - position_data[1]

        # Along-track distance (signed)
        along_track_distance = np.einsum("ij,ij->i", relative_position, t_hat) / 1e3 # [km]
        radial_distance = np.einsum("ij,ij->i", relative_position, r_hat) / 1e3      # [km]
        cross_track_distance = np.einsum("ij,ij->i", relative_position, h_hat) / 1e3 # [km]
        relative_position_norm = np.linalg.norm(relative_position, axis=1)           # [m]
        line_colors = self._get_colors_palette(4)
        fig = plt.figure(figsize=(6.8, 5.8), dpi=360)
        plt.plot(t_days, along_track_distance, linewidth=3, label="Along-track (T)", color=line_colors[0])
        plt.plot(t_days, radial_distance, linewidth=2.5, label="Radial (R)", color=line_colors[3])
        plt.plot(t_days, cross_track_distance, linewidth=2.8, linestyle="--", label="Cross-track (N)", color=line_colors[2])
        plt.plot(t_days, relative_position_norm / 1e3, linewidth=1.8, linestyle="--", label=r"Range ($\rho$)", color=line_colors[1])

        # plt.title(first_figure_title)
        plt.xlabel("Propagation time [days]")
        plt.ylabel("Distance [km]")
        self._style_axes(plt.gca())
        plt.legend()
        plt.tight_layout()
        fig.savefig(self.output_path / first_file_name)
        plt.close(fig)

        fig = plt.figure(figsize=(9.2, 7.2), dpi=300)
        ax = fig.add_subplot(111, projection="3d")

        # Crop the data for better visualization of the relative trajectory shape
        cross_track_distance = cross_track_distance[:30000]
        radial_distance = radial_distance[:30000]
        along_track_distance = along_track_distance[:30000]
        propagation_time_hours = ((time_data - time_data[0]) / 3600.0)[:30000]


        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        relative_position_rtn = np.column_stack(
            (cross_track_distance * 1e3, radial_distance * 1e3, along_track_distance * 1e3)
        )
        trajectory_colored_segments = np.stack(
            (relative_position_rtn[:-1], relative_position_rtn[1:]),
            axis=1,
        )
        trajectory_color_norm = mpl.colors.Normalize(
            vmin=propagation_time_hours[0],
            vmax=propagation_time_hours[-1],
        )
        trajectory_collection = Line3DCollection(
            trajectory_colored_segments,
            cmap="viridis",
            norm=trajectory_color_norm,
            linewidth=1.5,
        )
        trajectory_collection.set_array(propagation_time_hours[:-1])
        ax.add_collection3d(trajectory_collection)


        ax.set_title(second_figure_title, fontsize=13)
        ax.set_xlabel("N [m]", labelpad=8)
        ax.set_ylabel("R [m]", labelpad=24)
        ax.set_zlabel("T [m]", labelpad=30)
        ax.grid(True)
        self._set_equal_3d_axes(ax, data=relative_position_rtn, additional_plane_padding=True)
        self._plot_3d_box_projections(
            ax,
            relative_position_rtn[:, 0],
            relative_position_rtn[:, 1],
            relative_position_rtn[:, 2],
        )

        ax.tick_params(axis="x", which="major", pad=2)
        ax.tick_params(axis="y", which="major", pad=14)
        ax.tick_params(axis="z", which="major", pad=18)

        colorbar = fig.colorbar(
            trajectory_collection,
            ax=ax,
            pad=0.12,
            fraction=0.035,
            shrink=0.82,
        )
        colorbar.set_label("Propagation time [hours]", rotation=90, labelpad=14)


        fig.tight_layout(rect=(0.0, 0.0, 0.92, 1.0))
        fig.savefig(self.output_path / second_file_name, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

        return relative_position_norm

    def plot_impulsive_delta_v_time_series(
        self,
        maneuver_log: list[dict[str, float | bool]],
        simulation_start_epoch: float,
        file_name: str = "grace_c_impulsive_delta_v_sequence.png",
    ) -> None:
        """Plot the signed impulsive delta-v sequence for GRACE C."""

        if not maneuver_log:
            return

        entry_times_hours = []
        entry_delta_v_values = []
        exit_times_hours = []
        exit_delta_v_values = []

        for maneuver in maneuver_log:
            entry_times_hours.append(
                (float(maneuver["trigger_epoch"]) - simulation_start_epoch) / 3600.0
            )
            entry_delta_v_values.append(float(maneuver["entry_delta_v"]))

            if maneuver.get("completed", False):
                exit_times_hours.append(
                    (float(maneuver["exit_epoch"]) - simulation_start_epoch) / 3600.0
                )
                exit_delta_v_values.append(float(maneuver["exit_delta_v"]))

        line_colors = self._get_colors_palette(2)
        fig = plt.figure(figsize=self._get_single_panel_size(), dpi=300)
        ax = fig.add_subplot(111)

        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.vlines(
            entry_times_hours,
            0.0,
            entry_delta_v_values,
            color=line_colors[0],
            linewidth=2.0,
            label="Entry impulse",
        )
        ax.scatter(
            entry_times_hours,
            entry_delta_v_values,
            color=line_colors[0],
            s=36,
            zorder=3,
        )

        if exit_times_hours:
            ax.vlines(
                exit_times_hours,
                0.0,
                exit_delta_v_values,
                color=line_colors[1],
                linewidth=2.0,
                label="Exit impulse",
            )
            ax.scatter(
                exit_times_hours,
                exit_delta_v_values,
                color=line_colors[1],
                s=36,
                zorder=3,
            )

        # ax.set_title(r"Impulsive $\Delta V$  sequence — GRACE C")
        ax.set_xlabel("Propagation time [hours]")
        ax.set_ylabel(r"$\Delta V$ [m/s]")
        self._style_axes(ax)
        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(self.output_path / file_name, bbox_inches="tight")
        plt.close(fig)

    def plot_custom_acceleration_time_series(
        self,
        dependent_variables_array: np.ndarray,
        states_array: np.ndarray,
        first_figure_title: str = "Thrust acceleration norm time evolution — GRACE-C",
        second_figure_title: str = "Thrust acceleration RTN components time evolution — GRACE-C",
    ) -> None:
        """
        Plot the custom acceleration norm and RTN components for GRACE-C.

        The RTN reference frame is built from the instantaneous GRACE-C state vector.
        """

        dependent_variables_array = np.asarray(dependent_variables_array, dtype=float)
        states_array = np.asarray(states_array, dtype=float)

        if dependent_variables_array.ndim != 2 or dependent_variables_array.shape[1] < 50:
            raise ValueError(
                "dependent_variables_array must be 2D with at least 50 columns to include custom acceleration."
            )
        if states_array.ndim != 2 or states_array.shape[1] < 7:
            raise ValueError("states_array must be 2D with at least 7 columns.")

        time_seconds = dependent_variables_array[:, 0]
        time_hours = (time_seconds - time_seconds[0]) / 3600.0

        inertial_thrust_acceleration = dependent_variables_array[:, 47:50]
        inertial_thrust_acceleration_acceleration_norm = np.linalg.norm(inertial_thrust_acceleration, axis=1)

        grace_c_position_history = states_array[:, 1:4]
        grace_c_velocity_history = states_array[:, 4:7]
        thrust_acceleration_rtn = transform_vector_history_inertial_to_rtn(
            inertial_thrust_acceleration,
            grace_c_position_history,
            grace_c_velocity_history,
        )

        fig = plt.figure(figsize=self._get_single_panel_size(), dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(
            time_hours,
            inertial_thrust_acceleration_acceleration_norm,
            "-o",
            color=RED,
            linewidth=1.5,
            markersize=2.5,
        )
        # ax.set_title(first_figure_title)
        ax.set_xlabel("Propagation time [hours]")
        ax.set_ylabel(r"$\|a_{\mathrm{thrust}}\|$ [m/s$^2$]")
        self._style_axes(ax)
        fig.tight_layout()
        fig.savefig(
            self.output_path / "grace_c_thrust_acc_norm_time_evolution.png",
            bbox_inches="tight",
        )
        plt.close(fig)

        fig = plt.figure(figsize=self._get_single_panel_size(), dpi=300)
        ax = fig.add_subplot(111)
        component_labels = ("R", "T", "N")
        component_colors = self._get_colors_palette(4)[[0, 2, 3]]
        component_linestyles = ("--", "-", ":")
        linewidths = (5, 3, 2)

        for component_idx, (component_label, component_color, component_linestyle, linewidth) in enumerate(
            zip(component_labels, component_colors, component_linestyles, linewidths)
        ):
            ax.plot(
                time_hours,
                thrust_acceleration_rtn[:, component_idx],
                color=component_color,
                linewidth=linewidth,
                label=component_label,
                linestyle=component_linestyle,
            )

        # ax.set_title(second_figure_title)
        ax.set_xlabel("Propagation time [hours]")
        ax.set_ylabel(r"$a_{\mathrm{thrust,RTN}}$ [m/s$^2$]")
        self._style_axes(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(
            self.output_path / "grace_c_thrust_acc_rtn_components_time_evolution.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    def plot_rtn_error_projections(self, samples_rtn: np.ndarray, sigma_rtn: np.ndarray, file_name: str):
        """Plot RTN projections of noise samples with 1σ/2σ/3σ ellipses."""
        planes = [
            (0, 1, "R", "T"),  # R-T plane
            (0, 2, "R", "N"),  # R-N plane
            (1, 2, "T", "N"),  # T-N plane
        ]

        ellipse_angle = np.linspace(0.0, 2.0 * np.pi, 400)
        colors = self._get_colors_palette(5)
        confidence_interval_colors = ["#000000", "#555555", "#AAAAAA"]


        fig = plt.figure(figsize=(14, 4.8))
        for i, (a, b, la, lb) in enumerate(planes, start=1):
            ax = fig.add_subplot(1, 3, i)

            # Scatter of samples projected into the plane
            ax.scatter(samples_rtn[:, a], samples_rtn[:, b], s=1, alpha=0.9, color=RED)

            for ksig, reference_color in zip((1, 2, 3), confidence_interval_colors):

                xa = ksig * sigma_rtn[a] * np.cos(ellipse_angle)
                xb = ksig * sigma_rtn[b] * np.sin(ellipse_angle)
                ax.plot(xa, xb, linewidth=2, label=f"{ksig}σ", color=reference_color)

            ax.set_xlabel(f"{la} [m]")
            ax.set_ylabel(f"{lb} [m]")
            # ax.set_title(f"{la}-{lb} plane projection")
            ax.axis("equal")
            self._style_axes(ax)
            ax.legend(loc="upper right")

        # fig.suptitle("RTN noise samples and uncertainty ellipses")
        plt.tight_layout()
        fig.savefig(self.output_path / file_name)
        plt.close(fig)

    def plot_acceleration_finite_difference_statistics(
        self,
        scenario: str,
        time: np.ndarray,
        results: dict[int, dict],
    ) -> None:
        """
        Create plots for finite-difference acceleration validation.

        Produces:
          - plots of the error components time evolution, overlaying all accuracy orders
          - plots of the error norm time evolution, overlaying all accuracy orders and including RMS in legend

        Parameters
        ----------
        satellite_name : str
            Identifier used in titles and filenames.
        time : np.ndarray, shape (N,)
            Time stamps [s].
        results : dict[int, dict]
            Output of validate_numerical_position_differentiation keyed by accuracy order.
            Must contain keys: 'error_vector', 'error_norm', 'error_rms'.
        file_prefix : str | None
            Optional prefix for filenames. If None, derived from satellite_name.
        """
        if time.ndim != 1:
            raise ValueError("Time must be a 1D array.")
        file_prefix = scenario.replace(" ", "_")
        propagation_time = time - time[0] # [s] time since start


        # -------------
        # Component plots
        # -------------
        comp_labels = [
            r"$|\epsilon_{a_x}(t)|$",
            r"$|\epsilon_{a_y}(t)|$",
            r"$|\epsilon_{a_z}(t)|$",
        ]

        file_suffix = ["x", "y", "z"]
        comp_titles = [
            r"Acceleration error component: $\epsilon_{a_x}(t)$",
            r"Acceleration error component: $\epsilon_{a_y}(t)$",
            r"Acceleration error component: $\epsilon_{a_z}(t)$",
        ]

        for j in range(3):
            fig = plt.figure(figsize=(8.5, 5), dpi=160)
            ax = fig.add_subplot(111)

            for accuracy in sorted(results.keys()):
                error_vector = np.asarray(results[accuracy]["error_vector"], dtype=float)
                
                linestyle = "-"
                linewidth = 2.5
                color = None

                if accuracy in (8, 12, 14):
                    linestyle = "--"
                    linewidth = 1.5

                if accuracy == 10:
                    color = "#84E42A"
                if accuracy == 14:
                    color = "#E9C236"
                

                ax.plot(
                    propagation_time,
                    np.abs(error_vector[:, j]),
                    linestyle,
                    linewidth=linewidth,
                    label=rf"$p={accuracy}$",
                    color=color,
                )

            ax.set_title(rf"{scenario} — {comp_titles[j]}", pad=14)
            ax.set_xlabel("Propagation Time [s]")
            ax.set_ylabel(rf"{comp_labels[j]} [m/s$^2$] ")
            ax.set_yscale("log")


            self._style_axes(ax, publication_style=False)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.20),   # below x-axis
                ncol=min(5, len(results)),     # one row (up to 5 columns; adjust if needed)
                frameon=True,
                fontsize=10,
                handlelength=2.0,
                columnspacing=1.2,
            )

            fig.tight_layout()
            fig.subplots_adjust(bottom=0.25)   # add room for legend below
            fig.savefig(self.output_path / f"{file_prefix}_acceleration_error_component_{file_suffix[j]}.png", bbox_inches="tight")
            plt.close(fig)

        # -------------
        # Norm plot with RMS in legend
        # -------------
        fig = plt.figure(figsize=(9.5, 5.3), dpi=160)
        ax = fig.add_subplot(111)

        for accuracy in sorted(results.keys()):
            error_norm = np.asarray(results[accuracy]["error_norm"], dtype=float).reshape(-1)

            rms = float(results[accuracy]["error_rms"])

            linestyle = "-"
            linewidth = 2.5
            color = None
            if accuracy in (12, 14):
                linestyle = "--"
                linewidth = 1.5
            if accuracy == 10:
                color = "#84E42A"
            if accuracy == 14:
                color = "#E9C236"


            ax.plot(
                propagation_time,
                error_norm,
                linestyle,
                linewidth=linewidth,
                label=rf"$p={accuracy}\;(\mathrm{{RMS}}={rms:.3e}\,\mathrm{{m\,s^{{-2}}}})$",
                color=color,
            )

        ax.set_title(rf"{scenario} — Acceleration error norm $\|\epsilon(t)\|$", pad=10)
        ax.set_xlabel("Propagation Time [s]")
        ax.set_ylabel(r"$\|\epsilon(t)\|$ [m/s$^2$]")
        ax.set_yscale("log")

        self._style_axes(ax, publication_style=False)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),   # below x-axis
            ncol=min(3, len(results)),     # one row (up to 5 columns; adjust if needed)
            frameon=True,
            fontsize=10,
            handlelength=2.0,
            columnspacing=1.2,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.10)   # add room for legend below
        out = self.output_path / f"{file_prefix}_acceleration_error_norm.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

    def plot_los_intersatellite_acceleration_finite_difference_statistics(
        self,
        scenario: str,
        time: np.ndarray,
        results: dict[int, dict],
    ) -> None:
        """
        Plot LOS-projected inter-satellite acceleration validation results.

        Produces:
        - Absolute Error with legend including RMS for each accuracy order (log y)

        """
        if time.ndim != 1:
            raise ValueError("Time must be a 1D array.")

        file_prefix = scenario.replace(" ", "_")
        propagation_time = time - time[0]

        fig = plt.figure(figsize=(9.5, 6), dpi=160)
        ax = fig.add_subplot(111)

        for accuracy in sorted(results.keys()):
            absolute_error = np.asarray(results[accuracy]["absolute_error"], dtype=float).reshape(-1)
            error_rms = float(results[accuracy]["error_rms"])

            linestyle = "None"
            marker = "o"
            color = None
            markersize = 2
            markerevery = None

            if accuracy == 18:
                color = "#84E42A"
                markerevery = 150

            ax.plot(
                propagation_time,
                absolute_error,
                linestyle=linestyle,
                markersize=markersize,
                marker=marker,
                color=color,
                label=rf"$p={accuracy}\;(\mathrm{{RMS}}={error_rms:.3e}\,\mathrm{{m\,s^{{-2}}}})$",
                markevery = markerevery
            )

        ax.set_title(rf"{scenario} ", pad=10)
        ax.set_xlabel("Propagation Time [s]")
        ax.set_ylabel(r"$|\epsilon_{a_{\mathrm{LOS}}(t)}|$ [m/s$^2$]")
        ax.set_yscale("log")

        self._style_axes(ax, article_style=False)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(3, len(results)),
            frameon=True,
            fontsize=10,
            handlelength=2.0,
            columnspacing=1.2,
            markerscale=2,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.10)
        fig.savefig(self.output_path / f"{file_prefix}.png", bbox_inches="tight")
        plt.close(fig)

    def plot_numerical_derivative_statistics(
        self,
        scenario: str,
        time: np.ndarray,
        results: dict[int, dict],
        file_name: str,
        ylabel: str,
        title: str,
        rms_unit_label: str,
        time_samples_window_size: int  = 8640,
    ) -> None:
        """Plot finite-difference schemes numerical derivation absolute errors for several accuracy orders."""

        if time.ndim != 1:
            raise ValueError("Time must be a 1D array.")

        time_samples_window_size = min(len(time), time_samples_window_size)

        propagation_time = time - time[0]
        propagation_time = propagation_time[:time_samples_window_size]

        fig = plt.figure(figsize=(9.5, 6), dpi=160)
        ax = fig.add_subplot(111)

        for accuracy in sorted(results.keys()):
            absolute_error = np.asarray(results[accuracy]["absolute_error"], dtype=float).reshape(-1)
            absolute_error = absolute_error[:time_samples_window_size]

            error_rms = float(results[accuracy]["error_rms"])

            color = None
            marker = "o"
            markersize = 2
            linestyle = "None"

            ax.plot(
                propagation_time,
                absolute_error,
                color=color,
                linestyle=linestyle,
                label=rf"$p={accuracy}\;(\mathrm{{RMS}}={error_rms:.3e}\,{rms_unit_label})$",
                marker=marker,
                markersize=markersize,
            )

        # ax.set_title(rf"{scenario} — {title}", pad=10)
        ax.set_xlabel("Propagation Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")

        self._style_axes(ax)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(3, len(results)),
            frameon=True,
            fontsize=11,
            handlelength=2.0,
            columnspacing=1.2,
        )

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.10)
        fig.savefig(self.output_path / file_name, bbox_inches="tight")
        plt.close(fig)

    def plot_lgd_error_propagation_time_series(
        self,
        time: np.ndarray,
        lgd_error: np.ndarray,
        file_name: str,
        title: str = "LGD Error Time Evolution",
    ) -> None:
        """Plot the propagated LGD error time series"""

        time = np.asarray(time, dtype=float).reshape(-1)
        lgd_error = np.asarray(lgd_error, dtype=float).reshape(-1)

        if not (
            time.shape == lgd_error.shape
        ):
            raise ValueError("All LGD error-propagation histories must share the same length.")

        finite_mask = np.isfinite(lgd_error)
        if not np.any(finite_mask):
            raise ValueError("At least one finite LGD error sample is required for plotting.")

        time_hours = (time - time[0]) / 3600.0

        lgd_error_rms = float(np.sqrt(np.mean(lgd_error[finite_mask]**2)))

        fig = plt.figure(figsize=self._get_single_panel_size(), dpi=240)
        ax = fig.add_subplot(111)

        ax.plot(
            time_hours,
            lgd_error,
            color=RED,
            linewidth=1.8,
            label=(
                r"$\epsilon_{\mathrm{LGD}}$ "
                rf"($\mathrm{{RMS}}={lgd_error_rms:.3e}\,\mathrm{{m\,s^{{-2}}}}$)"
            ),
        )

        # ax.set_title(title)
        ax.set_xlabel("Propagation time [hours]")
        ax.set_ylabel(r"Acceleration error [m/s$^2$]")
        self._style_axes(ax)
        ax.legend(loc="best", frameon=True)

        fig.tight_layout()
        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_lgd_error_propagation_asd(
        self,
        frequencies: np.ndarray,
        lgd_error_asd: np.ndarray,
        file_name: str,
        title: str = "LGD Error ASD Spectrum",
        x_limit_inf: float | None = 1e-5,
        x_limit_sup: float | None = 1e-1,
        reference_orbital_period_seconds: float | None = None,
    ) -> None:
        """Plot the ASD of the LGD error."""

        frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
        lgd_error_asd = np.asarray(lgd_error_asd, dtype=float).reshape(-1)

        fig = plt.figure(figsize=self._get_single_panel_size(), dpi=240)
        ax = fig.add_subplot(111)

        ax.loglog(
            frequencies,
            lgd_error_asd,
            color=RED,
            linewidth=2.0,
            label = r"$\mathrm{ASD}\,\!\left(\epsilon_{\mathrm{LGD}}\right)$"
            )

        # ax.set_title(title, pad=14.0)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"ASD [m s$^{-2}$ Hz$^{-1/2}$]")
        self._style_axes(ax)
        ax.legend(loc="best", frameon=True)

        if x_limit_inf is not None and x_limit_sup is not None:
            ax.set_xlim(x_limit_inf, x_limit_sup)

        self._add_cycles_per_revolution_secondary_xaxis(
            ax=ax,
            reference_orbital_period_seconds=reference_orbital_period_seconds,
        )

        fig.tight_layout()
        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    @staticmethod
    def _get_colors_palette(number_of_colors: int) -> list[str]:
        return [
            PRIMARY_COLORS[color_index % len(PRIMARY_COLORS)]
            for color_index in range(number_of_colors)
        ]

    @staticmethod
    def _get_single_panel_size() -> tuple[float, float]:
        return (7.2, 6.0)

    @staticmethod
    def _get_article_three_row_size() -> tuple[float, float]:
        return (13.2, 7.2)

    def _style_axes(self, ax: plt.Axes, *, publication_style: bool = True) -> None:
        ax.minorticks_on()

        if not publication_style:
            ax.spines["left"].set_linewidth(1.2)
            ax.spines["bottom"].set_linewidth(1.2)
            ax.spines["top"].set_linewidth(1.2)
            ax.spines["right"].set_linewidth(1.2)
            ax.set_axisbelow(True)
            ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.6)
            ax.grid(True, which="minor", linestyle="-", linewidth=0.7, alpha=0.45)
            return

        ax.spines["left"].set_linewidth(1.15)
        ax.spines["bottom"].set_linewidth(1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.55, color=GRID_COLOR)
        ax.grid(False, which="minor")
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            labelsize=13,
            width=0.95,
            length=5.5,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            width=0.75,
            length=3.0,
        )
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
        ax.title.set_size(15)

    def _add_cycles_per_revolution_secondary_xaxis(
        self,
        ax: plt.Axes,
        reference_orbital_period_seconds: float | None,
    ) -> None:
        """Add a top x-axis expressing frequency in cycles per revolution."""

        if reference_orbital_period_seconds is None:
            return
        if reference_orbital_period_seconds <= 0.0:
            raise ValueError("reference_orbital_period_seconds must be positive when provided.")

        def convert_hertz_to_cycle_per_revolution(frequency_hz: np.ndarray | float) -> np.ndarray:
            return np.asarray(frequency_hz, dtype=float) * reference_orbital_period_seconds

        def convert_cycle_per_revolution_to_hertz(frequency_cpr: np.ndarray | float) -> np.ndarray:
            return np.asarray(frequency_cpr, dtype=float) / reference_orbital_period_seconds

        secondary_xaxis = ax.secondary_xaxis(
            "top",
            functions=(
                convert_hertz_to_cycle_per_revolution,
                convert_cycle_per_revolution_to_hertz,
            ),
        )
        secondary_xaxis.set_xscale(ax.get_xscale())
        secondary_xaxis.set_xlabel("Frequency [CPR]", labelpad=10.0, fontsize=14)
        secondary_xaxis.tick_params(axis="x", which="both", direction="in", labelsize=12)

    def plot_pointing_angles_asd(
        self,
        file_name: str,
        pitch_history_json_path: Path,
        yaw_history_json_path: Path,
        roll_history_json_path: Path
    ) -> None:
        """Plot ASD of roll/pitch/yaw pointing angles from plot-digitizer JSON files."""

        json_paths = [roll_history_json_path, pitch_history_json_path, yaw_history_json_path]
        color = self._get_colors_palette(3)
        labels = ["Roll", "Pitch", "Yaw"]

        fig = plt.figure(figsize=self._get_single_panel_size())

        for i, path in enumerate(json_paths):
            with open(path, "r") as f:
                data = json.load(f)

            x = np.array([float(d["x"]) for d in data], dtype=float)
            y = np.array([float(d["y"]) for d in data], dtype=float)
            # Sort by frequency
            idx = np.argsort(x)
            # Match the paper legend colors: Roll=blue, Pitch=red, Yaw=green
            plt.loglog(x[idx],  y[idx],  linewidth=2, color=color[i],  label=labels[i])

        # plt.title("ASD of Pointing Angles (plot digitizer)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"$\mathrm{ASD}\left(\theta_x,\ \theta_y,\ \theta_z\right)\ [\mathrm{rad}\,\mathrm{Hz}^{-1/2}]$") 
        self._style_axes(plt.gca())
        plt.legend(loc="upper right", frameon=True)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_pointing_angle_asd_combination_effect(
        self,
        frequencies: np.ndarray,
        star_camera_assembly_asd_roll: np.ndarray,
        star_camera_assembly_asd_pitch_yaw: np.ndarray,
        inertial_measurement_unit_pointing_angle_asd: np.ndarray,
        combined_asd_roll: np.ndarray,
        combined_asd_pitch_yaw: np.ndarray,
        file_name: str,
    ) -> None:
        """Plot the star-camera, IMU, and combined pointing-angle ASD models."""

        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.ndim != 1:
            raise ValueError("frequencies must be a 1D array.")

        colors = self._get_colors_palette(5)
        data = [
            ("Star Camera Roll Noise", np.asarray(star_camera_assembly_asd_roll, dtype=float), colors[2], "-", 2.0),
            ("Star Camera Pitch/Yaw Noise", np.asarray(star_camera_assembly_asd_pitch_yaw, dtype=float), colors[0], "-", 2.0),
            ("IMU Isotropic Noise", np.asarray(inertial_measurement_unit_pointing_angle_asd, dtype=float), colors[1], "-", 2.8),
            ("Combined Roll Noise", np.asarray(combined_asd_roll, dtype=float), colors[3], "--", 2.0),
            ("Combined Pitch/Yaw Noise", np.asarray(combined_asd_pitch_yaw, dtype=float), colors[4], ":", 2.0),
        ]

        fig = plt.figure(figsize=self._get_single_panel_size())
        ax = fig.add_subplot(111)

        for label, asd_values, color, linestyle, linewidth in data:
            if asd_values.shape != frequencies.shape:
                raise ValueError(
                    f"{label} ASD must have the same shape as the frequencies span."
                )

            ax.loglog(
                frequencies,
                asd_values,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
            )

        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"ASD [rad Hz$^{-1/2}$]")
        ax.set_xlim(
            1e-5,
            1e-1,
        )
        ax.set_ylim(
            1e-8,
            1e-0,
        )
        self._style_axes(ax)
        ax.legend(loc="best", frameon=True)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_relative_position_error_asd(
        self,
        file_name: str,
        relative_position_error_asd_json_path: Path,
    ) -> None:
        """
        Plot the amplitude spectral density (ASD) of the relative position error.

        The spectrum is based on Figure 6.3 from:

        Teixeira da Encarnação, J. G. (2015).
        "Next-generation satellite gravimetry for measuring mass transport in the Earth system."
        PhD dissertation, Delft University of Technology.

        The data have been reconstructed from the original figure using a plot-digitizer
        (JSON export), and therefore represent an approximate numerical reproduction
        of the published results.
        """

        fig = plt.figure(figsize=self._get_single_panel_size())
        ax = fig.add_subplot(111)

        with open(relative_position_error_asd_json_path, "r") as f:
            data = json.load(f)

            x = np.array([float(d["x"]) for d in data], dtype=float)
            y = np.array([float(d["y"]) for d in data], dtype=float)
            # Convert the digitized frequencies from Hz to mHz for display.
            idx = np.argsort(x)
            x_mhz = x[idx] * 1e3
            ax.loglog(x_mhz, y[idx], linewidth=1.8, color=RED)

        # ax.set_title("Relative Position Error ASD")
        ax.set_xlabel("Frequency [mHz]")
        ax.set_ylabel(r"ASD [m Hz$^{-1/2}$]")
        ax.set_ylim(1e-4, 1e0)
        self._style_axes(ax)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_linear_interpolation_comparison(
        self,
        original_frequencies: np.ndarray,
        original_asd_values: np.ndarray,
        interpolated_frequencies: np.ndarray,
        interpolated_asd_values: np.ndarray,
        file_name: str,
        ordinate_label = r"ASD [rad Hz$^{-1/2}$]",
    ) -> None:
        """Plot comparison between original and linearly interpolated ASD data."""

        colors = self._get_colors_palette(2)
        fig = plt.figure(figsize=self._get_single_panel_size())

        plt.loglog(original_frequencies,  original_asd_values, color=colors[0], label='Original', linewidth=2.5)
        plt.loglog(interpolated_frequencies, interpolated_asd_values, color=colors[1], label='Interpolated', linewidth=1.8, linestyle='--')

        # plt.title("ASD Data: Original vs Interpolated")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(ordinate_label)
        self._style_axes(plt.gca())
        plt.legend(loc="upper right", frameon=True)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_angle_noise_time_series(
        self,
        noise_time_series,
        file_name: str,
        ordinate_label: str,
    ) -> None:
        """Plot time series of pointing angle noise."""

        fig = plt.figure(figsize=self._get_single_panel_size())

        plt.plot(noise_time_series.sample_times, noise_time_series, color=RED, linewidth=1.8)

        # plt.title("Pointing Angle Noise Time Series")
        plt.xlabel("Time [s]")
        plt.ylabel(ordinate_label)
        self._style_axes(plt.gca())

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_absolute_position_error_time_series(
        self,
        noise_time_series: list | np.ndarray,
        file_name: str
    ) -> None:
        """Plot time series of absolute position noise."""

        if (
            isinstance(noise_time_series, list)
            and len(noise_time_series) > 0
        ):
            component_labels = (r"$r_x$", r"$r_y$", r"$r_z$")
            fig, axes = plt.subplots(1, len(noise_time_series), figsize=(12.6, 4.8), dpi=300, sharex=True)
            axes = np.atleast_1d(axes)

            for component_idx, (ax, component_time_series, label) in enumerate(zip(axes, noise_time_series, component_labels)):

                ax.plot(
                    component_time_series.sample_times,
                    component_time_series,
                    color=RED,
                    linewidth=1.8,
                    label=label,
                )
                # ax.set_title(f"{component_label} Component")
                ax.set_xlabel("Time [s]")
                if component_idx == 0:
                    ax.set_ylabel("Position Error [m]")
                ax.legend(loc="upper right", frameon=True, fontsize=13)  

                self._style_axes(ax)

            # fig.suptitle("Absolute Inertial Position Noise Time Series", y=0.97)
            fig.tight_layout()
            fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return

        fig = plt.figure(figsize=self._get_single_panel_size())
        plt.plot(noise_time_series.sample_times, noise_time_series, color=RED, linewidth=1.8)
        # plt.title("Absolute Inertial Position Noise Time Series")
        plt.xlabel("Time [s]")
        plt.ylabel("Position Error [m]")
        self._style_axes(plt.gca())
        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_welch_estimated_psd_comparison(
        self,
        estimated_frequencies: np.ndarray,
        estimated_psd_values: np.ndarray,
        input_frequencies: np.ndarray,
        input_psd_values: np.ndarray,
        file_name: str,
        ordinate_label: str = r"PSD [rad$^2$ Hz$^{-1}$]",
        title: str = "Pointing Angle PSD",
        x_limit_inf: float | None = None,
        x_limit_sup: float | None = None
    ) -> None:
        """Plot comparison between Welch estimated PSD and input PSD."""

        if isinstance(estimated_frequencies, np.ndarray):
            if estimated_frequencies.ndim == 1:
                estimated_frequencies = [estimated_frequencies]
        if isinstance(estimated_frequencies, list):
                estimated_frequencies =  [np.asarray(component_frequencies, dtype=float) for component_frequencies in estimated_frequencies]
        if isinstance(estimated_psd_values, np.ndarray):
            if estimated_psd_values.ndim == 1:
                estimated_psd_values = [estimated_psd_values]
        if isinstance(estimated_psd_values, list):
                estimated_psd_values =  [np.asarray(component_psd_values, dtype=float) for component_psd_values in estimated_psd_values]

        if isinstance(input_frequencies, np.ndarray):
            if input_frequencies.ndim == 1:
                input_frequencies = [input_frequencies]
        if isinstance(input_frequencies, list):
                input_frequencies = [np.asarray(component_frequencies, dtype=float) for component_frequencies in input_frequencies]
        if isinstance(input_psd_values, np.ndarray):
            if input_psd_values.ndim == 1:
                input_psd_values = [input_psd_values]
        if isinstance(input_psd_values, list):
                input_psd_values = [np.asarray(component_psd_values, dtype=float) for component_psd_values in input_psd_values]

        if len(estimated_frequencies) != len(estimated_psd_values):
            raise ValueError("Estimated frequency and PSD inputs must contain the same number of components.")
        if len(input_frequencies) != len(input_psd_values):
            raise ValueError("Input frequency and PSD inputs must contain the same number of components.")

        if len(input_frequencies) == 1 and len(estimated_frequencies) > 1:
            input_frequencies = input_frequencies * len(estimated_frequencies)
            input_psd_values = input_psd_values * len(estimated_psd_values)
        elif len(input_frequencies) != len(estimated_frequencies):
            raise ValueError(
                "Input PSD data must either provide one common spectrum or one spectrum per component."
            )

        if len(estimated_frequencies) > 1:
            component_labels = ("x", "y", "z")
            colors = self._get_colors_palette(2)
            fig, axes = plt.subplots(1, len(estimated_frequencies), figsize=(12.6, 4.8), dpi=300, sharey=True)
            axes = np.atleast_1d(axes)

            for component_idx, ax in enumerate(axes):
                component_label = component_labels[component_idx]
                first_line, = ax.loglog(
                    estimated_frequencies[component_idx],
                    estimated_psd_values[component_idx],
                    color=colors[0],
                    label="Welch Estimation",
                    linewidth=2,
                )
                second_line, = ax.loglog(
                    input_frequencies[component_idx],
                    input_psd_values[component_idx],
                    color=colors[1],
                    label="Input",
                    linewidth=1.5,
                )
                ax.set_xlabel("Frequency [Hz]")
                if component_idx == 0:
                    ax.set_ylabel(ordinate_label)
                self._style_axes(ax)

                if x_limit_inf is not None and x_limit_sup is not None:
                    ax.set_xlim(x_limit_inf, x_limit_sup)

                ax.set_title(f"{component_label} component", fontsize = 13)

            fig.legend(
                [first_line, second_line],
                ["Welch Estimation", "Input"],
                loc="lower center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=2,
                frameon=True,
                fontsize=13,
            )
            # fig.suptitle(title)
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.22)
            fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
            plt.close(fig)
            return

        colors = self._get_colors_palette(2)
        fig = plt.figure(figsize=self._get_single_panel_size())
        plt.loglog(
            estimated_frequencies[0],
            estimated_psd_values[0],
            color=colors[0],
            label='Welch Estimation',
            linewidth=1.8,
        )
        plt.loglog(
            input_frequencies[0],
            input_psd_values[0],
            color=colors[1],
            label='Input',
            linewidth=1.5,
            linestyle='--',
        )
        # plt.title(title)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(ordinate_label)
        self._style_axes(plt.gca())
        plt.legend(loc="best", frameon=True)

        if x_limit_inf is not None and x_limit_sup is not None:
            plt.xlim(x_limit_inf, x_limit_sup)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_welch_estimated_asd(
        self,
        frequencies: np.ndarray,
        asd_values: np.ndarray,
        file_name: str,
        ordinate_label: str,
        title: str,
        line_label: str = "Estimation",
        x_limit_inf: float | None = None,
        x_limit_sup: float | None = None,
        reference_orbital_period_seconds: float | None = None,
    ) -> None:
        """Plot the Welch-estimated ASD spectrum."""

        frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
        asd_values = np.asarray(asd_values, dtype=float).reshape(-1)

        fig = plt.figure(figsize=self._get_single_panel_size())
        ax = fig.add_subplot(111)

        ax.loglog(
            frequencies,
            asd_values,
            color=RED,
            linewidth=2.0,
            label=line_label,
        )

        # ax.set_title(title, pad=14.0)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(ordinate_label)
        self._style_axes(ax)
        ax.legend(loc="best", frameon=True)

        if x_limit_inf is not None and x_limit_sup is not None:
            ax.set_xlim(x_limit_inf, x_limit_sup)

        self._add_cycles_per_revolution_secondary_xaxis(
            ax=ax,
            reference_orbital_period_seconds=reference_orbital_period_seconds,
        )

        fig.tight_layout()
        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_welch_estimated_asd_comparison(
        self,
        estimated_frequencies: np.ndarray,
        estimated_asd_values: np.ndarray,
        reference_frequencies: np.ndarray,
        reference_asd_values: np.ndarray,
        file_name: str,
        ordinate_label: str,
        title: str,
        estimated_label: str = "Estimation",
        reference_label: str = "Analytical",
        x_limit_inf: float | None = None,
        x_limit_sup: float | None = None,
        reference_orbital_period_seconds: float | None = None,
    ) -> None:
        """Plot a Welch-estimated ASD against a reference ASD curve."""

        estimated_frequencies = np.asarray(estimated_frequencies, dtype=float).reshape(-1)
        estimated_asd_values = np.asarray(estimated_asd_values, dtype=float).reshape(-1)
        reference_frequencies = np.asarray(reference_frequencies, dtype=float).reshape(-1)
        reference_asd_values = np.asarray(reference_asd_values, dtype=float).reshape(-1)

        colors = self._get_colors_palette(2)
        fig = plt.figure(figsize=self._get_single_panel_size())
        ax = fig.add_subplot(111)

        ax.loglog(
            estimated_frequencies,
            estimated_asd_values,
            color=colors[0],
            linewidth=1.6,
            label=estimated_label,
        )
        ax.loglog(
            reference_frequencies,
            reference_asd_values,
            color=colors[1],
            linewidth=1.6,
            linestyle="--",
            label=reference_label,
        )

        # ax.set_title(title, pad=14.0)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(ordinate_label)
        self._style_axes(ax)
        ax.legend(loc="best", frameon=True)

        if x_limit_inf is not None and x_limit_sup is not None:
            ax.set_xlim(x_limit_inf, x_limit_sup)

        self._add_cycles_per_revolution_secondary_xaxis(
            ax=ax,
            reference_orbital_period_seconds=reference_orbital_period_seconds,
        )

        fig.tight_layout()
        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_spectral_sensitivity_analysis_results(
        self,
        spectral_sensitivity_analysis: list[dict[str, float]],
        file_name: str,
        title: str,
        colorbar_label: str,
    ) -> None:
        """Plot the Welch parameter sweep used to fit numerical and analytical spectra."""

        if not spectral_sensitivity_analysis:
            raise ValueError("At least one spectral sensitivity analysis record is required.")

        segment_strides = np.array([record["segment_stride"] for record in spectral_sensitivity_analysis], dtype=float)
        lowest_frequency_resolutions_hz = np.array(
            [record["lowest_frequency_resolution_hz"] for record in spectral_sensitivity_analysis],
            dtype=float,
        )
        log_rms_misfits = np.array([record["log_rms_misfit"] for record in spectral_sensitivity_analysis], dtype=float)

        best_index = int(np.argmin(log_rms_misfits))

        fig = plt.figure(figsize=self._get_single_panel_size(), dpi=220)
        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            segment_strides,
            lowest_frequency_resolutions_hz,
            c=log_rms_misfits,
            cmap="viridis",
            s=10,
            edgecolors="none",
            linewidths=0,
        )

        ax.scatter(
            segment_strides[best_index],
            lowest_frequency_resolutions_hz[best_index],
            color="red",
            s=80,
            marker="X",
            edgecolors="none",
            linewidths=0,
            label="Best fit",
            zorder=3,
        )

        # ax.set_title(title)
        ax.set_xlabel("Segment Stride [samples]")
        ax.set_ylabel(r"$1 / T_{\mathrm{segment}}$ [Hz]")
        self._style_axes(ax)
        ax.legend(loc="best", frameon=True)

        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label(colorbar_label, fontsize=14)
        colorbar.ax.tick_params(labelsize=12)

        fig.tight_layout()
        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_accelerometer_noise_asd(
        self,
        frequencies: np.ndarray,
        sensitive_axis_asd: np.ndarray,
        normal_axis_asd: np.ndarray,
        file_name: str,
    ) -> None:
        """Plot the sensitive-axis and normal-axis accelerometer ASD models."""

        frequencies = np.asarray(frequencies, dtype=float)
        sensitive_axis_asd = np.asarray(sensitive_axis_asd, dtype=float)
        normal_axis_asd = np.asarray(normal_axis_asd, dtype=float)

        if frequencies.ndim != 1:
            raise ValueError("frequencies must be a 1D array.")
        if sensitive_axis_asd.shape != frequencies.shape or normal_axis_asd.shape != frequencies.shape:
            raise ValueError("Accelerometer ASD arrays must match the frequency span.")

        colors = self._get_colors_palette(2)
        fig = plt.figure(figsize=self._get_single_panel_size())
        ax = fig.add_subplot(111)

        ax.loglog(
            frequencies,
            sensitive_axis_asd,
            color=colors[0],
            linewidth=2.5,
            label=r"$X_{\mathrm{ACC}},\ Z_{\mathrm{ACC}}$",
        )
        ax.loglog(
            frequencies,
            normal_axis_asd,
            color=colors[1],
            linewidth=2.5,
            label=r"$Y_{\mathrm{ACC}}$",
        )

        # ax.set_title("Accelerometer Random Noise ASD")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel(r"ASD [m s$^{-2}$ Hz$^{-1/2}$]")
        ax.set_xlim(1e-5, 1e-1)
        self._style_axes(ax)
        ax.legend(loc="best", frameon=True)

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_accelerometer_noise_time_series(
        self,
        component_noise_time_series: list,
        file_name: str,
        suptitle: str,
        window_length_samples: int | None = 4320,
    ) -> None:
        """Plot the accelerometer noise time history for a single satellite."""

        if len(component_noise_time_series) != 3:
            raise ValueError("Exactly three component noise time series are required.")

        component_labels = ("x", "y", "z")

        fig, axes = plt.subplots(3, 1, figsize=self._get_article_three_row_size(), dpi=300, sharex=True)
        axes = np.atleast_1d(axes)

        for component_idx, (ax, component_label, component_series) in enumerate(
            zip(axes, component_labels, component_noise_time_series)
        ):

            if window_length_samples is not None:
                n_samples = int(window_length_samples)
                series = component_series[:n_samples]
            else:
                series = component_series
            
            ax.plot(
                series.sample_times,
                series,
                color=RED,
                linewidth=1.6,
                label=component_labels[component_idx],
            )
            # ax.set_title(f"{component_label} Component")
            ax.set_ylabel(r"Noise [m s$^{-2}$]")
            ax.legend(loc="upper right", frameon=True, fontsize = 13)
            self._style_axes(ax)
            if component_idx == len(component_noise_time_series) - 1:
                ax.set_xlabel("Time [s]")

        # fig.suptitle(suptitle, y=0.98)
        fig.tight_layout()
        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_kbr_system_and_oscillator_asd(
        self,
        frequencies: np.ndarray,
        asd_values: np.ndarray,
        file_name: str
    ) -> None:
        """Plot KBR system and oscillator ASD."""

        fig = plt.figure(figsize=self._get_single_panel_size())

        plt.loglog(frequencies,  asd_values, color=RED, linewidth=2.5)

        # plt.title("KBR System and Oscillator Noise ASD")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"ASD [m Hz$^{-1/2}$]")
        plt.xlim(1e-5, 1e-1)
        self._style_axes(plt.gca())

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_kbr_system_and_oscillator_noise_time_series(
        self,
        noise_time_series,
        file_name: str
    ) -> None:
        """Plot time series of KBR noise and oscillator noise."""

        fig = plt.figure(figsize=self._get_single_panel_size())

        plt.plot(noise_time_series.sample_times, noise_time_series, color=RED, linewidth=1.8)

        # plt.title("KBR System and Oscillator Noise Time Series")
        plt.xlabel("Time [s]")
        plt.ylabel("Error [m]")
        self._style_axes(plt.gca())

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_apc_pointing_jitter_coupling_time_series_demeaned(
        self,
        apc_pointing_jitter_coupling_noise: np.ndarray,
        time_seconds: np.ndarray,
        satellite_label: str,
        file_name: str,
    ) -> None:
        """
        Plot APC offset pointing jitter coupling noise (demeaned) for a single satellite.

        Parameters
        ----------
        apc_pointing_jitter_coupling_noise : dict[str, np.ndarray]
            Dict containing APC coupling noise arrays in meters, keyed by satellite name.
        time_seconds : np.ndarray
            Time array in seconds (shape (N,)).
        satellite_label : str
            Satellite key.
        file_name : str
            Output file name (saved under self.output_path).
        """


        value = np.asarray(apc_pointing_jitter_coupling_noise, dtype=float).reshape(-1)
        time = np.asarray(time_seconds, dtype=float).reshape(-1)

        if time.shape[0] != value.shape[0]:
            raise ValueError(f"Time and APC noise length mismatch: len(time)={time.shape[0]} vs len(value)={value.shape[0]}")
        value_demeaned = value - np.mean(value)

        fig = plt.figure(figsize=self._get_single_panel_size())
        plt.plot(time, value_demeaned, color=RED, linewidth=1.8)

        # plt.title(f"APC Pointing Jitter Coupling Noise (Demeaned) — {satellite_label}")
        plt.xlabel("Time [s]")
        plt.ylabel("Error (demeaned) [m]")
        self._style_axes(plt.gca())

        fig.savefig(self.output_path / file_name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    def plot_residual_apc_coupling_jitter_noise_time_series(
        self,
        residual_apc_coupling_jitter_noise: np.ndarray,
        time_seconds: np.ndarray,
        satellite: str,
        file_name: str,
    ) -> None:
        """Plot the residual APC coupling jitter noise time series."""
        
        residual_noise = np.asarray(residual_apc_coupling_jitter_noise, dtype=float).reshape(-1)
        time_seconds = np.asarray(time_seconds, dtype=float)

        if residual_noise.shape[0] != time_seconds.shape[0]:
            raise ValueError(
                f"Time and residual APC noise length mismatch: "
                f"{time_seconds.shape[0]} vs {residual_noise.shape[0]}."
            )

        fig, ax = plt.subplots(figsize=self._get_single_panel_size())
        ax.plot(time_seconds, residual_noise, color=RED, linewidth=1.8)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error [m]")
        satellite_label = "GRACE C" if satellite == "GRACE C" else "GRACE D"
        # ax.set_title(f"Residual APC Coupling Jitter Noise - {satellite_label}")
        self._style_axes(ax)

        fig.tight_layout()
        fig.savefig(self.output_path / file_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_apc_offset_vector_error_components_time_series(
        self,
        error_vector_history: np.ndarray,
        time_seconds: np.ndarray,
        satellite_label: str,
        file_name: str,
    ) -> None:
        """Plot the APC offset-vector error components in the satellite frame."""

        error_vector_history = np.asarray(error_vector_history, dtype=float)
        time_seconds = np.asarray(time_seconds, dtype=float).reshape(-1)

        if error_vector_history.ndim != 2 or error_vector_history.shape[1] != 3:
            raise ValueError("error_vector_history must be a 2D array with shape (N, 3).")
        if error_vector_history.shape[0] != time_seconds.shape[0]:
            raise ValueError(
                "Time and APC offset-vector error histories must contain the same number of samples."
            )

        time_hours = (time_seconds - time_seconds[0]) / 3600.0
        component_labels = ("x", "y", "z")

        fig, axes = plt.subplots(3, 1, figsize=self._get_article_three_row_size(), dpi=300, sharex=True)

        for component_idx, (axis, component_label) in enumerate(
            zip(axes, component_labels)
        ):
            axis.plot(
                time_hours,
                error_vector_history[:, component_idx],
                color=RED,
                linewidth=1.8,
            )
            axis.set_ylabel(rf"$\Delta p_{{{component_label},SF}}$ [m]")
            # axis.set_title(f"{component_label.upper()} component")
            self._style_axes(axis)

        axes[-1].set_xlabel("Propagation time [hours]")
        # fig.suptitle(f"APC Offset-Vector Error Components - {satellite_label}", y=0.95)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(self.output_path / file_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_srp_acceleration_time_series(
        self,
        dependent_variables_array: np.ndarray,
        norm_title: str = "SRP acceleration norm time evolution — GRACE-FO",
        components_title_prefix: str = "SRP acceleration component",
    ) -> None:
        """
        Plot SRP acceleration norm time evolution for GRACE C and GRACE D,
        and additionally create 3 separate figures for the SRP acceleration components
        (x, y, z) for both satellites.
        """

        if dependent_variables_array.ndim != 2 or dependent_variables_array.shape[1] < 35:
            raise ValueError(
                "dependent_variables_array must be 2D with at least 35 columns: "
            )

        num_samples = min(dependent_variables_array.shape[0], 4320)
        
        time_seconds = dependent_variables_array[:num_samples, 0]
        time_hours = (time_seconds - time_seconds[0]) / 3600.0

        srp_acc_norm_grace_c = dependent_variables_array[:num_samples, 1]
        srp_acc_norm_grace_d = dependent_variables_array[:num_samples, 2]

        norm_file_name = "grace_fo_srp_acc_norm_time_evolution.png"

        colors = self._get_colors_palette(2)
        fig = plt.figure(figsize=self._get_single_panel_size(), dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(time_hours, srp_acc_norm_grace_c, color=colors[0], linewidth=2.5, label="GRACE C")
        ax.plot(time_hours, srp_acc_norm_grace_d, color=colors[1], linewidth=2.5, linestyle="--", label="GRACE D")
        # ax.set_title(norm_title)
        ax.set_xlabel("Propagation time [hours]")
        ax.set_ylabel(r"$\|a_{\mathrm{SRP}}\|$ [m/s$^2$]")
        self._style_axes(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.output_path / norm_file_name, bbox_inches="tight")
        plt.close(fig)

        srp_acc_j2000_grace_c = dependent_variables_array[:num_samples, 3:6]  # [ax, ay, az]
        srp_acc_j2000_grace_d = dependent_variables_array[:num_samples, 6:9]  # [ax, ay, az]
        rotation_j2000_to_sf_grace_c = dependent_variables_array[:num_samples, 17:26]
        rotation_j2000_to_sf_grace_d = dependent_variables_array[:num_samples, 26:35]
        srp_acc_sf_grace_c = transform_vector_history_inertial_to_satellite_frame(
            srp_acc_j2000_grace_c, rotation_j2000_to_sf_grace_c
        )
        srp_acc_sf_grace_d = transform_vector_history_inertial_to_satellite_frame(
            srp_acc_j2000_grace_d, rotation_j2000_to_sf_grace_d
        )

        comp_labels = ["x", "y", "z"]
        for j, comp in enumerate(comp_labels):
            fig = plt.figure(figsize=self._get_single_panel_size(), dpi=300)
            ax = fig.add_subplot(111)

            ax.plot(time_hours, srp_acc_sf_grace_c[:, j], color=colors[0], linewidth=2.5, label="GRACE C")
            ax.plot(time_hours, srp_acc_sf_grace_d[:, j], color=colors[1], linewidth=2.5, linestyle="--", label="GRACE D")

            # ax.set_title(f"{components_title_prefix} {comp}_SF time evolution — GRACE-FO")
            ax.set_xlabel("Propagation time [hours]")
            ax.set_ylabel(rf"$a_{{\mathrm{{SRP}},{comp}_{{SF}}}}$ [m/s$^2$]")
            self._style_axes(ax)
            ax.legend(loc="best")

            fig.tight_layout()

            component_file_name = f"grace_fo_srp_acc_sf_comp_{comp}_time_evolution.png"

            fig.savefig(self.output_path / component_file_name, bbox_inches="tight")
            plt.close(fig)

    def plot_aerodynamic_acceleration_time_series(
        self,
        dependent_variables_array: np.ndarray,
        norm_title: str = "Aerodynamic acceleration norm time evolution — GRACE-FO",
        components_title_prefix: str = "Aerodynamic acceleration component",
    ) -> None:
        """
        Plot aerodynamic acceleration norm time evolution for GRACE C and GRACE D,
        and additionally create 3 separate figures for the aerodynamic acceleration components
        (x, y, z) for both satellites.
        """

        if dependent_variables_array.ndim != 2 or dependent_variables_array.shape[1] < 35:
            raise ValueError(
                "dependent_variables_array must be 2D with at least 35 columns: "
            )

        num_samples = min(dependent_variables_array.shape[0], 4320)

        time_seconds = dependent_variables_array[:num_samples, 0]
        time_hours = (time_seconds - time_seconds[0]) / 3600.0

        aero_acc_norm_grace_c = dependent_variables_array[:num_samples, 9]
        aero_acc_norm_grace_d = dependent_variables_array[:num_samples, 10]

        norm_file_name = "grace_fo_aero_acc_norm_time_evolution.png"

        colors = self._get_colors_palette(2)
        fig = plt.figure(figsize=self._get_single_panel_size(), dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(time_hours, aero_acc_norm_grace_c, color=colors[0], linewidth=2.5, label="GRACE-C")
        ax.plot(time_hours, aero_acc_norm_grace_d, color=colors[1], linewidth=2.5, linestyle="--", label="GRACE-D")
        # ax.set_title(norm_title)
        ax.set_xlabel("Propagation time [hours]")
        ax.set_ylabel(r"$\|a_{\mathrm{aero}}\|$ [m/s$^2$]")
        self._style_axes(ax)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(self.output_path / norm_file_name, bbox_inches="tight")
        plt.close(fig)

        aero_acc_j2000_grace_c = dependent_variables_array[:num_samples, 11:14]  # [ax, ay, az]
        aero_acc_j2000_grace_d = dependent_variables_array[:num_samples, 14:17]  # [ax, ay, az]
        rotation_j2000_to_sf_grace_c = dependent_variables_array[:num_samples, 17:26]
        rotation_j2000_to_sf_grace_d = dependent_variables_array[:num_samples, 26:35]
        aero_acc_sf_grace_c = transform_vector_history_inertial_to_satellite_frame(
            aero_acc_j2000_grace_c, rotation_j2000_to_sf_grace_c
        )
        aero_acc_sf_grace_d = transform_vector_history_inertial_to_satellite_frame(
            aero_acc_j2000_grace_d, rotation_j2000_to_sf_grace_d
        )

        comp_labels = ["x", "y", "z"]
        for j, comp in enumerate(comp_labels):
            fig = plt.figure(figsize=self._get_single_panel_size(), dpi=300)
            ax = fig.add_subplot(111)

            ax.plot(time_hours, aero_acc_sf_grace_c[:, j], color=colors[0], linewidth=2.5, label="GRACE-C")
            ax.plot(time_hours, aero_acc_sf_grace_d[:, j], color=colors[1], linewidth=2.5, linestyle="--", label="GRACE-D")

            # ax.set_title(f"{components_title_prefix} {comp}_SF time evolution — GRACE-FO")
            ax.set_xlabel("Propagation time [hours]")
            ax.set_ylabel(rf"$a_{{\mathrm{{aero}},{comp}_{{SF}}}}$ [m/s$^2$]")

            self._style_axes(ax)
            ax.legend(loc="best")

            fig.tight_layout()

            component_file_name = f"grace_fo_aero_acc_sf_comp_{comp}_time_evolution.png"
            fig.savefig(self.output_path / component_file_name, bbox_inches="tight")
            plt.close(fig)

    def plot_keplerian_elements_difference_time_evolution(
            self,
            time_data: np.ndarray,
            dependent_variables_array: np.ndarray,
    )-> None:
        """Plot GRACE-FO Keplerian element differences time history."""

        time_data = np.asarray(time_data, dtype=float)
        dependent_variables_array = np.asarray(dependent_variables_array, dtype=float)

        if time_data.ndim != 1:
            raise ValueError("time_data must be a 1D array.")
        if dependent_variables_array.ndim != 2:
            raise ValueError("dependent_variables_array must be a 2D array.")

        elapsed_time_days = (time_data - time_data[0]) / 86400.0

        grace_c_keplerian_history = dependent_variables_array[:, 35:41]
        grace_d_keplerian_history = dependent_variables_array[:, 41:47]

        keplerian_difference = grace_c_keplerian_history - grace_d_keplerian_history

        # Wrap angular element differences to [-pi, pi] before converting to degrees.
        keplerian_difference[:, 2:] = np.arctan2(
            np.sin(keplerian_difference[:, 2:]),
            np.cos(keplerian_difference[:, 2:]),
        )

        subplot_metadata = [
            ("Semi-major axis", keplerian_difference[:, 0], r"$\Delta a$ [m]"),
            ("Eccentricity", keplerian_difference[:, 1], r"$\Delta e$ [-]"),
            ("Inclination", np.degrees(keplerian_difference[:, 2]), r"$\Delta i$ [deg]"),
            ("Argument of periapsis", np.degrees(keplerian_difference[:, 3]), r"$\Delta \omega$ [deg]"),
            ("RAAN", np.degrees(keplerian_difference[:, 4]), r"$\Delta \Omega$ [deg]"),
            ("True anomaly", np.degrees(keplerian_difference[:, 5]), r"$\Delta \nu$ [deg]"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(12.0, 8.0), dpi=300, sharex=True)
        line_color = RED

        for axis, (title, values, ylabel) in zip(axes.flat, subplot_metadata):
            axis.plot(elapsed_time_days, values, color=line_color, linewidth=1.8)
            axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
            # axis.set_title(title)
            axis.set_ylabel(ylabel)
            self._style_axes(axis)

            max_abs_value = np.nanmax(np.abs(values))
            if max_abs_value > 0.0 and (max_abs_value < 1e-2 or max_abs_value >= 1e3):
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((0, 0))
                axis.yaxis.set_major_formatter(formatter)

        for axis in axes[-1, :]:
            axis.set_xlabel("Propagation time [days]")

        # fig.suptitle("Keplerian element differences between GRACE C and GRACE D")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(
            self.output_path / "grace_fo_keplerian_elements_difference_time_evolution.png",
            bbox_inches="tight",
        )
        plt.close(fig)
