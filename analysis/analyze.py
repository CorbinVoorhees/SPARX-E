import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any


class Analyzer:
    def __init__(self, logs: dict, save_dir: str | None = None):
        self.logs = logs
        self.save_dir = save_dir

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        self.fs = 16
        self.lw = 2.8
        self.alpha = 0.22

        plt.rcParams.update({
            "font.size": self.fs,
            "axes.titlesize": self.fs,
            "axes.labelsize": self.fs,
            "xtick.labelsize": self.fs - 2,
            "ytick.labelsize": self.fs - 2,
            "legend.fontsize": self.fs - 3,
            "lines.linewidth": self.lw,
            "figure.figsize": (16, 10),
        })

    # -------------------------------------------------------------------------
    # Generic extraction helpers
    # -------------------------------------------------------------------------

    def _blocks(self, name: str):
        return self.logs.get(name, []) or []

    def _get(self, d: dict, path: str, default: Any = np.nan) -> Any:
        cur: Any = d
        for p in path.split("."):
            if not isinstance(cur, dict) or p not in cur or cur[p] is None:
                return default
            cur = cur[p]
        return cur

    def _scalar_series(self, blocks, path: str):
        vals = []
        for b in blocks:
            v = self._get(b, path, np.nan)
            if isinstance(v, (int, float, np.integer, np.floating)):
                vals.append(float(v))
            else:
                vals.append(np.nan)
        return np.asarray(vals, dtype=float)

    def _time(self, blocks):
        t = self._scalar_series(blocks, "time.t")
        if len(t) == 0 or np.all(np.isnan(t)):
            return t
        if np.nanmin(t) > 1e5:
            t = t - t[0]
        return t

    def _save_or_show(self, fig, name: str):
        fig.tight_layout()
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, name)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def _format_ax(self, ax):
        ax.grid(True, alpha=0.35)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")

    def _plot_series(self, ax, t, y, label="value", color=None, linewidth=None):
        mask = np.isfinite(t) & np.isfinite(y)
        if np.any(mask):
            ax.plot(
                t[mask],
                y[mask],
                label=label,
                color=color,
                linewidth=linewidth if linewidth is not None else self.lw,
                zorder=5,
            )
        return mask

    def _plot_with_cov(self, ax, t, y, cov=None, label="value", color=None):
        mask = self._plot_series(ax, t, y, label=label, color=color)

        if cov is not None and len(cov) == len(y):
            cov_mask = mask & np.isfinite(cov)
            if np.any(cov_mask):
                sigma = np.sqrt(np.maximum(cov[cov_mask], 0.0))
                ax.fill_between(
                    t[cov_mask],
                    y[cov_mask] - 2.0 * sigma,
                    y[cov_mask] + 2.0 * sigma,
                    alpha=self.alpha,
                    color=color,
                    label=r"$\pm 2\sigma$",
                    zorder=1,
                )

        self._format_ax(ax)

    def _plot_mean_std(self, ax, t, mean, std, label="Mean", color=None):
        mean_mask = np.isfinite(t) & np.isfinite(mean)

        if std is not None and len(std) == len(mean):
            std_mask = mean_mask & np.isfinite(std)
            if np.any(std_mask):
                ax.fill_between(
                    t[std_mask],
                    mean[std_mask] - 2.0 * std[std_mask],
                    mean[std_mask] + 2.0 * std[std_mask],
                    alpha=self.alpha,
                    color=color,
                    label=r"$\pm 2\sigma$",
                    zorder=1,
                )

        if np.any(mean_mask):
            ax.plot(
                t[mean_mask],
                mean[mean_mask],
                label=label,
                color=color,
                linewidth=3.5,
                zorder=5,
            )

        self._format_ax(ax)

    # -------------------------------------------------------------------------
    # Sensor diagnostic plots
    # -------------------------------------------------------------------------

    def plot_accel_diagnostics(self):
        nav = self._blocks("nav")
        t = self._time(nav)

        fig, axs = plt.subplots(3, 4, sharex=True, figsize=(22, 12))

        items = [
            ("ax", "bax", "Accel X"),
            ("ay", "bay", "Accel Y"),
            ("az", "baz", "Accel Z"),
        ]

        for i, (axis, bias_key, title) in enumerate(items):
            raw = self._scalar_series(nav, f"sensor_raw.accWorld.{axis}")
            bias = self._scalar_series(nav, f"state.accBias.{bias_key}")
            p_bias = self._scalar_series(nav, f"covariance.P_loc_bias.{bias_key}")
            corrected = raw - bias

            prof_mean = self._scalar_series(nav, f"profile.accMean.{axis}")
            prof_std = self._scalar_series(nav, f"profile.accStd.{axis}")

            self._plot_series(axs[i, 0], t, raw, label="Raw", color="tab:gray")
            axs[i, 0].set_ylabel(f"{title}\n[m/s²]")
            if i == 0:
                axs[i, 0].set_title("Raw")
            self._format_ax(axs[i, 0])

            self._plot_with_cov(
                axs[i, 1],
                t,
                bias,
                p_bias,
                label=bias_key,
                color="tab:orange",
            )
            if i == 0:
                axs[i, 1].set_title("Bias Estimate")

            self._plot_series(
                axs[i, 2],
                t,
                corrected,
                label="Corrected",
                color="tab:blue",
            )
            if i == 0:
                axs[i, 2].set_title("Corrected (Raw - Bias)")
            self._format_ax(axs[i, 2])

            self._plot_mean_std(
                axs[i, 3],
                t,
                prof_mean,
                prof_std,
                label="Mean",
                color="tab:green",
            )
            if i == 0:
                axs[i, 3].set_title("Profiled Value")

        for j in range(4):
            axs[-1, j].set_xlabel("Time [s]")

        self._save_or_show(fig, "accel_diagnostics.png")

    def plot_gyro_diagnostics(self):
        nav = self._blocks("nav")
        t = self._time(nav)

        fig, axs = plt.subplots(3, 4, sharex=True, figsize=(22, 12))

        items = [
            ("wx", "bgx", "Gyro X"),
            ("wy", "bgy", "Gyro Y"),
            ("wz", "bgz", "Gyro Z"),
        ]

        for i, (axis, bias_key, title) in enumerate(items):
            raw = self._scalar_series(nav, f"sensor_raw.gyro.{axis}")
            bias = self._scalar_series(nav, f"state.gyroBias.{bias_key}")
            p_bias = self._scalar_series(nav, f"covariance.P_rot_diag.{bias_key}")
            corrected = raw - bias

            prof_mean = self._scalar_series(nav, f"profile.gyroMean.{axis}")
            prof_std = self._scalar_series(nav, f"profile.gyroStd.{axis}")

            self._plot_series(axs[i, 0], t, raw, label="Raw", color="tab:gray")
            axs[i, 0].set_ylabel(f"{title}\n[rad/s]")
            if i == 0:
                axs[i, 0].set_title("Raw")
            self._format_ax(axs[i, 0])

            self._plot_with_cov(
                axs[i, 1],
                t,
                bias,
                p_bias,
                label=bias_key,
                color="tab:orange",
            )
            if i == 0:
                axs[i, 1].set_title("Bias Estimate")

            self._plot_series(
                axs[i, 2],
                t,
                corrected,
                label="Corrected",
                color="tab:blue",
            )
            if i == 0:
                axs[i, 2].set_title("Corrected (Raw - Bias)")
            self._format_ax(axs[i, 2])

            self._plot_mean_std(
                axs[i, 3],
                t,
                prof_mean,
                prof_std,
                label="Mean",
                color="tab:green",
            )
            if i == 0:
                axs[i, 3].set_title("Profiled Value")

        for j in range(4):
            axs[-1, j].set_xlabel("Time [s]")

        self._save_or_show(fig, "gyro_diagnostics.png")

    def plot_range_diagnostics(self):
        nav = self._blocks("nav")
        t = self._time(nav)

        fig, axs = plt.subplots(2, 4, sharex=True, figsize=(22, 8))

        metrics = [
            (
                "sensor_raw.range.raw",
                "state.rngBias.br",
                "covariance.P_loc_bias.br",
                "profile.rangeMean.r",
                "profile.rangeStd.r",
                "Range",
                "[m]",
            ),
            (
                "sensor_raw.rangeDot.raw",
                "state.rngBias.brr",
                "covariance.P_loc_bias.brr",
                "profile.rangeMean.rr",
                "profile.rangeStd.rr",
                "Range-Rate",
                "[m/s]",
            ),
        ]

        for i, (raw_key, bias_key, cov_key, mean_key, std_key, title, unit) in enumerate(metrics):
            raw = self._scalar_series(nav, raw_key)
            bias = self._scalar_series(nav, bias_key)
            p_bias = self._scalar_series(nav, cov_key)
            corrected = raw - bias

            prof_mean = self._scalar_series(nav, mean_key)
            prof_std = self._scalar_series(nav, std_key)

            self._plot_series(axs[i, 0], t, raw, label="Raw", color="tab:gray")
            axs[i, 0].set_ylabel(f"{title}\n{unit}")
            if i == 0:
                axs[i, 0].set_title("Raw")
            self._format_ax(axs[i, 0])

            self._plot_with_cov(
                axs[i, 1],
                t,
                bias,
                p_bias,
                label=bias_key.split(".")[-1],
                color="tab:orange",
            )
            if i == 0:
                axs[i, 1].set_title("Bias Estimate")

            self._plot_series(
                axs[i, 2],
                t,
                corrected,
                label="Corrected",
                color="tab:blue",
            )
            if i == 0:
                axs[i, 2].set_title("Corrected (Raw - Bias)")
            self._format_ax(axs[i, 2])

            self._plot_mean_std(
                axs[i, 3],
                t,
                prof_mean,
                prof_std,
                label="Mean",
                color="tab:green",
            )
            if i == 0:
                axs[i, 3].set_title("Profiled Value")

        for j in range(4):
            axs[-1, j].set_xlabel("Time [s]")

        self._save_or_show(fig, "range_diagnostics.png")

    # -------------------------------------------------------------------------
    # State estimation plots
    # -------------------------------------------------------------------------

    def plot_state_estimation(self):
        nav = self._blocks("nav")
        t = self._time(nav)

        fig, axs = plt.subplots(2, 3, sharex=True, figsize=(22, 10))

        items = [
            ("state.pos.x", "covariance.P_loc_pos.x", "x", "Position X", "[m]", 0, 0),
            ("state.pos.y", "covariance.P_loc_pos.y", "y", "Position Y", "[m]", 0, 1),
            ("state.pos.z", "covariance.P_loc_pos.z", "z", "Position Z", "[m]", 0, 2),
            ("state.vel.vx", "covariance.P_loc_vel.vx", "vx", "Velocity X", "[m/s]", 1, 0),
            ("state.vel.vy", "covariance.P_loc_vel.vy", "vy", "Velocity Y", "[m/s]", 1, 1),
            ("state.vel.vz", "covariance.P_loc_vel.vz", "vz", "Velocity Z", "[m/s]", 1, 2),
        ]

        for state_key, cov_key, label, title, unit, r, c in items:
            y = self._scalar_series(nav, state_key)
            cov = self._scalar_series(nav, cov_key)

            self._plot_with_cov(
                axs[r, c],
                t,
                y,
                cov,
                label=label,
                color="tab:blue",
            )

            axs[r, c].set_title(title)
            axs[r, c].set_ylabel(unit)

        for c in range(3):
            axs[-1, c].set_xlabel("Time [s]")

        self._save_or_show(fig, "state_estimation.png")

    def plot_attitude_estimation(self):
        nav = self._blocks("nav")
        t = self._time(nav)

        fig, axs = plt.subplots(2, 3, sharex=True, figsize=(22, 10))

        items = [
            ("state.rpy.roll", None, "roll", "Roll", "[rad]", 0, 0),
            ("state.rpy.pitch", "covariance.P_loc_att.pitch", "pitch", "Pitch", "[rad]", 0, 1),
            ("state.rpy.yaw", "covariance.P_loc_att.yaw", "yaw", "Yaw", "[rad]", 0, 2),
            ("state.gyroBias.bgx", "covariance.P_rot_diag.bgx", "bgx", "Gyro Bias X", "[rad/s]", 1, 0),
            ("state.gyroBias.bgy", "covariance.P_rot_diag.bgy", "bgy", "Gyro Bias Y", "[rad/s]", 1, 1),
            ("state.gyroBias.bgz", "covariance.P_rot_diag.bgz", "bgz", "Gyro Bias Z", "[rad/s]", 1, 2),
        ]

        for state_key, cov_key, label, title, unit, r, c in items:
            y = self._scalar_series(nav, state_key)
            cov = self._scalar_series(nav, cov_key) if cov_key is not None else None

            self._plot_with_cov(
                axs[r, c],
                t,
                y,
                cov,
                label=label,
                color="tab:purple",
            )

            axs[r, c].set_title(title)
            axs[r, c].set_ylabel(unit)

        for c in range(3):
            axs[-1, c].set_xlabel("Time [s]")

        self._save_or_show(fig, "attitude_estimation.png")

    def plot_bias_estimation(self):
        nav = self._blocks("nav")
        t = self._time(nav)

        fig, axs = plt.subplots(2, 3, sharex=True, figsize=(22, 10))

        items = [
            ("state.accBias.bax", "covariance.P_loc_bias.bax", "bax", "Accel Bias X", "[m/s²]", 0, 0),
            ("state.accBias.bay", "covariance.P_loc_bias.bay", "bay", "Accel Bias Y", "[m/s²]", 0, 1),
            ("state.accBias.baz", "covariance.P_loc_bias.baz", "baz", "Accel Bias Z", "[m/s²]", 0, 2),
            ("state.rngBias.br", "covariance.P_loc_bias.br", "br", "Range Bias", "[m]", 1, 0),
            ("state.rngBias.brr", "covariance.P_loc_bias.brr", "brr", "Range-Rate Bias", "[m/s]", 1, 1),
        ]

        for state_key, cov_key, label, title, unit, r, c in items:
            y = self._scalar_series(nav, state_key)
            cov = self._scalar_series(nav, cov_key)

            self._plot_with_cov(
                axs[r, c],
                t,
                y,
                cov,
                label=label,
                color="tab:orange",
            )

            axs[r, c].set_title(title)
            axs[r, c].set_ylabel(unit)

        axs[1, 2].axis("off")

        for c in range(3):
            axs[-1, c].set_xlabel("Time [s]")

        self._save_or_show(fig, "bias_estimation.png")

    # -------------------------------------------------------------------------
    # Main entry
    # -------------------------------------------------------------------------

    def plot_all(self):
        self.plot_accel_diagnostics()
        self.plot_gyro_diagnostics()
        self.plot_range_diagnostics()
        self.plot_state_estimation()
        self.plot_attitude_estimation()
        self.plot_bias_estimation()