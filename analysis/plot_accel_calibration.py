import os
from pathlib import Path

import matplotlib

if "MPLBACKEND" not in os.environ:
    if os.environ.get("WAYLAND_DISPLAY"):
        os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
        matplotlib.use("QtAgg")
    else:
        matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
SENSORS_CSV = ANALYSIS_DIR.parent / "sparxe_sensors_raw.csv"
MEKF_CSV = ANALYSIS_DIR.parent / "sparxe_mekf_test1__1_1.csv"
CALIBRATION_FILE = ANALYSIS_DIR / "accel_calibration.txt"
GRAVITY = 9.80655


def rotation_matrices(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation = np.empty((len(roll), 3, 3))
    rotation[:, 0, 0] = cy * cp
    rotation[:, 0, 1] = cy * sp * sr - sy * cr
    rotation[:, 0, 2] = cy * sp * cr + sy * sr
    rotation[:, 1, 0] = sy * cp
    rotation[:, 1, 1] = sy * sp * sr + cy * cr
    rotation[:, 1, 2] = sy * sp * cr - cy * sr
    rotation[:, 2, 0] = -sp
    rotation[:, 2, 1] = cp * sr
    rotation[:, 2, 2] = cp * cr
    return rotation


def integrate(values, time):
    """Cumulatively integrate vector samples using the trapezoidal rule."""
    integrated = np.zeros_like(values)
    dt = np.diff(time)
    integrated[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * dt[:, np.newaxis], axis=0
    )
    return integrated


def main():
    calibration = np.loadtxt(CALIBRATION_FILE)
    bias = calibration[0]
    transform = calibration[1:4]

    sensors = pd.read_csv(
        SENSORS_CSV,
        usecols=["time", "type", "v0", "v1", "v2"],
        low_memory=False,
    )
    attitude = pd.read_csv(MEKF_CSV, usecols=["time", "roll", "pitch", "yaw"])

    numeric_columns = ["time", "v0", "v1", "v2"]
    sensors[numeric_columns] = sensors[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    accel = sensors[sensors["type"] == "accel"].copy().sort_values("time")
    attitude = attitude.sort_values("time")
    attitude = attitude[(attitude[["roll", "pitch", "yaw"]] != 0).any(axis=1)]
    accel = accel[
        accel["time"].between(attitude["time"].iloc[0], attitude["time"].iloc[-1])
    ]
    accel = pd.merge_asof(accel, attitude, on="time", direction="nearest")

    raw = accel[["v0", "v1", "v2"]].to_numpy()
    calibrated = (transform @ (raw - bias).T).T
    rpy = np.deg2rad(accel[["roll", "pitch", "yaw"]].to_numpy())
    rotation = rotation_matrices(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    inertial = np.einsum("nij,nj->ni", rotation, calibrated)
    elapsed = (accel["time"] - accel["time"].iloc[0]).to_numpy()

    linear_acceleration = inertial.copy()
    linear_acceleration[:, 2] -= GRAVITY
    velocity = integrate(linear_acceleration, elapsed)
    position = integrate(velocity, elapsed)

    planes = (
        (0, 1, "XY"),
        (0, 2, "XZ"),
        (1, 2, "YZ"),
    )
    labels = ("X", "Y", "Z")
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)

    points = None
    for column, (horizontal, vertical, title) in enumerate(planes):
        for row, (values, state) in enumerate(
            (
                (raw, "Before calibration: body frame"),
                (calibrated, "After calibration: body frame"),
                (inertial, "After calibration: inertial frame"),
            )
        ):
            ax = axes[row, column]
            points = ax.scatter(
                values[:, horizontal],
                values[:, vertical],
                c=elapsed,
                cmap="viridis",
                s=8,
                alpha=0.75,
                linewidths=0,
            )
            ax.set_title(f"{state}: {title}")
            ax.set_xlabel(f"{labels[horizontal]} acceleration (m/s²)")
            ax.set_ylabel(f"{labels[vertical]} acceleration (m/s²)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)

    fig.colorbar(points, ax=axes, label="Elapsed time (s)", shrink=0.85)
    fig.suptitle(f"Accelerometer calibration ({len(accel):,} samples)")

    integrated_fig, integrated_axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, constrained_layout=True
    )
    for axis, label in enumerate(labels):
        integrated_axes[0].plot(elapsed, velocity[:, axis], label=label)
        integrated_axes[1].plot(elapsed, position[:, axis], label=label)

    integrated_axes[0].set_ylabel("Velocity (m/s)")
    integrated_axes[1].set_ylabel("Position (m)")
    integrated_axes[1].set_xlabel("Elapsed time (s)")
    for ax in integrated_axes:
        ax.grid(True, alpha=0.25)
        ax.legend()
    integrated_fig.suptitle("Integrated inertial acceleration (gravity removed)")
    plt.show()


if __name__ == "__main__":
    main()
