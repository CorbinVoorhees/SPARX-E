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
CALIBRATION_FILE = ANALYSIS_DIR / "magnm_calibration.txt"
FIELD_STRENGTH_UT = 50.0
GYRO_STATIONARY_THRESHOLD = 0.01
TESLA_PER_MICROTESLA = 1e-6


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


def main():
    calibration = np.loadtxt(CALIBRATION_FILE)
    bias_t = calibration[0]
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
    magnm = sensors[sensors["type"] == "magnm"].copy().sort_values("time")
    gyro = sensors[sensors["type"] == "gyro"].copy().sort_values("time")
    gyro = gyro.rename(columns={"v0": "gx", "v1": "gy", "v2": "gz"})
    magnm = pd.merge_asof(
        magnm,
        gyro[["time", "gx", "gy", "gz"]],
        on="time",
        direction="nearest",
    )
    magnm = magnm[
        np.linalg.norm(magnm[["gx", "gy", "gz"]], axis=1)
        < GYRO_STATIONARY_THRESHOLD
    ]

    attitude = attitude.sort_values("time")
    attitude = attitude[(attitude[["roll", "pitch", "yaw"]] != 0).any(axis=1)]
    magnm = magnm[
        magnm["time"].between(
            attitude["time"].iloc[0], attitude["time"].iloc[-1]
        )
    ]
    magnm = pd.merge_asof(magnm, attitude, on="time", direction="nearest")

    raw_t = magnm[["v0", "v1", "v2"]].to_numpy()
    calibrated_t = (transform @ (raw_t - bias_t).T).T
    raw_ut = raw_t / TESLA_PER_MICROTESLA
    calibrated_ut = calibrated_t / TESLA_PER_MICROTESLA
    rpy = np.deg2rad(magnm[["roll", "pitch", "yaw"]].to_numpy())
    rotation = rotation_matrices(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    inertial_ut = np.einsum("nij,nj->ni", rotation, calibrated_ut)
    elapsed = (magnm["time"] - magnm["time"].iloc[0]).to_numpy()

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
                (raw_ut, "Before calibration: body frame"),
                (calibrated_ut, "After calibration: body frame"),
                (inertial_ut, "After calibration: inertial frame"),
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
            ax.add_patch(
                plt.Circle(
                    (0, 0), FIELD_STRENGTH_UT, fill=False, color="black"
                )
            )
            ax.set_title(f"{state}: {title}")
            ax.set_xlabel(f"{labels[horizontal]} magnetic field (µT)")
            ax.set_ylabel(f"{labels[vertical]} magnetic field (µT)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)

    fig.colorbar(points, ax=axes, label="Elapsed time (s)", shrink=0.85)
    fig.suptitle(f"Magnetometer calibration ({len(magnm):,} samples)")
    plt.show()


if __name__ == "__main__":
    main()
