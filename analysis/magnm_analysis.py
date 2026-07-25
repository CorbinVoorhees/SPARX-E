import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIELD_STRENGTH_UT = 50.0
TESLA_PER_MICROTESLA = 1e-6
ROOT = Path(__file__).parents[1]
LOG_FILE = ROOT / "chuudkf_logs.csv"
CALIBRATION = ROOT / "magnm_calibration.txt"


def fit_ellipsoid(samples_ut):
    """Fit hard/soft-iron terms that map samples onto a 50 uT sphere."""
    x, y, z = samples_ut.T
    design = np.column_stack(
        (x**2, y**2, z**2, 2 * x * y, 2 * y * z, 2 * x * z, x, y, z)
    )
    p = np.linalg.lstsq(
        design, np.full(len(samples_ut), FIELD_STRENGTH_UT**2), rcond=None
    )[0]

    shape = np.array(
        [[p[0], p[3], p[5]], [p[3], p[1], p[4]], [p[5], p[4], p[2]]]
    )
    bias_ut = -0.5 * np.linalg.solve(shape, p[6:9])
    shape *= FIELD_STRENGTH_UT**2 / (
        FIELD_STRENGTH_UT**2 + bias_ut @ shape @ bias_ut
    )
    transform = np.linalg.cholesky(shape).T
    return bias_ut, transform


def chuudkf_magnetometer_samples(path):
    rows = []
    with path.open(newline="") as log:
        for row in csv.DictReader(log):
            if row.get("sensor") != "magnm":
                continue

            try:
                timestamp = float(row["time"])
                measurement = np.fromstring(row["measurement"], sep=" ")
            except (KeyError, TypeError, ValueError):
                continue

            if measurement.size == 3:
                rows.append((timestamp, *measurement))

    return pd.DataFrame(rows, columns=("time", "v0_m", "v1_m", "v2_m"))


def plot_planes(raw_ut, calibrated_ut, elapsed):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    planes = ((0, 1, "XY"), (1, 2, "YZ"), (0, 2, "XZ"))
    points = None
    for row, (samples, label) in enumerate(
        ((raw_ut, "Raw"), (calibrated_ut, "Calibrated"))
    ):
        for ax, (i, j, title) in zip(axes[row], planes):
            points = ax.scatter(samples[:, i], samples[:, j], c=elapsed, s=3)
            ax.add_patch(
                plt.Circle((0, 0), FIELD_STRENGTH_UT, fill=False, color="black")
            )
            ax.set(
                title=f"{label} {title}",
                aspect="equal",
                xlabel="µT",
                ylabel="µT",
            )

    fig.colorbar(points, ax=axes, label="time (s)")
    plt.show()


def main():
    data = chuudkf_magnetometer_samples(LOG_FILE)
    if len(data) < 9:
        raise ValueError(f"Need at least 9 magnetometer samples; got {len(data)}")

    raw_ut = data[["v0_m", "v1_m", "v2_m"]].to_numpy() / TESLA_PER_MICROTESLA
    bias_ut, transform = fit_ellipsoid(raw_ut)
    calibrated_ut = (raw_ut - bias_ut) @ transform.T

    bias_t = bias_ut * TESLA_PER_MICROTESLA
    np.savetxt(CALIBRATION, np.vstack((bias_t, transform)))

    magnitude_error = np.linalg.norm(calibrated_ut, axis=1) - FIELD_STRENGTH_UT
    print("bias (T):", bias_t)
    print("transform:\n", transform)
    print("magnitude RMS error (µT):", np.sqrt(np.mean(magnitude_error**2)))
    print("sample count:", len(raw_ut))

    elapsed = data.time.to_numpy() - data.time.iloc[0]
    plot_planes(raw_ut, calibrated_ut, elapsed)


if __name__ == "__main__":
    main()
