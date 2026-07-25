from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

G = 9.80655
ROOT = Path(__file__).parents[1]
CALIBRATION = Path(__file__).with_name("accel_calibration.txt")


def fit_ellipsoid(samples):
    x, y, z = samples.T
    design = np.column_stack(
        (x**2, y**2, z**2, 2 * x * y, 2 * y * z, 2 * x * z, x, y, z)
    )
    p = np.linalg.lstsq(design, np.full(len(samples), G**2), rcond=None)[0]

    shape = np.array(
        [[p[0], p[3], p[5]], [p[3], p[1], p[4]], [p[5], p[4], p[2]]]
    )
    bias = -0.5 * np.linalg.solve(shape, p[6:9])
    shape *= G**2 / (G**2 + bias @ shape @ bias)
    stretch = np.linalg.cholesky(shape).T
    return bias, stretch


def main():
    sensors = pd.read_csv(ROOT / "sparxe_sensors_raw.csv", low_memory=False)
    numeric_columns = ["time", "v0", "v1", "v2"]
    sensors[numeric_columns] = sensors[numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    accel = sensors[sensors.type == "accel"].set_index("time")[["v0", "v1", "v2"]]
    gyro = sensors[sensors.type == "gyro"].set_index("time")[["v0", "v1", "v2"]]

    data = accel.join(gyro, lsuffix="_a", rsuffix="_g").dropna()
    data = data[np.linalg.norm(data[["v0_g", "v1_g", "v2_g"]], axis=1) < 0.01]
    raw = data[["v0_a", "v1_a", "v2_a"]].to_numpy()

    bias, stretch = fit_ellipsoid(raw)
    calibrated = (raw - bias) @ stretch.T
    np.savetxt(CALIBRATION, np.vstack((bias, stretch)))

    print("bias:", bias)
    print("stretch:\n", stretch)
    print("radius RMS:", np.sqrt(np.mean((np.linalg.norm(calibrated, axis=1) - G) ** 2)))

    elapsed = data.index.to_numpy() - data.index[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (i, j, title) in zip(axes, ((0, 1, "XY"), (1, 2, "YZ"), (0, 2, "XZ"))):
        points = ax.scatter(calibrated[:, i], calibrated[:, j], c=elapsed, s=3)
        ax.add_patch(plt.Circle((0, 0), G, fill=False, color="black"))
        ax.set(title=title, aspect="equal", xlabel="m/s²", ylabel="m/s²")

    fig.colorbar(points, ax=axes, label="time (s)")
    plt.show()


if __name__ == "__main__":
    main()
