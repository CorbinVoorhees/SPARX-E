#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial.transform import Rotation


LOG = Path(__file__).resolve().parents[1] / "chuudkf_logs.csv"
G = np.array([0.0, 0.0, 9.80665])


def vec(text):
    return np.fromstring(text, sep=" ")


def read_state():
    q = Rotation.identity()
    p = np.zeros(3)
    v = np.zeros(3)
    a = np.zeros(3)

    gyro_time = None
    accel_time = None

    with open(LOG, newline="") as file:
        for row in csv.DictReader(file):
            if row["filter"] != "CHUUDKF":
                continue

            sensor = row["sensor"]

            if sensor not in ("gyro_unbiased", "accel"):
                continue

            t = float(row["time"])
            z = vec(row["measurement"])
            x = vec(row["x_post"])

            if sensor == "gyro_unbiased":
                if gyro_time is not None:
                    q = q * Rotation.from_rotvec(
                        z * (t - gyro_time)
                    )

                gyro_time = t

            else:
                if accel_time is not None:
                    dt = t - accel_time
                    accel_bias = x[10:13]

                    a = G - q.apply(z - accel_bias)

                    p += v * dt + 0.5 * a * dt**2
                    v += a * dt

                accel_time = t

    return q, p, v, a


fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(projection="3d")
ax.set_proj_type("ortho")


def draw(_):
    q, p, v, a = read_state()

    R = q.as_matrix()
    rpy = q.as_euler("xyz", degrees=True)

    ax.clear()

    radius = max(1.0, np.max(np.abs(p)) * 1.3)
    arrow = radius * 0.25

    ax.scatter(0, 0, 0, marker="+", s=80)
    ax.scatter(*p, s=80)
    ax.plot([0, p[0]], [0, p[1]], [0, p[2]], ":")

    for i, color in enumerate(("r", "g", "b")):
        ax.quiver(
            *p,
            *R[:, i],
            length=arrow,
            normalize=True,
            color=color,
        )

    if np.linalg.norm(v):
        ax.quiver(
            *p,
            *(v / np.linalg.norm(v)),
            length=arrow,
            normalize=True,
            color="m",
        )

    if np.linalg.norm(a):
        ax.quiver(
            *p,
            *(a / np.linalg.norm(a)),
            length=arrow,
            normalize=True,
            color="k",
        )

    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(30, -45)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(
        f"integrated RPY: {rpy.round(2)} deg\n"
        f"accel: {a.round(3)} m/s²\n"
        f"velocity: {v.round(3)} m/s\n"
        f"position: {p.round(3)} m"
    )


animation = FuncAnimation(
    fig,
    draw,
    interval=200,
    cache_frame_data=False,
)

plt.show()