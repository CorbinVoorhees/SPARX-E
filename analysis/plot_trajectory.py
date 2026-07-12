#!/usr/bin/env python3
"""Plot EKF position, velocity, and attitude with an X11-friendly 2D default."""

import argparse
from pathlib import Path

import matplotlib


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trans", default="sparxe_ekf_test1__1_1.csv")
    parser.add_argument("--attitude", default="sparxe_mekf_test1__1_1.csv")
    parser.add_argument("--3d", dest="three_d", action="store_true",
                        help="show a 3D trajectory instead of the 2D dashboard")
    parser.add_argument("--arrows", type=int, default=30,
                        help="maximum orientation/velocity arrows (default: 30)")
    parser.add_argument("--velocity-scale", type=float, default=0.5,
                        help="seconds represented by each velocity arrow")
    parser.add_argument("--save", metavar="FILE",
                        help="save without opening an X11 window")
    return parser.parse_args()


args = parse_args()
if args.save:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_numeric(path, required):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing {path}; run nav_filters first")
    data = pd.read_csv(path)
    missing = set(required) - set(data.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    for column in set(required) | {"time"}:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=required).sort_values("time").drop_duplicates("time")


trans = read_numeric(args.trans, ["time", "px", "py", "pz", "vx", "vy", "vz"])
att = read_numeric(args.attitude, ["time", "roll", "pitch", "yaw"])
if trans.empty or att.empty:
    raise SystemExit("the input logs contain no usable samples")

# Attach the nearest attitude sample to every translational state sample.
median_dt = att.time.diff().median()
tolerance = max(0.25, 5.0 * median_dt) if np.isfinite(median_dt) else 0.25
data = pd.merge_asof(trans, att[["time", "roll", "pitch", "yaw"]],
                     on="time", direction="nearest", tolerance=tolerance)
data = data.dropna(subset=["roll", "pitch", "yaw"]).copy()
if data.empty:
    raise SystemExit("position and attitude logs do not overlap in time")
data["t"] = data.time - data.time.iloc[0]

arrow_count = max(1, min(args.arrows, len(data)))
arrow_rows = data.iloc[np.unique(np.linspace(0, len(data) - 1, arrow_count).astype(int))]
yaw = np.deg2rad(arrow_rows.yaw.to_numpy())
path_span = max(np.ptp(data.px), np.ptp(data.py), 0.1)
heading_length = 0.06 * path_span

if args.three_d:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(data.px, data.py, data.pz, color="tab:blue", label="trajectory")
    ax.scatter(data.px.iloc[0], data.py.iloc[0], data.pz.iloc[0],
               color="green", s=55, label="start")
    ax.scatter(data.px.iloc[-1], data.py.iloc[-1], data.pz.iloc[-1],
               color="red", s=55, label="end")
    ax.quiver(arrow_rows.px, arrow_rows.py, arrow_rows.pz,
              np.cos(yaw), np.sin(yaw), np.zeros(len(arrow_rows)),
              length=heading_length, normalize=True, color="tab:orange",
              label="heading")
    ax.quiver(arrow_rows.px, arrow_rows.py, arrow_rows.pz,
              arrow_rows.vx, arrow_rows.vy, arrow_rows.vz,
              length=args.velocity_scale, normalize=False, color="tab:purple",
              label="velocity")
    ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
           title="Rover trajectory, heading, and velocity")
    ax.legend()
    ax.grid(True, alpha=0.35)
else:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    path_ax, pos_ax, vel_ax, att_ax = axes.flat

    path_ax.plot(data.px, data.py, color="tab:blue", label="trajectory")
    path_ax.scatter(data.px.iloc[0], data.py.iloc[0], color="green", s=45, label="start")
    path_ax.scatter(data.px.iloc[-1], data.py.iloc[-1], color="red", s=45, label="end")
    path_ax.quiver(arrow_rows.px, arrow_rows.py, np.cos(yaw), np.sin(yaw),
                   angles="xy", scale_units="xy", scale=1.0 / heading_length,
                   width=0.004, color="tab:orange", label="heading")
    path_ax.quiver(arrow_rows.px, arrow_rows.py, arrow_rows.vx, arrow_rows.vy,
                   angles="xy", scale_units="xy", scale=1.0 / args.velocity_scale,
                   width=0.003, color="tab:purple", alpha=0.7, label="velocity")
    path_ax.set(xlabel="x (m)", ylabel="y (m)", title="Top-down trajectory")
    path_ax.axis("equal")
    path_ax.legend(fontsize=8)

    for column in ("px", "py", "pz"):
        pos_ax.plot(data.t, data[column], label=column)
    pos_ax.set(title="Position", ylabel="m")
    pos_ax.legend(ncol=3)

    for column in ("vx", "vy", "vz"):
        vel_ax.plot(data.t, data[column], label=column)
    vel_ax.set(title="Velocity", xlabel="time (s)", ylabel="m/s")
    vel_ax.legend(ncol=3)

    for column in ("roll", "pitch", "yaw"):
        att_ax.plot(data.t, data[column], label=column)
    att_ax.set(title="Relative attitude", xlabel="time (s)", ylabel="degrees")
    att_ax.legend(ncol=3)

    for ax in axes.flat:
        ax.grid(True, linestyle="--", alpha=0.35)

fig.tight_layout()
if args.save:
    fig.savefig(args.save, dpi=160, bbox_inches="tight")
    print(f"saved {args.save}")
else:
    plt.show()
