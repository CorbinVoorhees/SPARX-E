#!/usr/bin/env python3
"""Plot MEKF log CSV.
Columns written by log_func:
  time,nis,roll,pitch,yaw,gx,gy,gz,gunbx,gunby,gunbz,bgx,bgy,bgz,gmx,gmy,gmz
Usage: python3 plot_mekf.py mekf_log.csv [--nis-dim 3]
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def require_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing CSV columns: {missing}")


def plot_one(ax, t, df, key, title, ylabel):
    ax.plot(t, df[key])
    ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)


def main(path, nis_dim):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "time", "nis", "roll", "pitch", "yaw",
        "gx", "gy", "gz", "gunbx", "gunby", "gunbz",
        "bgx", "bgy", "bgz", "gmx", "gmy", "gmz",
    ]
    require_cols(df, required)

    # 'time' is absolute steady_clock seconds -> elapsed from first row
    t = df["time"] - df["time"].iloc[0]

    # Figure 1: gyro, rows = x/y/z, cols = raw / corrected / bias / mean
    fig, ax = plt.subplots(3, 4, figsize=(18, 10), sharex=True)
    for i, c in enumerate("xyz"):
        plot_one(ax[i, 0], t, df, f"g{c}",    f"gyro raw {c}",       "rad/s")
        plot_one(ax[i, 1], t, df, f"gunb{c}", f"gyro corrected {c}", "rad/s")
        plot_one(ax[i, 2], t, df, f"bg{c}",   f"gyro bias {c}",      "rad/s")
        plot_one(ax[i, 3], t, df, f"gm{c}",   f"gyro mean {c}",      "rad/s")
    for a in ax[-1, :]:
        a.set_xlabel("t [s]")
    fig.tight_layout()
    out = path.rsplit(".", 1)[0] + "_gyro.png"
    fig.savefig(out, dpi=130); print(f"saved {out}")

    # Figure 2: attitude (radians, as logged)
    fig3, ax3 = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for a, key in zip(ax3, ["roll", "pitch", "yaw"]):
        plot_one(a, t, df, key, key, "rad")
    ax3[-1].set_xlabel("t [s]")
    fig3.tight_layout()
    rpy_out = path.rsplit(".", 1)[0] + "_rpy.png"
    fig3.savefig(rpy_out, dpi=130); print(f"saved {rpy_out}")

    # Figure 3: NIS
    fig2, ax2 = plt.subplots(figsize=(15, 4.5))
    ax2.plot(t, df["nis"])
    ax2.axhline(nis_dim, linestyle="--", linewidth=1.2, label=f"expected mean = {nis_dim:g}")
    ax2.set_title("NIS"); ax2.set_ylabel("νᵀS⁻¹ν"); ax2.set_xlabel("t [s]")
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8)
    fig2.tight_layout()
    nis_out = path.rsplit(".", 1)[0] + "_nis.png"
    fig2.savefig(nis_out, dpi=130); print(f"saved {nis_out}")

    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--nis-dim", type=float, default=3.0,
                   help="measurement dim m; expected NIS mean is m")
    a = p.parse_args()
    main(a.csv, a.nis_dim)