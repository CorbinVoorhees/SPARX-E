
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "time" in df.columns and not df.empty:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).copy()
        df["time"] -= df["time"].iloc[0]
    return df


def save_plot(df, columns, title, ylabel, output):
    columns = [c for c in columns if c in df.columns]
    if df.empty or not columns:
        return

    plt.figure(figsize=(11, 6))
    for column in columns:
        plt.plot(df["time"], df[column], label=column)

    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Wrote {output}")


def plot_sensors(df: pd.DataFrame, out_dir: Path):
    if df.empty or "type" not in df.columns:
        return

    sensor_columns = {
        "gyro": ["v0", "v1", "v2"],
        "accel": ["v0", "v1", "v2"],
        "magnm": ["v0", "v1", "v2"],
        "uwb": ["v0"],
        "control": ["v0", "v1", "v2", "v3"],
    }

    for sensor, columns in sensor_columns.items():
        data = df[df["type"] == sensor].copy()
        if data.empty:
            continue

        data["time"] -= data["time"].iloc[0]
        save_plot(
            data,
            columns,
            f"Raw sensor: {sensor}",
            "Measurement",
            out_dir / f"sensor_{sensor}.png",
        )


def plot_innovations(df: pd.DataFrame, out_dir: Path):
    groups = {
        "uwb_innovation": ["uwb_r"],
        "accel_innovation": ["acc_rx", "acc_ry", "acc_rz"],
        "attitude_accel_innovation": ["macc_rx", "macc_ry", "macc_rz"],
        "mag_innovation": ["mmag_rx", "mmag_ry", "mmag_rz"],
        "control_innovation": ["mctl_r"],
    }

    for name, columns in groups.items():
        save_plot(
            df,
            columns,
            name.replace("_", " ").title(),
            "Innovation",
            out_dir / f"{name}.png",
        )


def plot_state(
    attitude: pd.DataFrame,
    translation: pd.DataFrame,
    out_dir: Path,
):
    save_plot(
        attitude,
        ["roll", "pitch", "yaw"],
        "Attitude state",
        "Angle [deg]",
        out_dir / "state_attitude.png",
    )

    save_plot(
        attitude,
        ["bgx", "bgy", "bgz"],
        "Gyroscope bias state",
        "Bias [rad/s]",
        out_dir / "state_gyro_bias.png",
    )

    save_plot(
        translation,
        ["px", "py", "pz"],
        "Position state",
        "Position [m]",
        out_dir / "state_position.png",
    )

    save_plot(
        translation,
        ["vx", "vy", "vz"],
        "Velocity state",
        "Velocity [m/s]",
        out_dir / "state_velocity.png",
    )

    save_plot(
        translation,
        ["mur", "mul"],
        "Wheel efficiency state",
        "Efficiency",
        out_dir / "state_wheel_efficiency.png",
    )

    save_plot(
        translation,
        ["nis"],
        "Latest NIS",
        "NIS",
        out_dir / "state_nis.png",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("."),
        help="Folder containing the CSV files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Folder for generated plots",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sensors = load(args.log_dir / "sparxe_sensors_raw.csv")
    gains = load(args.log_dir / "sparxe_gains.csv")
    attitude = load(args.log_dir / "sparxe_mekf_test1__1_1.csv")
    translation = load(args.log_dir / "sparxe_ekf_test1__1_1.csv")

    plot_sensors(sensors, args.out_dir)
    plot_innovations(gains, args.out_dir)
    plot_state(attitude, translation, args.out_dir)


if __name__ == "__main__":
    main()