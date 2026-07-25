import argparse
import csv
import time
from math import asin, atan2, degrees, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

LOG_FILE = Path(__file__).resolve().parents[1] / "chuudkf_logs.csv"
FIELD_STRENGTH_UT = 50.0
TESLA_TO_MICROTESLA = 1e6
DEFAULT_SMOOTH_SIGMA = 25.0
CORRECTION_BIN_SECONDS = 1.0
AUTOCORR_LAGS = (1, *range(5, 51, 5))
DEFAULT_AUTOCORR_WINDOW = 250
DEFAULT_STATIC_MAX_POINTS = 20000
STATE_SIZE = 12
STATE_GROUPS = (
    ("Orientation", ("roll", "pitch", "yaw"), "degrees"),
    ("Position", ("x", "y", "z"), "m"),
    ("Velocity", ("x", "y", "z"), "m/s"),
    ("Wheel scales", ("mu_c", "mu_d"), "scale"),
)


def vector(text):
    return [float(value) for value in text.split()]


def nis_values(row):
    dof = int(row.get("nis_dof", "0") or 0)
    if dof <= 0:
        return None
    return (
        float(row["nis"]),
        float(row["nis_lower_critical"]),
        float(row["nis_upper_critical"]),
    )


def quaternion_to_rpy(q):
    w, x, y, z = q
    norm = sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = asin(max(-1, min(1, 2 * (w * y - z * x))))
    yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return degrees(roll), degrees(pitch), degrees(yaw)


def state_vectors(state):
    return (
        quaternion_to_rpy(state[:4]),
        state[4:7],
        state[7:10],
        state[10:12],
    )


def parse_log_row(row):
    """Parse one complete CHUUDKF CSV row, returning None when malformed."""
    try:
        state = vector(row["x_post"])
        measurement = vector(row["measurement"])
        nis = nis_values(row)
        if len(state) < STATE_SIZE or not measurement:
            return None
        return (
            float(row["time"]),
            row["sensor"],
            state,
            measurement,
            nis,
            row["applied"] == "1",
        )
    except (KeyError, TypeError, ValueError):
        return None


def iter_log_rows(log_path):
    """Yield parsed rows without retaining the complete log in memory."""
    with log_path.open(newline="") as log_file:
        for row in csv.DictReader(log_file):
            parsed = parse_log_row(row)
            if parsed is not None:
                yield parsed


def scan_log_time_bounds(log_path):
    """Return first timestamp, last timestamp, and approximate complete-row count."""
    first_timestamp = None
    last_timestamp = None
    complete_rows = 0

    with log_path.open(newline="") as log_file:
        for row in csv.DictReader(log_file):
            try:
                if len(row["x_post"].split()) < STATE_SIZE:
                    continue
                if not row["measurement"].strip():
                    continue
                timestamp = float(row["time"])
            except (KeyError, TypeError, ValueError):
                continue

            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp
            complete_rows += 1

    if first_timestamp is None or last_timestamp is None:
        raise RuntimeError(f"No complete filter rows found in {log_path}")

    return first_timestamp, last_timestamp, complete_rows


def sparse_bucket(timestamp, start_timestamp, end_timestamp, bucket_count):
    if bucket_count <= 1 or end_timestamp <= start_timestamp:
        return 0
    fraction = (timestamp - start_timestamp) / (end_timestamp - start_timestamp)
    return min(bucket_count - 1, max(0, int(fraction * bucket_count)))


def load_sparse_static_data(log_path, max_points, last=None, time_range=None):
    """Load a bounded, uniformly time-binned representation of a large log.

    State, each measurement stream, and each NIS stream retain at most
    ``max_points`` samples. The complete CSV is scanned sequentially, so memory
    use is independent of the log duration.
    """
    t0, t_last, complete_rows = scan_log_time_bounds(log_path)
    duration = t_last - t0

    if last is not None:
        start_elapsed = max(0.0, duration - last)
        end_elapsed = duration
    elif time_range is not None:
        start_elapsed, end_elapsed = time_range
        start_elapsed = max(0.0, start_elapsed)
        end_elapsed = min(duration, end_elapsed)
    else:
        start_elapsed = 0.0
        end_elapsed = duration

    if end_elapsed < start_elapsed:
        raise RuntimeError(
            f"Requested time range {start_elapsed:g}..{end_elapsed:g} s is empty"
        )

    start_timestamp = t0 + start_elapsed
    end_timestamp = t0 + end_elapsed

    state_buckets = {}
    sensor_buckets = {}
    nis_buckets = {}
    dt_buckets = {}

    selected_rows = 0
    previous_timestamp = None

    for parsed in iter_log_rows(log_path):
        timestamp, sensor, state, measurement, nis, applied = parsed
        if timestamp < start_timestamp or timestamp > end_timestamp:
            continue

        selected_rows += 1
        bucket = sparse_bucket(
            timestamp, start_timestamp, end_timestamp, max_points
        )

        # Keep the latest state in each time bucket.
        state_buckets[bucket] = (timestamp, state)

        # Keep each sensor sparse independently, so a high-rate stream cannot
        # erase a low-rate stream that shares the same time bucket.
        sensor_buckets[(sensor, bucket)] = (timestamp, measurement)

        if nis is not None:
            nis_buckets[(sensor, bucket)] = (timestamp, *nis, applied)

        # Preserve the largest real timestamp gap in each bucket. Computing dt
        # from sparse state samples would otherwise manufacture large fake dt.
        if previous_timestamp is not None and timestamp > previous_timestamp + 1e-9:
            dt_ms = 1000.0 * (timestamp - previous_timestamp)
            current = dt_buckets.get(bucket)
            if current is None or dt_ms > current[1]:
                dt_buckets[bucket] = (timestamp, dt_ms)
        if previous_timestamp is None or timestamp > previous_timestamp:
            previous_timestamp = timestamp

    if not state_buckets:
        raise RuntimeError(
            f"No complete filter rows found in requested range for {log_path}"
        )

    data = LiveData(None)
    data.t0 = t0

    for _, (timestamp, state) in sorted(state_buckets.items()):
        data.times.append(timestamp - t0)
        data.states.append(state)

    for (sensor, _), (timestamp, measurement) in sorted(
        sensor_buckets.items(), key=lambda item: (item[0][0], item[1][0])
    ):
        data.sensors.setdefault(sensor, []).append((timestamp - t0, measurement))

    for (sensor, _), sample in sorted(
        nis_buckets.items(), key=lambda item: (item[0][0], item[1][0])
    ):
        timestamp, nis, lower, upper, applied = sample
        elapsed = timestamp - t0
        data.nis.setdefault(sensor, []).append((elapsed, nis, lower, upper))
        data.corrections.append((elapsed, applied))

    for _, (timestamp, dt_ms) in sorted(dt_buckets.items()):
        data.dt_times.append(timestamp - t0)
        data.dt_ms.append(dt_ms)

    stats = {
        "complete_rows": complete_rows,
        "selected_rows": selected_rows,
        "retained_states": len(data.states),
        "retained_measurements": sum(len(samples) for samples in data.sensors.values()),
    }
    return data, stats


class LogTail:
    def __init__(self, path):
        self.path = path
        self.file = None
        self.fieldnames = None
        self.previous_size = 0
        self.file_identity = None

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def _recent_start(self, row_count, header_end):
        """Return the byte offset of the last ``row_count`` rows."""
        with self.path.open("rb") as raw_file:
            raw_file.seek(0, 2)
            cursor = raw_file.tell()
            if cursor <= header_end:
                return header_end

            raw_file.seek(cursor - 1)
            if raw_file.read(1) == bytes((10,)):
                cursor -= 1

            rows_remaining = row_count
            chunk_size = 1024 * 1024
            while cursor > header_end:
                chunk_start = max(header_end, cursor - chunk_size)
                raw_file.seek(chunk_start)
                chunk = raw_file.read(cursor - chunk_start)
                search_end = len(chunk)
                while rows_remaining > 0:
                    newline = chunk.rfind(bytes((10,)), 0, search_end)
                    if newline < 0:
                        break
                    rows_remaining -= 1
                    if rows_remaining == 0:
                        return chunk_start + newline + 1
                    search_end = newline
                cursor = chunk_start
        return header_end

    def reopen(self, recent_rows=None):
        self.close()
        self.file = self.path.open(newline="")
        header = self.file.readline()
        if not header:
            raise RuntimeError(f"Missing CSV header in {self.path}")
        self.fieldnames = next(csv.reader([header]))
        header_end = self.file.tell()
        stat = self.path.stat()
        if recent_rows is not None:
            self.file.seek(self._recent_start(recent_rows, header_end))
        self.previous_size = stat.st_size
        self.file_identity = (stat.st_dev, stat.st_ino)

    def read_new(self, recent_rows=None):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return [], False

        identity = (stat.st_dev, stat.st_ino)
        reset = (
            self.file is None
            or identity != self.file_identity
            or stat.st_size < self.previous_size
        )
        if reset:
            try:
                self.reopen(recent_rows)
            except FileNotFoundError:
                return [], False

        rows = []
        while True:
            position = self.file.tell()
            line = self.file.readline()
            if not line:
                break
            if not line.endswith(chr(10)):
                self.file.seek(position)
                break

            values = next(csv.reader([line]))
            if len(values) != len(self.fieldnames):
                continue
            row = dict(zip(self.fieldnames, values))
            parsed = parse_log_row(row)
            if parsed is not None:
                rows.append(parsed)

        self.previous_size = stat.st_size
        return rows, reset

    def read_one(self):
        if self.file is None:
            self.reopen()
        while True:
            line = self.file.readline()
            if not line:
                return None
            values = next(csv.reader([line]))
            if len(values) != len(self.fieldnames):
                continue
            row = dict(zip(self.fieldnames, values))
            parsed = parse_log_row(row)
            if parsed is not None:
                return parsed


class LiveData:
    def __init__(self, max_points):
        self.max_points = max_points
        self.reset()

    def reset(self):
        self.t0 = None
        self.times = []
        self.states = []
        self.sensors = {}
        self.nis = {}
        self.corrections = []
        self.dt_times = []
        self.dt_ms = []
        self._last_unique_timestamp = None

    def extend(self, rows):
        for timestamp, sensor, state, measurement, nis, applied in rows:
            if self.t0 is None:
                self.t0 = timestamp
            elapsed = timestamp - self.t0
            if (
                self._last_unique_timestamp is not None
                and timestamp > self._last_unique_timestamp + 1e-9
            ):
                self.dt_times.append(elapsed)
                self.dt_ms.append(
                    1000.0 * (timestamp - self._last_unique_timestamp)
                )
            if (
                self._last_unique_timestamp is None
                or timestamp > self._last_unique_timestamp
            ):
                self._last_unique_timestamp = timestamp
            self.times.append(elapsed)
            self.states.append(state)
            self.sensors.setdefault(sensor, []).append((elapsed, measurement))
            if nis is not None:
                self.nis.setdefault(sensor, []).append((elapsed, *nis))
                self.corrections.append((elapsed, applied))
        if self.max_points is not None and len(self.times) > self.max_points:
            del self.times[: -self.max_points]
            del self.states[: -self.max_points]
        if self.max_points is not None and len(self.dt_times) > self.max_points:
            del self.dt_times[: -self.max_points]
            del self.dt_ms[: -self.max_points]
        if (
            self.max_points is not None
            and len(self.corrections) > self.max_points
        ):
            del self.corrections[: -self.max_points]
        for samples in (*self.sensors.values(), *self.nis.values()):
            if self.max_points is not None and len(samples) > self.max_points:
                del samples[: -self.max_points]


def padded_limits(values):
    low = min(values)
    high = max(values)
    span = high - low
    padding = 0.05 * span if span > 0 else max(abs(low) * 0.05, 1e-6)
    return low - padding, high + padding


def gaussian_trend(values, sigma, unwrap_degrees=False):
    """Return a Gaussian-smoothed copy, preserving continuous angle wrapping."""
    samples = np.asarray(values, dtype=float)
    if samples.size < 2 or sigma <= 0:
        return samples
    if unwrap_degrees:
        samples = np.rad2deg(np.unwrap(np.deg2rad(samples)))
    return gaussian_filter1d(samples, sigma=sigma, mode="nearest")


def rolling_autocorrelation(values, lag, window, unwrap_degrees=False):
    """Return rolling Pearson correlation against a lagged copy."""
    samples = np.asarray(values, dtype=float)
    if unwrap_degrees:
        samples = np.rad2deg(np.unwrap(np.deg2rad(samples)))
    result = np.full(samples.size, np.nan)
    if samples.size < lag + window:
        return result

    current = samples[lag:]
    delayed = samples[:-lag]

    def rolling_sum(array):
        cumulative = np.concatenate(([0.0], np.cumsum(array)))
        return cumulative[window:] - cumulative[:-window]

    sum_current = rolling_sum(current)
    sum_delayed = rolling_sum(delayed)
    sum_squares_current = rolling_sum(current * current)
    sum_squares_delayed = rolling_sum(delayed * delayed)
    sum_products = rolling_sum(current * delayed)
    covariance = sum_products - sum_current * sum_delayed / window
    variance_current = sum_squares_current - sum_current**2 / window
    variance_delayed = sum_squares_delayed - sum_delayed**2 / window
    denominator = np.sqrt(
        np.maximum(variance_current * variance_delayed, 0.0)
    )
    correlations = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > np.finfo(float).eps,
    )
    result[lag + window - 1 :] = correlations
    return result


class LivePlots:
    def __init__(
        self,
        max_points,
        smooth_sigma=DEFAULT_SMOOTH_SIGMA,
        autocorr_window=DEFAULT_AUTOCORR_WINDOW,
        show_autocorrelation=False,
    ):
        self.max_points = max_points
        self.smooth_sigma = smooth_sigma
        self.autocorr_window = autocorr_window
        self.projection_figure = plt.figure(figsize=(15, 14))
        self.projection_signature = None
        self.nis_figure = plt.figure(figsize=(12, 8))
        self.nis_signature = None
        self.autocorr_figure = None
        if show_autocorrelation:
            self.autocorr_figure = plt.figure(figsize=(15, 14))
            self._build_autocorrelation()

    @property
    def figures(self):
        figures = [self.projection_figure, self.nis_figure]
        if self.autocorr_figure is not None:
            figures.append(self.autocorr_figure)
        return tuple(figures)

    def _build_autocorrelation(self):
        self.autocorr_axes = self.autocorr_figure.subplots(
            len(STATE_GROUPS), 3, sharex=True, sharey=True, squeeze=False
        )
        self.autocorr_lines = []
        for row, (title, labels, _) in enumerate(STATE_GROUPS):
            row_lines = []
            for column, axis in enumerate(self.autocorr_axes[row]):
                if column >= len(labels):
                    axis.set_visible(False)
                    continue

                label = labels[column]
                lines = []
                for lag in AUTOCORR_LAGS:
                    (line,) = axis.plot([], [], linewidth=0.9, label=f"lag {lag}")
                    lines.append(line)
                confidence = 1.96 / sqrt(self.autocorr_window)
                axis.axhline(
                    -confidence, linestyle="--", linewidth=0.8, alpha=0.5
                )
                axis.axhline(
                    confidence, linestyle="--", linewidth=0.8, alpha=0.5
                )
                axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
                axis.set_title(f"{title}: {label}")
                axis.set_ylabel("correlation")
                axis.set_ylim(-1.05, 1.05)
                axis.grid(True, alpha=0.25)
                row_lines.append(lines)
            self.autocorr_lines.append(row_lines)
        for axis in self.autocorr_axes[-1]:
            if axis.get_visible():
                axis.set_xlabel("elapsed time (s)")
        self.autocorr_figure.suptitle(
            f"Rolling filter-state autocorrelation — window={self.autocorr_window} samples"
        )
        self.autocorr_figure.legend(
            handles=self.autocorr_lines[0][0],
            loc="upper center",
            ncol=len(AUTOCORR_LAGS),
            fontsize=7,
            bbox_to_anchor=(0.5, 0.975),
        )
        self.autocorr_figure.tight_layout(rect=(0, 0, 1, 0.94))

    def _build_nis(self, sensors):
        signature = tuple(sorted(sensors))
        if not signature or signature == self.nis_signature:
            return
        self.nis_signature = signature
        self.nis_figure.clear()
        axes = self.nis_figure.subplots(
            len(signature), 1, sharex=True, squeeze=False
        )
        self.nis_lines = {}
        for row, sensor in enumerate(signature):
            axis = axes[row, 0]
            (nis_line,) = axis.plot([], [], linewidth=0.9, label="NIS")
            (lower_line,) = axis.plot(
                [], [], "--", linewidth=0.9, label="lower critical"
            )
            (upper_line,) = axis.plot(
                [], [], "--", linewidth=0.9, label="upper critical"
            )
            self.nis_lines[sensor] = (nis_line, lower_line, upper_line)
            axis.set_title(f"{sensor} NIS")
            axis.set_ylabel("NIS")
            axis.grid(True, alpha=0.25)
            axis.legend(loc="upper right", fontsize=8)
        axes[-1, 0].set_xlabel("elapsed time (s)")
        self.nis_figure.suptitle(
            "Per-sensor NIS with two-sided 95% chi-square bounds"
        )
        self.nis_figure.tight_layout(rect=(0, 0, 1, 0.95))

    def _build_state(self):
        self.state_figure.clear()
        self.state_axes = self.state_figure.subplots(
            len(STATE_GROUPS), 3, sharex=True, squeeze=False
        )
        self.state_lines = []
        for row, (title, labels, unit) in enumerate(STATE_GROUPS):
            row_lines = []
            for column, axis in enumerate(self.state_axes[row]):
                if column >= len(labels):
                    axis.set_visible(False)
                    continue

                label = labels[column]
                (line,) = axis.plot([], [])
                row_lines.append(line)
                axis.set_title(f"{title} {label}")
                axis.set_ylabel(unit)
                axis.grid(True)
            self.state_lines.append(row_lines)
        for axis in self.state_axes[-1]:
            if axis.get_visible():
                axis.set_xlabel("time (s)")
        self.state_figure.tight_layout(rect=(0, 0, 1, 0.96))

    def _build_measurements(self, sensors):
        signature = tuple(
            (name, len(samples[0][1])) for name, samples in sensors.items()
        )
        if signature == self.measurement_signature:
            return
        self.measurement_signature = signature
        self.measurement_figure.clear()
        columns = max(length for _, length in signature)
        self.measurement_axes = self.measurement_figure.subplots(
            len(signature), columns, sharex=True, squeeze=False
        )
        self.measurement_lines = {}
        for row, (sensor, length) in enumerate(signature):
            labels = ("x", "y", "z") if length == 3 else None
            lines = []
            for column, axis in enumerate(self.measurement_axes[row]):
                if column >= length:
                    axis.set_visible(False)
                    continue
                (line,) = axis.plot([], [])
                lines.append(line)
                label = labels[column] if labels else str(column)
                axis.set_title(f"{sensor} {label}")
                axis.grid(True)
            self.measurement_lines[sensor] = lines
        for axis in self.measurement_axes[-1]:
            if axis.get_visible():
                axis.set_xlabel("time (s)")
        self.measurement_figure.suptitle("CHUUDKF diagnostic measurements")
        self.measurement_figure.tight_layout(rect=(0, 0, 1, 0.95))

    def _projection_rows(self, data):
        rows = [
            (
                "Orientation",
                data.times,
                [quaternion_to_rpy(state[:4]) for state in data.states],
                ("roll", "pitch", "yaw"),
                "degrees",
                None,
            ),
            (
                "Position",
                data.times,
                [state[4:7] for state in data.states],
                ("x", "y", "z"),
                "m",
                None,
            ),
            (
                "Velocity",
                data.times,
                [state[7:10] for state in data.states],
                ("x", "y", "z"),
                "m/s",
                None,
            ),
            (
                "Wheel scales",
                data.times,
                [state[10:12] for state in data.states],
                ("mu_c", "mu_d"),
                "scale",
                None,
            ),
        ]
        if "accel_inertial" in data.sensors:
            samples = data.sensors["accel_inertial"]
            rows.append(
                (
                    "Inertial acceleration",
                    [time for time, _ in samples],
                    [value for _, value in samples],
                    ("x", "y", "z"),
                    "m/s²",
                    None,
                )
            )
        if "gyro_inertial" in data.sensors:
            samples = data.sensors["gyro_inertial"]
            rows.append(
                (
                    "Inertial gyro",
                    [time for time, _ in samples],
                    [value for _, value in samples],
                    ("x", "y", "z"),
                    "rad/s",
                    None,
                )
            )
        if "wheels_inertial" in data.sensors:
            samples = data.sensors["wheels_inertial"]
            rows.append(
                (
                    "Wheel velocity",
                    [time for time, _ in samples],
                    [value for _, value in samples],
                    ("x", "y", "z"),
                    "m/s",
                    None,
                )
            )
        if "magnm_inertial" in data.sensors:
            samples = data.sensors["magnm_inertial"]
            rows.append(
                (
                    "Inertial magnetometer",
                    [time for time, _ in samples],
                    [
                        [component * TESLA_TO_MICROTESLA for component in value]
                        for _, value in samples
                    ],
                    ("x", "y", "z"),
                    "µT",
                    FIELD_STRENGTH_UT,
                )
            )
        return rows

    def _build_projections(self, rows):
        signature = tuple(row[0] for row in rows)
        if signature == self.projection_signature:
            return
        self.projection_signature = signature
        self.projection_figure.clear()
        self.projection_axes = self.projection_figure.subplots(
            len(rows), 3, sharex=True, squeeze=False
        )
        self.projection_lines = {}
        self.projection_trend_lines = {}
        self.overlay_axes = []
        self.correction_spans = []
        state_titles = {group[0] for group in STATE_GROUPS}
        for row_index, (name, _, _, labels, unit, radius) in enumerate(rows):
            lines = []
            trend_lines = []
            for component, axis in enumerate(self.projection_axes[row_index]):
                if component >= len(labels):
                    axis.set_visible(False)
                    continue

                (line,) = axis.plot(
                    [], [], linewidth=0.7, alpha=0.45, label="raw"
                )
                (trend_line,) = axis.plot(
                    [], [], linewidth=1.6, label="Gaussian trend"
                )
                lines.append(line)
                trend_lines.append(trend_line)
                if name in state_titles:
                    self.overlay_axes.append(axis)
                if radius is not None:
                    axis.axhline(radius, color="black", linewidth=0.8, alpha=0.5)
                    axis.axhline(-radius, color="black", linewidth=0.8, alpha=0.5)
                axis.set_title(f"{name}: {labels[component]}")
                axis.set_ylabel(unit)
                axis.grid(True, alpha=0.25)
                axis.legend(loc="best", fontsize=7)
            self.projection_lines[name] = lines
            self.projection_trend_lines[name] = trend_lines
        for axis in self.projection_axes[-1]:
            if axis.get_visible():
                axis.set_xlabel("elapsed time (s)")
        self.dt_axis = self.projection_figure.add_axes([0.81, 0.60, 0.17, 0.30])
        (self.dt_line,) = self.dt_axis.plot([], [])
        self.dt_axis.set_title("State timestamp dt")
        self.dt_axis.set_xlabel("elapsed time (s)")
        self.dt_axis.set_ylabel("dt (ms)")
        self.dt_axis.grid(True, alpha=0.25)
        self.projection_figure.suptitle(
            f"CHUUDKF full trajectory — Gaussian trend σ={self.smooth_sigma:g} "
            "samples — shading: green = corrections applied, red = rejected"
        )
        self.projection_figure.subplots_adjust(
            left=0.07,
            right=0.78,
            bottom=0.05,
            top=0.93,
            hspace=0.48,
            wspace=0.35,
        )

    def update(self, data):
        rows = self._projection_rows(data)
        self._build_projections(rows)
        self._build_nis(data.nis)
        self._update_projections(rows, data)
        self._update_correction_regions(data.corrections)
        self._update_nis(data)
        if self.autocorr_figure is not None:
            self._update_autocorrelation(data)
        for figure in self.figures:
            figure.canvas.draw_idle()
            figure.canvas.flush_events()

    def _update_nis(self, data):
        if not data.nis or self.nis_signature is None:
            return
        for sensor, lines in self.nis_lines.items():
            samples = data.nis[sensor]
            if self.max_points is not None:
                samples = samples[-self.max_points :]
            times = [sample[0] for sample in samples]
            nis_values = [sample[1] for sample in samples]
            lower_values = [sample[2] for sample in samples]
            upper_values = [sample[3] for sample in samples]
            for line, values in zip(
                lines, (nis_values, lower_values, upper_values)
            ):
                line.set_data(times, values)
            axis = lines[0].axes
            axis.set_xlim(*padded_limits(times))
            axis.set_ylim(
                *padded_limits([0.0, *nis_values, *lower_values, *upper_values])
            )

    def _update_autocorrelation(self, data):
        states = data.states
        times = data.times
        if self.max_points is not None:
            states = states[-self.max_points :]
            times = times[-self.max_points :]
        values = [state_vectors(state) for state in states]
        for row, (_, labels, _) in enumerate(STATE_GROUPS):
            for column in range(len(labels)):
                component_values = [value[row][column] for value in values]
                for lag, line in zip(
                    AUTOCORR_LAGS, self.autocorr_lines[row][column]
                ):
                    correlations = rolling_autocorrelation(
                        component_values,
                        lag,
                        self.autocorr_window,
                        unwrap_degrees=row == 0,
                    )
                    line.set_data(times, correlations)
                axis = self.autocorr_axes[row, column]
                if times:
                    axis.set_xlim(*padded_limits(times))

    def _update_state(self, data):
        times = data.times[-self.max_points :]
        states = data.states[-self.max_points :]
        values = [state_vectors(state) for state in states]
        for row, (_, labels, _) in enumerate(STATE_GROUPS):
            for column in range(len(labels)):
                line = self.state_lines[row][column]
                line.set_data(times, [value[row][column] for value in values])
                axis = self.state_axes[row, column]
                axis.relim()
                axis.autoscale_view()
        self.state_figure.suptitle(
            f"CHUUDKF state — t={data.times[-1]:.1f} s"
        )

    def _update_measurements(self, data):
        for sensor, lines in self.measurement_lines.items():
            samples = data.sensors[sensor][-self.max_points :]
            times = [time for time, _ in samples]
            for column, line in enumerate(lines):
                line.set_data(times, [value[column] for _, value in samples])
                axis = line.axes
                axis.relim()
                axis.autoscale_view()

    def _correction_segments(self, corrections):
        """Majority-vote applied/rejected per time bin, merged into spans."""
        bins = {}
        for elapsed, applied in corrections:
            index = int(elapsed // CORRECTION_BIN_SECONDS)
            counts = bins.setdefault(index, [0, 0])
            counts[applied] += 1
        segments = []
        for index in sorted(bins):
            rejected_count, applied_count = bins[index]
            good = applied_count >= rejected_count
            start = index * CORRECTION_BIN_SECONDS
            end = start + CORRECTION_BIN_SECONDS
            if (
                segments
                and segments[-1][2] == good
                and abs(segments[-1][1] - start) < 1e-9
            ):
                segments[-1][1] = end
            else:
                segments.append([start, end, good])
        return segments

    def _update_correction_regions(self, corrections):
        for span in self.correction_spans:
            span.remove()
        self.correction_spans = []
        if self.max_points is not None:
            corrections = corrections[-self.max_points :]
        for start, end, good in self._correction_segments(corrections):
            color = "tab:green" if good else "tab:red"
            for axis in self.overlay_axes:
                self.correction_spans.append(
                    axis.axvspan(
                        start,
                        end,
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                        zorder=0,
                    )
                )

    def _update_projections(self, rows, data):
        for name, sample_times, values, _, _, _ in rows:
            if self.max_points is not None:
                sample_times = sample_times[-self.max_points :]
                values = values[-self.max_points :]
            if not sample_times:
                continue
            for component, line in enumerate(self.projection_lines[name]):
                component_values = [value[component] for value in values]
                line.set_data(sample_times, component_values)
                trend = gaussian_trend(
                    component_values,
                    self.smooth_sigma,
                    unwrap_degrees=name == "Orientation",
                )
                self.projection_trend_lines[name][component].set_data(
                    sample_times, trend
                )
                line.axes.set_xlim(*padded_limits(sample_times))
                line.axes.set_ylim(
                    *padded_limits([*component_values, *trend.tolist()])
                )
        self.dt_line.set_data(data.dt_times, data.dt_ms)
        if data.dt_times:
            self.dt_axis.set_xlim(*padded_limits(data.dt_times))
            self.dt_axis.set_ylim(*padded_limits(data.dt_ms))


def static_plot(
    log_path,
    smooth_sigma=DEFAULT_SMOOTH_SIGMA,
    autocorr_window=DEFAULT_AUTOCORR_WINDOW,
    last=None,
    time_range=None,
    max_points=DEFAULT_STATIC_MAX_POINTS,
):
    data, stats = load_sparse_static_data(
        log_path,
        max_points=max_points,
        last=last,
        time_range=time_range,
    )

    print(
        f"scanned {stats['complete_rows']} complete rows; "
        f"selected {stats['selected_rows']}"
    )
    print(
        f"retained {stats['retained_states']} state points and "
        f"{stats['retained_measurements']} measurement points"
    )
    print(f"trajectory duration: {data.times[-1] - data.times[0]:.3f} s")

    plots = LivePlots(
        None,
        smooth_sigma,
        autocorr_window,
        show_autocorrelation=True,
    )
    plots.update(data)
    plt.show()


def live_plot(
    log_path,
    refresh,
    max_points,
    smooth_sigma=DEFAULT_SMOOTH_SIGMA,
):
    tail = LogTail(log_path)
    data = LiveData(max_points)
    plots = None
    waiting_message_shown = False
    plt.ion()
    try:
        while True:
            rows, reset = tail.read_new(recent_rows=max_points)
            if reset:
                data.reset()
            if rows:
                data.extend(rows)
                if plots is None:
                    plots = LivePlots(max_points, smooth_sigma)
                    plt.show(block=False)
                plots.update(data)
                waiting_message_shown = False
            elif plots is None and not waiting_message_shown:
                print(f"waiting for complete filter rows in {log_path}")
                waiting_message_shown = True

            if plots is not None and not all(
                plt.fignum_exists(figure.number) for figure in plots.figures
            ):
                break
            if plots is None:
                time.sleep(refresh)
            else:
                plt.pause(refresh)
    except KeyboardInterrupt:
        pass
    finally:
        tail.close()
        plt.ioff()


def main():
    parser = argparse.ArgumentParser(
        description="Static or live CHUUDKF state and measurement plot"
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=LOG_FILE,
        help=f"CHUUDKF CSV log (default: {LOG_FILE})",
    )
    parser.add_argument(
        "--live-plot",
        "--live_plot",
        "--live",
        dest="live_plot",
        action="store_true",
        help="follow the CSV and update as new filter rows arrive",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.25,
        metavar="SECONDS",
        help="live refresh interval (default: 0.25)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        metavar="COUNT",
        help="maximum rows retained in live mode (default: 5000)",
    )
    parser.add_argument(
        "--static-max-points",
        type=int,
        default=DEFAULT_STATIC_MAX_POINTS,
        metavar="COUNT",
        help=(
            "maximum points retained per state/sensor series in static mode "
            f"(default: {DEFAULT_STATIC_MAX_POINTS})"
        ),
    )
    parser.add_argument(
        "--last",
        type=float,
        metavar="SECONDS",
        help="static mode: only plot the last SECONDS of the log",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        dest="time_range",
        help="static mode: only plot elapsed seconds START to END",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=DEFAULT_SMOOTH_SIGMA,
        metavar="SAMPLES",
        help=(
            "Gaussian trend standard deviation in samples "
            f"(default: {DEFAULT_SMOOTH_SIGMA:g})"
        ),
    )
    parser.add_argument(
        "--autocorr-window",
        type=int,
        default=DEFAULT_AUTOCORR_WINDOW,
        metavar="SAMPLES",
        help=(
            "rolling autocorrelation window size for static plots "
            f"(default: {DEFAULT_AUTOCORR_WINDOW})"
        ),
    )
    args = parser.parse_args()

    if args.refresh <= 0:
        parser.error("--refresh must be greater than zero")
    if args.max_points <= 0:
        parser.error("--max-points must be greater than zero")
    if args.static_max_points <= 0:
        parser.error("--static-max-points must be greater than zero")
    if args.smooth_sigma <= 0:
        parser.error("--smooth-sigma must be greater than zero")
    if args.autocorr_window <= max(AUTOCORR_LAGS):
        parser.error(
            f"--autocorr-window must be greater than {max(AUTOCORR_LAGS)}"
        )
    if args.time_range is not None and args.time_range[1] < args.time_range[0]:
        parser.error("--range END must be greater than or equal to START")

    if args.live_plot:
        live_plot(
            args.log,
            args.refresh,
            args.max_points,
            args.smooth_sigma,
        )
    else:
        static_plot(
            args.log,
            args.smooth_sigma,
            args.autocorr_window,
            args.last,
            args.time_range,
            args.static_max_points,
        )


if __name__ == "__main__":
    main()