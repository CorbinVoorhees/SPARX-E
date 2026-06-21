import re
import mmap
import numpy as np


_FLOAT_RE = rb"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"


class Parser:
    def __init__(self, ekf_log_path: str, kalman_state_log: str, mekf_log_path: str):
        self.ekf_log_path = ekf_log_path
        self.kalman_state_log = kalman_state_log
        self.mekf_log_path = mekf_log_path

    # -------------------------
    # generic helpers
    # -------------------------

    def __to_bytes(self, x):
        if isinstance(x, bytes):
            return x
        return x.encode("utf-8")

    def __extract_blocks(self, text, title: str):
        """
        Returns all blocks matching:

        ========== TITLE ==========
        ...
        =====================================
        """
        title_b = self.__to_bytes(title)

        pattern = (
            rb"=+\s*" + re.escape(title_b) + rb"\s*=+"
            rb"(.*?)"
            rb"(?=\n=+\s*[A-Z][A-Z _]*\s*=+|\Z)"
        )

        return [m.group(1) for m in re.finditer(pattern, text, flags=re.S)]

    def __extract_section(self, block, section_name: str):
        """
        Extracts:

        --- STATE ---
        ...
        --- SENSOR RAW ---
        """
        section_b = self.__to_bytes(section_name)

        pattern = (
            rb"---\s*" + re.escape(section_b) + rb"\s*---"
            rb"(.*?)"
            rb"(?=\n---|\n=+|\Z)"
        )

        match = re.search(pattern, block, flags=re.S)
        return match.group(1) if match else b""

    def __parse_keyvals(self, text, label: str):
        """
        Parses:

        pos      : x=1.0, y=2.0, z=3.0
        rangeDot : raw=0.1, mean=0.2, std=0.3
        time     : t=123.0, dt=0.025
        """
        label_b = self.__to_bytes(label)

        pattern = re.escape(label_b) + rb"\s*:\s*(.+)"
        match = re.search(pattern, text)

        if not match:
            return None

        line = match.group(1)

        pairs = re.findall(
            rb"([A-Za-z_]\w*)\s*=\s*(" + _FLOAT_RE + rb")",
            line
        )

        return {k.decode("utf-8"): float(v) for k, v in pairs}

    def __parse_matrix(self, text, label: str):
        """
        Parses:

        K:
        1 2 3
        4 5 6

        Stops at next matrix-ish label, section, block, or EOF.
        """
        label_b = self.__to_bytes(label)

        pattern = (
            re.escape(label_b)
            + rb"\s*:\s*\n"
            + rb"(.*?)"
            + rb"(?=\n[A-Za-z_]\w*\s*:|\n---|\n=+|\Z)"
        )
        
        match = re.search(pattern, text, flags=re.S)

        if not match:
            return None

        rows = []

        for line in match.group(1).splitlines():
            nums = re.findall(_FLOAT_RE, line)
            if nums:
                rows.append([float(x) for x in nums])

        if not rows:
            return None

        # --- NEW CODE: Enforce homogeneous shape ---
        # Assume the length of the first parsed row is the correct matrix width
        expected_len = len(rows[0])
        
        # Filter out any malformed rows (like stray log lines)
        valid_rows = [r for r in rows if len(r) == expected_len]

        if not valid_rows:
            return None

        return np.array(valid_rows, dtype=float)

    def __read_mmap(self, path: str):
        with open(path, "rb") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return mm[:]

    # -------------------------
    # block parsers
    # -------------------------

    def __parse_mekf_block(self, block):
        prior = self.__extract_section(block, "PRIOR")
        pred = self.__extract_section(block, "PRED")
        update = self.__extract_section(block, "UPDATE")
        var = self.__extract_section(block, "VARIANCE")
        mats = self.__extract_section(block, "MATRICES")

        return {
            "time": self.__parse_keyvals(block, "time"),

            "prior": {
                "q_prior": self.__parse_keyvals(prior, "q_prior"),
                "b_prior": self.__parse_keyvals(prior, "b_prior"),
                "gyro_raw": self.__parse_keyvals(prior, "gyro_raw"),
                "gyro_cor": self.__parse_keyvals(prior, "gyro_cor"),
            },

            "pred": {
                "q_pred": self.__parse_keyvals(pred, "q_pred"),
                "q_meas": self.__parse_keyvals(pred, "q_meas"),
                "q_err": self.__parse_keyvals(pred, "q_err"),
                "dtheta": self.__parse_keyvals(pred, "dtheta"),
            },

            "update": {
                "dx_theta": self.__parse_keyvals(update, "dx_theta"),
                "dx_bias": self.__parse_keyvals(update, "dx_bias"),
                "dq": self.__parse_keyvals(update, "dq"),
                "q_out": self.__parse_keyvals(update, "q_out"),
                "b_out": self.__parse_keyvals(update, "b_out"),
                "nis": self.__parse_keyvals(update, "nis"),
            },

            "variance": {
                "P_diag": self.__parse_keyvals(var, "P_diag"),
            },

            "matrices": {
                "F": self.__parse_matrix(mats, "F"),
                "Q": self.__parse_matrix(mats, "Q"),
                "H": self.__parse_matrix(mats, "H"),
                "R": self.__parse_matrix(mats, "R"),
                "S": self.__parse_matrix(mats, "S"),
                "K": self.__parse_matrix(mats, "K"),
                "P": self.__parse_matrix(mats, "P"),
            },
        }

    def __parse_trans_block(self, block):
        state_prior = self.__extract_section(block, "STATE PRIOR")
        state_pred = self.__extract_section(block, "STATE PRED")
        inputs = self.__extract_section(block, "INPUTS")
        model = self.__extract_section(block, "MODEL")
        meas = self.__extract_section(block, "MEASUREMENT")
        var = self.__extract_section(block, "VARIANCE")
        mats = self.__extract_section(block, "MATRICES")

        return {
            "time": self.__parse_keyvals(block, "time"),

            "state_prior": {
                "pos": self.__parse_keyvals(state_prior, "pos"),
                "vel": self.__parse_keyvals(state_prior, "vel"),
                "att": self.__parse_keyvals(state_prior, "att"),
                "accBias": self.__parse_keyvals(state_prior, "accBias"),
                "rngBias": self.__parse_keyvals(state_prior, "rngBias"),
            },

            "state_pred": {
                "pos": self.__parse_keyvals(state_pred, "pos"),
                "vel": self.__parse_keyvals(state_pred, "vel"),
                "att": self.__parse_keyvals(state_pred, "att"),
                "accBias": self.__parse_keyvals(state_pred, "accBias"),
                "rngBias": self.__parse_keyvals(state_pred, "rngBias"),
            },

            "inputs": {
                "imu_raw": self.__parse_keyvals(inputs, "imu_raw"),
                "imu_corr": self.__parse_keyvals(inputs, "imu_corr"),
                "uwb": self.__parse_keyvals(inputs, "uwb"),
                "ctrl": self.__parse_keyvals(inputs, "ctrl"),
                "wheel": self.__parse_keyvals(inputs, "wheel"),
            },

            "model": {
                "acc_model": self.__parse_keyvals(model, "acc_model"),
                "ang_model": self.__parse_keyvals(model, "ang_model"),
                "range_model": self.__parse_keyvals(model, "range_model"),
            },

            "measurement": {
                "z": self.__parse_keyvals(meas, "z"),
                "z_pred": self.__parse_keyvals(meas, "z_pred"),
                "innov": self.__parse_keyvals(meas, "innov"),
                "nis": self.__parse_keyvals(meas, "nis"),
            },

            "variance": {
                "pos": self.__parse_keyvals(var, "pos"),
                "vel": self.__parse_keyvals(var, "vel"),
                "att": self.__parse_keyvals(var, "att"),
                "accBias": self.__parse_keyvals(var, "accBias"),
                "rngBias": self.__parse_keyvals(var, "rngBias"),
            },

            "matrices": {
                "F": self.__parse_matrix(mats, "F"),
                "W": self.__parse_matrix(mats, "W"),
                "R": self.__parse_matrix(mats, "R"),
                "P_pred": self.__parse_matrix(mats, "P_pred"),
                "H": self.__parse_matrix(mats, "H"),
                "S": self.__parse_matrix(mats, "S"),
                "K": self.__parse_matrix(mats, "K"),
            },
        }

    def __parse_nav_block(self, block):
        state = self.__extract_section(block, "STATE")
        sensor = self.__extract_section(block, "SENSOR RAW")
        profile = self.__extract_section(block, "PROFILE")
        cov = self.__extract_section(block, "COVARIANCE")

        return {
            "time": self.__parse_keyvals(block, "time"),

            "state": {
                "pos": self.__parse_keyvals(state, "pos"),
                "vel": self.__parse_keyvals(state, "vel"),
                "quat": self.__parse_keyvals(state, "quat"),
                "rpy": self.__parse_keyvals(state, "rpy"),
                "gyroBias": self.__parse_keyvals(state, "gyroBias"),
                "accBias": self.__parse_keyvals(state, "accBias"),
                "rngBias": self.__parse_keyvals(state, "rngBias"),
            },

            "sensor_raw": {
                "accel": self.__parse_keyvals(sensor, "accel"),
                "accWorld": self.__parse_keyvals(sensor, "accWorld"),
                "gyro": self.__parse_keyvals(sensor, "gyro"),
                "range": self.__parse_keyvals(sensor, "range"),
                "rangeDot": self.__parse_keyvals(sensor, "rangeDot"),
            },

            "profile": {
                "accMean": self.__parse_keyvals(profile, "accMean"),
                "accStd": self.__parse_keyvals(profile, "accStd"),
                "gyroMean": self.__parse_keyvals(profile, "gyroMean"),
                "gyroStd": self.__parse_keyvals(profile, "gyroStd"),
                "rangeMean": self.__parse_keyvals(profile, "rangeMean"),
                "rangeStd": self.__parse_keyvals(profile, "rangeStd"),
                "accEKF": self.__parse_keyvals(profile, "accEKF"),
                "accProf": self.__parse_keyvals(profile, "accProf"),
                "samples": self.__parse_keyvals(profile, "samples"),
            },

            "covariance": {
                "P_rot_diag": self.__parse_keyvals(cov, "P_rot_diag"),
                "P_loc_pos": self.__parse_keyvals(cov, "P_loc_pos"),
                "P_loc_vel": self.__parse_keyvals(cov, "P_loc_vel"),
                "P_loc_att": self.__parse_keyvals(cov, "P_loc_att"),
                "P_loc_bias": self.__parse_keyvals(cov, "P_loc_bias"),
            },
        }

    # -------------------------
    # public API
    # -------------------------

    def parse_logs(self):
        out = {}

        ekf_text = self.__read_mmap(self.ekf_log_path)
        trans_blocks = self.__extract_blocks(ekf_text, "TRANSLATIONAL EKF")
        out["translational"] = [self.__parse_trans_block(b) for b in trans_blocks]

        nav_text = self.__read_mmap(self.kalman_state_log)
        nav_blocks = self.__extract_blocks(nav_text, "NAV NODE OUT")
        out["nav"] = [self.__parse_nav_block(b) for b in nav_blocks]

        if self.mekf_log_path is not None:
            mekf_text = self.__read_mmap(self.mekf_log_path)
            mekf_blocks = self.__extract_blocks(mekf_text, "MEKF OUT")
            out["mekf"] = [self.__parse_mekf_block(b) for b in mekf_blocks]

        return out