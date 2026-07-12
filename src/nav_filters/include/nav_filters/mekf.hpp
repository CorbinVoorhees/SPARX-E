#pragma once

#include "ckf.hpp"
#include "nav_filters/ISensor.hpp"
#include <Eigen/Dense>

class MEKF : public ckf::CommonKF {
private:
  Eigen::Matrix3d Qw;
  snapshot gyro_snapshot;
  snapshot control_snapshot;

  // gyro offset = startup mean, subtracted as a plain constant (no online
  // bias estimation in the state).
  Eigen::Vector3d bg0{Eigen::Vector3d::Zero()};

  // wheel-odometry yaw, integrated from (wr - wl) — start-relative, used as a
  // 1-D yaw aiding measurement. R is its (slip-dependent) noise, 1x1.
  double yaw_wheel{0.0};
  Eigen::MatrixXd R_yaw_odo{
      Eigen::MatrixXd::Constant(1, 1, 0.01)}; // (~5.7 deg)^2

  auto q() { return x.segment<4>(0); }

  void apply_error(const Eigen::VectorXd &err) override {
    const Eigen::Vector3d dtheta = err.segment<3>(0);
    const Eigen::Vector4d o = q();
    Eigen::Quaterniond q_curr(o(0), o(1), o(2), o(3));
    Eigen::Quaterniond dq(1.0, 0.5 * dtheta.x(), 0.5 * dtheta.y(),
                          0.5 * dtheta.z());
    Eigen::Quaterniond q_new = (q_curr * dq).normalized();
    if (q_new.w() < 0.0)
      q_new.coeffs() *= -1.0;
    q() << q_new.w(), q_new.x(), q_new.y(), q_new.z();
  }

  void predict(double dt) override {
    if (!gyro_snapshot.fresh)
      return;

    const Eigen::Vector3d w_corr = gyro_snapshot.z - bg0;

    const Eigen::Vector4d o = q();
    Eigen::Quaterniond prior_q(o(0), o(1), o(2), o(3));
    prior_q.normalize();

    const Eigen::Vector3d theta = w_corr * dt;
    const double angle = theta.norm();
    Eigen::Quaterniond dq = Eigen::Quaterniond::Identity();
    if (angle > 1e-12)
      dq = Eigen::Quaterniond(Eigen::AngleAxisd(angle, theta / angle));

    Eigen::Quaterniond q_pred = (prior_q * dq).normalized();
    if (q_pred.w() < 0.0)
      q_pred.coeffs() *= -1.0;
    q() << q_pred.w(), q_pred.x(), q_pred.y(), q_pred.z();

    const Eigen::Matrix3d F = -skew(w_corr);
    const Eigen::Matrix3d Qk = Qw * dt;
    const Eigen::Matrix3d Phi = Eigen::Matrix3d::Identity() + F * dt;
    P = Phi * P * Phi.transpose() + Qk;
  }

  Eigen::Quaterniond q0_ref{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d magnm_mean{Eigen::Vector3d::Zero()};
  Eigen::MatrixXd R_accel, R_magnm;
  bool accel_finished{false};

  // offline magnetometer calibration (see analysis/calibrate.py
  // fit_ellipsoid / save_mag_calibration). m_cal = W_mag * (m_raw - bias_mag).
  // Defaults to identity/zero (no-op) until a calibration file is found.
  // Loaded lazily from the magnm init callback, not the constructor, since the
  // file may not exist yet at construction time.
  std::string mag_calibration_path{"mag_calibration.txt"};
  Eigen::Matrix3d W_mag{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d bias_mag{Eigen::Vector3d::Zero()};

  void load_mag_calibration() {
    std::vector<double> vals;
    if (!load_doubles(this->mag_calibration_path, vals, 12)) {
      std::cout << "[MEKF] no mag calibration at '"
                << this->mag_calibration_path
                << "' — using identity/zero (raw magnetometer)\n";
      return; // identity/zero stays in effect
    }
    this->W_mag =
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(vals.data());
    this->bias_mag = Eigen::Vector3d(vals[9], vals[10], vals[11]);
    std::cout << "[MEKF] loaded mag calibration from '"
              << this->mag_calibration_path << "'\n";
  }

public:
  bool suppress_magnm = false;
  bool is_finished_orienting() { return this->accel_finished; }

  MEKF(SteadyClock::time_point t0) : ckf::CommonKF(4, 3, t0) {
    using V3 = Eigen::Vector3d;
    using V4 = Eigen::Vector4d;
    x(0) = 1.0;
    Qw = Eigen::Matrix3d::Identity() * 1e-5;

    Sensor::SensorTable::bind<V3>(
        "accel",
        [this](const V3 &z, const Eigen::MatrixXd &R,
               SteadyClock::time_point t) {
          const double z_norm = std::max(z.norm(), 1e-9);
          const Eigen::MatrixXd R_accel_dir = R / (z_norm * z_norm);
          this->queue_correction(
              t, R_accel_dir,
              [this, z](Eigen::MatrixXd &H, Eigen::VectorXd &r,
                        Eigen::VectorXd &z_out) {
                const Eigen::Vector4d o = q();
                const Eigen::Quaterniond orientation(o(0), o(1), o(2), o(3));

                const Eigen::Vector3d grav_body =
                    orientation.normalized().toRotationMatrix().transpose() *
                    grav_vec;

                const double z_norm = z.norm();
                const double g_norm = grav_body.norm();
                if (z_norm < 1e-9 || g_norm < 1e-9)
                  return false;

                const Eigen::Vector3d z_dir = z / z_norm;
                const Eigen::Vector3d g_dir = grav_body / g_norm;
                const Eigen::Matrix3d tangent_projector =
                    Eigen::Matrix3d::Identity() - g_dir * g_dir.transpose();

                H = Eigen::MatrixXd::Zero(3, 3);
                H.block<3, 3>(0, 0) =
                    (tangent_projector / g_norm) * skew(grav_body);
                r = z_dir - g_dir;
                z_out = z_dir;
                return true;
              },
              "accel");
        },
        [this](const FilteredSampleProducer<V3> &prod) {
          const Eigen::Vector3d accel_mean = prod.get_mean();

          // z-down inertial frame: the rest accel (pointing up in body
          // coords) aligns to -Z, so q0 is near-identity for a level start
          this->q0_ref = Eigen::Quaterniond::FromTwoVectors(
              accel_mean.normalized(), -Eigen::Vector3d::UnitZ());

          q() << q0_ref.w(), q0_ref.x(), q0_ref.y(), q0_ref.z();

          P.diagonal().segment<3>(0).setConstant(
              std::pow(0.5 * M_PI / 180.0, 2));

          const double accel_norm = std::max(accel_mean.norm(), 1e-9);
          const Eigen::Matrix3d R_accel_raw = prod.get_variance().asDiagonal();
          R_accel = R_accel_raw / (accel_norm * accel_norm);
          accel_finished = true;
        });

    Sensor::SensorTable::bind<V3>(
        "magnm",
        [this](const V3 &z, const Eigen::MatrixXd &R,
               SteadyClock::time_point t) {
          const Eigen::MatrixXd R_magnm_cal =
              this->W_mag * R * this->W_mag.transpose();
          this->queue_correction(
              t, R_magnm_cal,
              [this, z](Eigen::MatrixXd &H, Eigen::VectorXd &r,
                        Eigen::VectorXd &z_out) {
                if (!this->accel_finished)
                  return false;

                // hard + soft iron correction (identity/zero => raw)
                const Eigen::Vector3d z_cal =
                    this->W_mag * (z - this->bias_mag);

                // the residual must be measurement minus what the ATTITUDE
                // predicts (startup reference rotated into the current body
                // frame) — comparing against the constant magnm_mean removes
                // the state from the loop and turns the correction into an
                // open integrator (constant-rate fake yaw spin).
                const Eigen::Vector4d o = this->q();
                const Eigen::Quaterniond orientation(o(0), o(1), o(2), o(3));
                const Eigen::Vector3d mean_north =
                    q0_ref.toRotationMatrix() * magnm_mean;
                const Eigen::Vector3d mag_body =
                    orientation.normalized().toRotationMatrix().transpose() *
                    mean_north;

                H = Eigen::MatrixXd::Zero(3, 3);
                H.block<3, 3>(0, 0) = skew(mag_body);
                r = z_cal - mag_body;
                z_out = z_cal;

                return true;
              },
              "magnm");
        },
        [this](const FilteredSampleProducer<V3> &prod) {
          this->load_mag_calibration();

          // affine, so calibrating the mean is equivalent to calibrating
          // every raw sample before averaging
          magnm_mean = this->W_mag * (prod.get_mean() - this->bias_mag);

          const Eigen::Matrix3d R_raw = prod.get_variance().asDiagonal();
          const Eigen::Matrix3d R_cal =
              this->W_mag * R_raw * this->W_mag.transpose();

          R_magnm = R_cal;
        });

    Sensor::SensorTable::bind<V3>(
        "gyro",
        [this](const V3 &z, const Eigen::MatrixXd &,
               SteadyClock::time_point t) {
          queue_task(t, [this, z, t]() { this->gyro_snapshot = {t, z, true}; });
        },
        [this](const FilteredSampleProducer<V3> &prod) {
          // plain constant offset from the startup mean — not estimated online
          bg0 = prod.get_mean();
        });
  };

  Eigen::Quaterniond orientation() const {
    return Eigen::Quaterniond(x(0), x(1), x(2), x(3)).normalized();
  }

  Eigen::Quaterniond orientation0() const { return this->q0_ref.normalized(); }

  Eigen::Quaterniond rel_orientation() const {
    auto curr = orientation();
    auto zero = orientation0();

    return (zero.conjugate() * curr).normalized();
  }

  Eigen::Vector3d get_bg() { return this->bg0; }

  Eigen::Vector3d rpy() const {
    const Eigen::Matrix3d R = this->orientation().toRotationMatrix();
    const double roll = std::atan2(R(2, 1), R(2, 2)) * 180.0 / M_PI;
    const double pitch =
        std::asin(std::max(-1.0, std::min(1.0, -R(2, 0)))) * 180.0 / M_PI;
    const double yaw = std::atan2(R(1, 0), R(0, 0)) * 180.0 / M_PI;

    return {roll, pitch, yaw};
  }

  Eigen::Vector3d gyro_bias() const { return bg0; }
};
