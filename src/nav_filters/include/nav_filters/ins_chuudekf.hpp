#pragma once


#include "sensor/sensor.hpp"
#include "ckf.hpp"
#include <Eigen/Dense>

// INS variant of CHUUDEKF: velocity is a strapdown state integrated from
// accel; the wheel model is a correction (not the definition), so slip shows
// up as a wheel-vs-INS residual instead of being trusted as truth.
//
// Nominal state:
//   q(4), p(3), v(3), mu_r, mu_l              size 12
//
// Error state:
//   dtheta(3), dp(3), dv(3), dmu_r, dmu_l     size 11
class INS_CHUUDEKF : public ckf::CommonKF {
private:
  using V3 = Eigen::Vector3d;
  using V4 = Eigen::Vector4d;

  // ---- MEKF members ----
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

  Eigen::Quaterniond q0_ref{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d magnm_mean{Eigen::Vector3d::Zero()};
  Eigen::MatrixXd R_accel, R_magnm;
  bool accel_finished{false};

  // ---- TRANSEKF members ----
  snapshot accel_snapshot;
  snapshot uwb_snapshot;

  V3 grav_ref = grav_vec;

  // ---- accessors ----
  // mu is estimated in common/differential coordinates so the observable
  // (common) and hard-to-observe (differential) directions are separate states:
  //   mu_c = (mu_r + mu_l)/2   observable from any forward accel (dwr+dwl != 0)
  //   mu_d = (mu_r - mu_l)/2   only observable with differential accel
  //   (dwr!=dwl)
  auto q() { return x.segment<4>(0); }
  auto position() { return x.segment<3>(4); }
  auto velocity() { return x.segment<3>(7); }
  double &mu_c() { return x(10); }
  double &mu_d() { return x(11); }
  double mu_r() { return mu_c() + mu_d(); }
  double mu_l() { return mu_c() - mu_d(); }

  void apply_error(const Eigen::VectorXd &err) override {
    // ---- MEKF apply_error (attitude) ----
    const Eigen::Vector3d dtheta = err.segment<3>(0);
    const Eigen::Vector4d o = q();
    Eigen::Quaterniond q_curr(o(0), o(1), o(2), o(3));
    Eigen::Quaterniond dq(1.0, 0.5 * dtheta.x(), 0.5 * dtheta.y(),
                          0.5 * dtheta.z());
    Eigen::Quaterniond q_new = (q_curr * dq).normalized();
    if (q_new.w() < 0.0)
      q_new.coeffs() *= -1.0;
    q() << q_new.w(), q_new.x(), q_new.y(), q_new.z();

    // ---- TRANSEKF apply_error (position / velocity / mu) ----
    position() += err.segment<3>(3);
    velocity() += err.segment<3>(6);
    mu_c() += err(9);
    mu_d() += err(10);
    mu_c() = std::max(mu_c(), 0.05); // common scale must stay positive
    // mu_d (differential) is signed — do not clamp
  }

  void predict(double dt) override {
    // ---- MEKF predict (attitude), verbatim ----
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

    const Eigen::Matrix3d F_att = -skew(w_corr);
    const Eigen::Matrix3d Qk = Qw * dt;
    const Eigen::Matrix3d Phi_att = Eigen::Matrix3d::Identity() + F_att * dt;
    P.block<3, 3>(0, 0) =
        Phi_att * P.block<3, 3>(0, 0) * Phi_att.transpose() + Qk;

    // ---- strapdown predict (position / velocity from accel) ----
    if (!is_finished_orienting() || !accel_snapshot.fresh)
      return;

    const Eigen::Quaterniond q_start_body = rel_orientation().normalized();
    const Eigen::Matrix3d R_I_B = q_start_body.toRotationMatrix();

    // Rotate calibrated body acceleration into the start-relative translation
    // frame, then subtract gravity captured in that same frame at startup.
    const V3 a_world = linear_accel_in_start_frame(
        q_start_body, accel_snapshot.z, grav_ref);

    position() += velocity() * dt;
    velocity() += a_world * dt;

    // p=3:6, v=6:9, mu=9:11. attitude block (0:3) already propagated above.
    Eigen::Matrix<double, 11, 11> dfdx = Eigen::Matrix<double, 11, 11>::Zero();
    dfdx.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    dfdx.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    dfdx.block<3, 3>(3, 6) = Eigen::Matrix3d::Identity() * dt;
    dfdx.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
    // d(velocity)/d(dtheta): rotating the measured accel through attitude error
    dfdx.block<3, 3>(6, 0) = -skew(R_I_B * accel_snapshot.z) * dt;
    dfdx(9, 9) = 1.0;
    dfdx(10, 10) = 1.0;

    Eigen::MatrixXd Qd = Eigen::MatrixXd::Zero(11, 11);
    // velocity Q raised: it integrates noisy accel now, so it needs headroom
    // for the wheel/UWB corrections to pull it. mu low (identified via corr).
    Qd.diagonal().segment<8>(3) << 1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1, 1e-6,
        1e-6;
    Qd.diagonal().segment<8>(3) *= dt;

    P = dfdx * P * dfdx.transpose() + Qd;
  }

public:
  bool suppress_magnm = false;
  bool is_finished_orienting() { return this->accel_finished; }

  // ---- MEKF outputs ----
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

  // ---- TRANSEKF outputs ----
  V3 get_position() const { return x.segment<3>(4); }
  V3 get_velocity() const { return x.segment<3>(7); }
  // x(10)=mu_c, x(11)=mu_d; report per-wheel for backward compatibility
  double get_mu_r() const { return x(10) + x(11); }
  double get_mu_l() const { return x(10) - x(11); }
  double get_mu_c() const { return x(10); }
  double get_mu_d() const { return x(11); }

  INS_CHUUDEKF(SteadyClock::time_point t0, Eigen::Vector3d pos0)
      : ckf::CommonKF(12, 11, t0) {
    using V3 = Eigen::Vector3d;
    using V4 = Eigen::Vector4d;

    // ---- MEKF init ----
    x(0) = 1.0;
    Qw = Eigen::Matrix3d::Identity() * 1e-5;

    // ---- TRANSEKF init ----
    control_snapshot.z = Eigen::Vector4d::Zero();
    control_snapshot.fresh = true;
    gyro_snapshot.z = V3::Zero();
    gyro_snapshot.fresh = true;

    position() = pos0;
    velocity() = Eigen::Vector3d::Zero();
    mu_c() = 0.5 * (GEOM.mu_r + GEOM.mu_l); // common scale (~0.95)
    mu_d() = 0.5 * (GEOM.mu_r - GEOM.mu_l); // differential (~0)

    this->P.diagonal().segment<3>(3) << 1e-4, 1e-4, 1e-4; // p
    this->P.diagonal().segment<3>(6) << 1e-3, 1e-3, 1e-3; // v
    this->P.diagonal().segment<2>(9) << 1e-6, 1e-6;       // mu (sigma ~0.05)

    // ================= MEKF accel (attitude) =================
    Sensor::SensorTable::bind<V3>(
        "accel",
        [this](const V3 &z, const Eigen::MatrixXd &R,
               SteadyClock::time_point t) {
          // ---- MEKF accel handler ----
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

                const Eigen::Vector3d z_dir = z / z_norm;
                const Eigen::Vector3d g_dir = grav_body / g_norm;
                const Eigen::Matrix3d tangent_projector =
                    Eigen::Matrix3d::Identity() - g_dir * g_dir.transpose();

                H = Eigen::MatrixXd::Zero(3, P.cols());
                H.block<3, 3>(0, 0) =
                    (tangent_projector / g_norm) * skew(grav_body);
                r = z_dir - g_dir;
                z_out = z_dir;
                return true;
              },
              "accel");

          queue_task(t, [this, z, t]() { accel_snapshot = {t, z, true}; });
        },
        [this](const Sensor::DataPrefilter<V3>::Stamped &sample) {
          // ---- MEKF accel init ----
          const Eigen::Vector3d accel_mean = sample.mean;
          this->grav_ref = accel_mean;

          this->q0_ref = Eigen::Quaterniond::FromTwoVectors(
              accel_mean.normalized(), Eigen::Vector3d::UnitZ());

          q() << q0_ref.w(), q0_ref.x(), q0_ref.y(), q0_ref.z();

          P.diagonal().segment<3>(0).setConstant(
              std::pow(0.5 * M_PI / 180.0, 2));

          const double accel_norm = std::max(accel_mean.norm(), 1e-9);
          const Eigen::Matrix3d R_accel_raw = sample.variance.asDiagonal();
          R_accel = R_accel_raw / (accel_norm * accel_norm);
          accel_finished = true;
        });

    // ================= MEKF magnm =================
    Sensor::SensorTable::bind<V3>(
        "magnm",
        [this](const V3 &z, const Eigen::MatrixXd &R,
               SteadyClock::time_point t) {
          if (!is_finished_orienting())
            return;

          // z arrives already calibrated (Sensor::DataPrefilter applies
          // mag_calibration.txt at the source) — no calibration here.
          this->queue_correction(
              t, R,
              [this, z](Eigen::MatrixXd &H, Eigen::VectorXd &r,
                        Eigen::VectorXd &z_out) {
                const Eigen::Vector4d o = this->q();
                const Eigen::Quaterniond orientation(o(0), o(1), o(2), o(3));
                const Eigen::Vector3d mean_north =
                    q0_ref.toRotationMatrix() * magnm_mean;
                const Eigen::Vector3d mag_body =
                    orientation.normalized().toRotationMatrix().transpose() *
                    mean_north;

                H = Eigen::MatrixXd::Zero(3, P.cols());
                H.block<3, 3>(0, 0) = skew(mag_body);
                r = z - mag_body;
                z_out = z;

                return true;
              },
              "magnm");
        },
        [this](const Sensor::DataPrefilter<V3>::Stamped &sample) {
          magnm_mean = sample.mean;
          R_magnm = sample.variance.asDiagonal();
        });

    // ================= TRANSEKF uwb =================
    Sensor::SensorTable::bind<double>(
        "uwb",
        [this](const double data, const Eigen::MatrixXd &,
               SteadyClock::time_point t) {
          queue_task(t, [this, data, t]() {
            uwb_snapshot = {t, Eigen::VectorXd::Constant(1, data), true};
          });
          const Eigen::MatrixXd R = Eigen::MatrixXd::Constant(1, 1, 0.04);
          queue_correction(
              t, R,
              [this, data](Eigen::MatrixXd &H, Eigen::VectorXd &r,
                           Eigen::VectorXd &z_out) {
                const V3 displacement = position();
                H = Eigen::MatrixXd::Zero(1, P.cols());
                H.block<1, 3>(0, 3) =
                    (displacement / displacement.norm()).transpose();
                r = Eigen::VectorXd::Constant(1, data - (displacement).norm());
                z_out = Eigen::VectorXd::Constant(1, data);

                return true;
              },
              "uwb");
        },
        [](const Sensor::DataPrefilter<double>::Stamped &) {});

    // ================= control =================
    Sensor::SensorTable::bind<V4>(
        "control",
        [this](const V4 &data, const Eigen::MatrixXd &,
               SteadyClock::time_point t) {
          queue_task(t, [this, data, t]() {
            this->control_snapshot = {t, data, true};
          });

          if (!accel_snapshot.fresh || !is_finished_orienting())
            return;

          const V3 gyro_at_t = gyro_snapshot.z.head<3>();
          const V3 gyro_bias_at_t = gyro_bias();

          const double wr_v = data(0);
          const double wl_v = data(1);
          const Eigen::Matrix3d R_B_I_v =
              rel_orientation().normalized().toRotationMatrix().transpose();
          // wheel-speed measures forward body velocity. residual = wheel-model
          // forward speed - INS body-forward speed. this is where SLIP lives:
          // if the wheels say we're going faster than the INS integrated, the
          // gap is slip and it pushes mu down. couples velocity (bounds INS
          // drift) and mu_c (forward slip) in one 1-D update.
          const Eigen::MatrixXd R_wv = Eigen::MatrixXd::Constant(1, 1, 4e-2);

          queue_correction(
              t, R_wv,
              [this, wr_v, wl_v, R_B_I_v](Eigen::MatrixXd &H, Eigen::VectorXd &r,
                                          Eigen::VectorXd &z_out) {
                const double v_fwd_ins = (R_B_I_v * velocity()).x();
                const double v_fwd_wheel =
                    GEOM.rw / 2 * (mu_r() * wr_v + mu_l() * wl_v);

                H = Eigen::MatrixXd::Zero(1, P.cols());
                H.block<1, 3>(0, 6) = R_B_I_v.row(0); // d(v_fwd_ins)/dv
                H(0, 9) = -GEOM.rw / 2 * (wr_v + wl_v); // -d(wheel)/dmu_c
                H(0, 10) = -GEOM.rw / 2 * (wr_v - wl_v); // -d(wheel)/dmu_d

                r = Eigen::VectorXd::Constant(1, v_fwd_wheel - v_fwd_ins);
                z_out = Eigen::VectorXd::Constant(1, v_fwd_wheel);
                return true;
              },
              "wheel_v");

          // ---- gyro yaw-rate vs wheel-differential: the strong mu_d observer
          // Wheel model predicts yaw rate psi_dot = (rw/B)(mu_l*wl - mu_r*wr).
          // Gyro measures body yaw rate (bias-removed, rotated into nav, z).
          // d(psi_dot)/d(mu_c) = (rw/B)(wl - wr),
          // d(psi_dot)/d(mu_d) = -(rw/B)(wl + wr)  <- scales with TOTAL wheel
          // speed, so mu_d is observable on ANY turn/curved path, not just
          // differential acceleration (which is what the accel path needs).
          const double wr_at_t = data(0);
          const double wl_at_t = data(1);
          const Eigen::Matrix3d R_I_B_at_t =
              rel_orientation().normalized().toRotationMatrix();
          const Eigen::MatrixXd R_yaw = Eigen::MatrixXd::Constant(1, 1, 1e-3);

          queue_correction(
              t, R_yaw,
              [this, wr_at_t, wl_at_t, gyro_at_t, gyro_bias_at_t,
               R_I_B_at_t](Eigen::MatrixXd &H, Eigen::VectorXd &r,
                           Eigen::VectorXd &z_out) {
                const double psi_dot_wheel =
                    GEOM.rw / GEOM.B * (mu_l() * wl_at_t - mu_r() * wr_at_t);

                const V3 omega_nav = R_I_B_at_t * (gyro_at_t - gyro_bias_at_t);
                const double omega_z = omega_nav.z();

                H = Eigen::MatrixXd::Zero(1, P.cols());
                H(0, 9) = GEOM.rw / GEOM.B * (wl_at_t - wr_at_t);
                H(0, 10) = -GEOM.rw / GEOM.B * (wl_at_t + wr_at_t);

                r = Eigen::VectorXd::Constant(1, psi_dot_wheel - omega_z);
                z_out = Eigen::VectorXd::Constant(1, omega_z);
                return true;
              },
              "gyro_mu_d");
        },
        [](const Sensor::DataPrefilter<V4>::Stamped &) {});

    // ================= gyro =================
    Sensor::SensorTable::bind<V3>(
        "gyro",
        [this](const V3 &z, const Eigen::MatrixXd &,
               SteadyClock::time_point t) {
          // ---- MEKF gyro handler ----
          queue_task(t, [this, z, t]() { this->gyro_snapshot = {t, z, true}; });
        },
        [this](const Sensor::DataPrefilter<V3>::Stamped &sample) {
          // plain constant offset from the startup mean — not estimated online
          bg0 = sample.mean;
        });
  }
};
