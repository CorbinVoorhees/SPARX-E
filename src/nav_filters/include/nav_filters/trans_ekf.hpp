#pragma once

#include "ISensor.hpp"
#include "ckf.hpp"
#include "mekf.hpp"
#include <Eigen/Dense>

using V3 = Eigen::Vector3d;
using V4 = Eigen::Vector4d;

// Adaptive-slip translational EKF (q0 frame), kinematic dynamics.
// x = [ px py pz  vx vy vz  mu_r  mu_l ]
class TRANSEKF : public ckf::CommonKF {

private:
  snapshot gyro_snapshot;
  snapshot control_snapshot;
  snapshot accel_snapshot;
  snapshot uwb_snapshot;
  std::shared_ptr<MEKF> mekf_ref;

  V3 grav_ref = grav_vec;

  auto position() { return x.segment<3>(0); }
  auto velocity() { return x.segment<3>(3); }
  double &mu_r() { return x(6); }
  double &mu_l() { return x(7); }

  void apply_error(const Eigen::VectorXd &err) override {
    position() += err.segment<3>(0);
    velocity() += err.segment<3>(3);
    mu_r() += err(6);
    mu_l() += err(7);
    mu_r() = std::max(mu_r(), 0.05);
    mu_l() = std::max(mu_l(), 0.05);
  }

  void predict(double dt) override {
    if (!mekf_ref->is_finished_orienting())
      return;

    const double wr = control_snapshot.z(0);
    const double wl = control_snapshot.z(1);

    const Eigen::Matrix3d R_I_B =
        mekf_ref->rel_orientation().normalized().toRotationMatrix();

    position() += velocity() * dt;
    velocity() =
        R_I_B * V3(GEOM.rw / 2 * (mu_r() * wr + mu_l() * wl), 0.0, 0.0);

    Eigen::Matrix<double, 8, 8> dfdx = Eigen::Matrix<double, 8, 8>::Zero();
    dfdx.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    dfdx.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    dfdx.block<3, 1>(3, 6) = R_I_B.col(0) * GEOM.rw / 2 * wr;
    dfdx.block<3, 1>(3, 7) = R_I_B.col(0) * GEOM.rw / 2 * wl;
    dfdx(6, 6) = 1.0;
    dfdx(7, 7) = 1.0;

    Eigen::MatrixXd Qd = Eigen::MatrixXd::Zero(8, 8);
    Qd.diagonal() << 1e-2, 1e-4, 1e-6, 1e-3, 1e-3, 1e-3, 1e-8, 1e-8;
    Qd *= dt;

    P = dfdx * P * dfdx.transpose() + Qd;
  }

  void print_diagnostic(const Eigen::MatrixXd &S, const Eigen::MatrixXd &,
                        const Eigen::MatrixXd &, const Eigen::VectorXd &d_r,
                        std::string name) override {
    if (!this->should_print_diagnostic)
      return;
    using clock = std::chrono::steady_clock;

    static auto last_print = clock::now();
    static bool first_print = true;
    const auto now = clock::now();

    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print)
            .count() < 1000)
      return;

    last_print = now;

    const int m = static_cast<int>(d_r.rows());

    std::array<double, 3> residual{};
    std::array<double, 3> normalized{};
    std::array<double, 3> s_diag{};

    for (int i = 0; i < std::min(m, 3); ++i) {
      residual[i] = d_r(i);
      s_diag[i] = S(i, i);
      if (s_diag[i] > 0.0)
        normalized[i] = residual[i] / std::sqrt(s_diag[i]);
    }

    const V3 p = position();
    const V3 v = velocity();

    const double wr = control_snapshot.fresh ? control_snapshot.z(0) : 0.0;
    const double wl = control_snapshot.fresh ? control_snapshot.z(1) : 0.0;

    const double uwb_meas = uwb_snapshot.fresh ? uwb_snapshot.z(0) : 0.0;
    const double uwb_pred = p.norm();
    const double uwb_resid = uwb_snapshot.fresh ? uwb_meas - uwb_pred : 0.0;
    const char *uwb_tag = uwb_snapshot.fresh ? "" : " [stale]";

    if (!first_print)
      std::cout << "\033[16A\033[J";
    first_print = false;

    std::cout
        << string_format(
               "===================== %s TRANSLATIONAL EKF (q0, kinematic, "
               "adaptive mu) =====================\n"
               "NIS          : %.6f (dim: %d)\n"
               "-----------------------------------------------------------\n"
               "position     : [% .6f, % .6f, % .6f] m\n"
               "velocity     : [% .6f, % .6f, % .6f] m/s\n"
               "-----------------------------------------------------------\n"
               "mu_r / mu_l  : % .4f / % .4f   (cmd wr/wl: % .3f / % .3f)\n"
               "wheel v(mu)  : % .4f m/s  (mu*rw*(wr+wl)/2)\n"
               "-----------------------------------------------------------\n"
               "UWB meas     : % .6f m%s\n"
               "UWB pred(|p|): % .6f m\n"
               "UWB resid    : % .6f m%s\n"
               "-----------------------------------------------------------\n"
               "residual     : [% .6f, % .6f, % .6f]\n"
               "normalized   : [% .4f, % .4f, % .4f]\n"
               "S            : [% .4e, % .4e, % .4e]\n"
               "===========================================================\n",

               name.c_str(), nis, m,

               p.x(), p.y(), p.z(), v.x(), v.y(), v.z(),

               x(6), x(7), wr, wl, GEOM.rw / 2 * (x(6) * wr + x(7) * wl),

               uwb_meas, uwb_tag, uwb_pred, uwb_resid, uwb_tag,

               residual[0], residual[1], residual[2], normalized[0],
               normalized[1], normalized[2], s_diag[0], s_diag[1], s_diag[2])
        << std::flush;
  }

public:
  V3 get_position() const { return x.segment<3>(0); }
  V3 get_velocity() const { return x.segment<3>(3); }
  double get_mu_r() const { return x(6); }
  double get_mu_l() const { return x(7); }
  bool should_print_diagnostic = true;

  TRANSEKF(SteadyClock::time_point t0, const std::shared_ptr<MEKF> &ref,
           Eigen::Vector3d pos0)
      : CommonKF(8, 8, t0), mekf_ref(ref) {

    control_snapshot.z = Eigen::Vector4d::Zero();
    control_snapshot.fresh = true;
    gyro_snapshot.z = V3::Zero();
    gyro_snapshot.fresh = true;

    position() = pos0;
    velocity() = Eigen::Vector3d::Zero();
    mu_r() = GEOM.mu_r;
    mu_l() = GEOM.mu_l;

    this->P.diagonal() << 1e-4, 1e-4, 1e-4, // p
        1e-3, 1e-3, 1e-3,                   // v
        1e-6, 1e-6;                         // mu (sigma ~0.05)

    Sensor::SensorTable::bind<V3>(
        "accel",
        [this](const V3 &data, const Eigen::MatrixXd &R,
               SteadyClock::time_point t) {
          queue_task(t,
                     [this, data, t]() { accel_snapshot = {t, data, true}; });

          const bool mekf_finished_at_t = mekf_ref->is_finished_orienting();
          const double dwr_at_t = control_snapshot.z(2);
          const double dwl_at_t = control_snapshot.z(3);
          const V3 gyro_at_t = gyro_snapshot.z.head<3>();
          const V3 gyro_bias_at_t = mekf_ref->gyro_bias();
          const V3 grav_at_t = grav_ref;
          const Eigen::Matrix3d R_B_I_at_t = mekf_ref->rel_orientation()
                                                 .normalized()
                                                 .toRotationMatrix()
                                                 .transpose();

          queue_correction(
              t, R,
              [this, data, mekf_finished_at_t, dwr_at_t, dwl_at_t, gyro_at_t,
               gyro_bias_at_t, grav_at_t,
               R_B_I_at_t](Eigen::MatrixXd &H, Eigen::VectorXd &r,
                           Eigen::VectorXd &z_out) {
                if (!mekf_finished_at_t)
                  return false;

                const V3 omega = gyro_at_t - gyro_bias_at_t;

                const V3 v_body = R_B_I_at_t * velocity();
                const V3 a_fwd(GEOM.rw / 2 *
                                   (mu_r() * dwr_at_t + mu_l() * dwl_at_t) /
                                   CTRL_PERIOD_S,
                               0, 0);
                const V3 h_pred =
                    R_B_I_at_t * grav_at_t + a_fwd + omega.cross(v_body);

                H = Eigen::MatrixXd::Zero(3, 8);
                H.block<3, 3>(0, 3) = skew(omega) * R_B_I_at_t;
                H.col(6) = V3::UnitX() * GEOM.rw / 2 * dwr_at_t / CTRL_PERIOD_S;
                H.col(7) = V3::UnitX() * GEOM.rw / 2 * dwl_at_t / CTRL_PERIOD_S;

                r = data - h_pred;
                z_out = data;
                return true;
              },
              "accel");
        },
        [this](const FilteredSampleProducer<V3> &prod) {
          const V3 g = prod.get_mean();
          queue_task(SteadyClock::now(), [this, g]() { this->grav_ref = g; });
        });

    Sensor::SensorTable::bind<double>(
        "uwb",
        [this](const double data, const Eigen::MatrixXd &R,
               SteadyClock::time_point t) {
          queue_task(t, [this, data, t]() {
            uwb_snapshot = {t, Eigen::VectorXd::Constant(1, data), true};
          });
          queue_correction(
              t, R,
              [this, data](Eigen::MatrixXd &H, Eigen::VectorXd &r,
                           Eigen::VectorXd &z_out) {
                const V3 displacement = position();
                H = Eigen::MatrixXd::Zero(1, 8);
                H.block<1, 3>(0, 0) =
                    (displacement / displacement.norm()).transpose();
                r = Eigen::VectorXd::Constant(1, data - (displacement).norm());
                z_out = Eigen::VectorXd::Constant(1, data);

                return true;
              },
              "uwb");
        },
        [](const FilteredSampleProducer<double> &) {});

    Sensor::SensorTable::bind<V4>(
        "control",
        [this](const V4 &data, const Eigen::MatrixXd &,
               SteadyClock::time_point t) {
          queue_task(t, [this, data, t]() {
            this->control_snapshot = {t, data, true};
          });
        },
        [](const FilteredSampleProducer<V4> &) {});

    Sensor::SensorTable::bind<V3>(
        "gyro",
        [this](const V3 &data, const Eigen::MatrixXd &,
               SteadyClock::time_point t) {
          queue_task(
              t, [this, data, t]() { this->gyro_snapshot = {t, data, true}; });
        },
        [](const FilteredSampleProducer<V3> &) {});
  }
};
