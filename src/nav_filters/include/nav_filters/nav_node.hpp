#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <memory>

#include <geometry_msgs/msg/quaternion.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>
#include <std_msgs/msg/float32.hpp>

#include "ISensor.hpp"
#include "ckf.hpp"
#include "mekf.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "trans_ekf.hpp"

using SteadyClock = std::chrono::steady_clock;

namespace nav {

class NavNode : public rclcpp::Node {
private:
  using V3 = Eigen::Vector3d;
  using V4 = Eigen::Vector4d;

  std::shared_ptr<MEKF> mekf_;
  std::unique_ptr<TRANSEKF> trans_ekf_;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<sensor_msgs::msg::MagneticField>::SharedPtr mag_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr uwb_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Quaternion>::SharedPtr cntrl_sub_;

  // wall timer for publisher...
  rclcpp::TimerBase::SharedPtr publisher_timer;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr estim_;

  Logger csv_log;
  Logger csv_log_trans;
  Logger csv_log_sensors;
  Logger csv_log_gains;
  Logger csv_log_corr;
  // full per-step Kalman pipeline trace (long format: one row per element)
  Logger csv_log_kf;

  inline static constexpr uint32_t min_gyro_samples = 3000;
  uint32_t n_gyro_samples = 0;

  inline static constexpr uint32_t min_accel_samples = 3000;
  uint32_t n_accel_samples = 0;

  inline static constexpr uint32_t min_magnm_samples = 3000;
  uint32_t n_magnm_samples = 0;

  inline static constexpr uint32_t min_uwb_samples = 1000;
  uint32_t n_uwb_samples = 0;

  // initialization...
  Eigen::Vector3d pos0 = Eigen::Vector3d{12.0, 0.0, -36.5} * 0.0254;

public:
  NavNode()
      : Node("nav_node"),
        csv_log("sparxe_mekf_test1__1_1.csv", std::chrono::milliseconds(25)),
        csv_log_trans("sparxe_ekf_test1__1_1.csv",
                      std::chrono::milliseconds(25)),
        csv_log_sensors("sparxe_sensors_raw.csv", std::chrono::milliseconds(0)),
        csv_log_gains("sparxe_gains.csv", std::chrono::milliseconds(25)),
        csv_log_corr("sparxe_corrections.csv", std::chrono::milliseconds(25)),
        csv_log_kf("sparxe_kf_trace.csv", std::chrono::milliseconds(0)) {

    Sensor::SensorTable::register_sensor<V3>(V3::Zero(), min_gyro_samples,
                                             "gyro");
    Sensor::SensorTable::register_sensor<V3>(V3::Zero(), min_accel_samples,
                                             "accel");
    Sensor::SensorTable::register_sensor<V3>(V3::Zero(), min_magnm_samples,
                                             "magnm");

    Sensor::SensorTable::register_sensor<double>(0.0, min_uwb_samples, "uwb");
    Sensor::SensorTable::register_sensor<V4>(V4::Zero(), 0, "control");

    csv_log.log("time,nis,roll,pitch,yaw,gx,gy,gz,"
                "gunbx,gunby,gunbz,bgx,bgy,bgz,gmx,gmy,gmz,"
                "cthx,cthy,cthz\n");

    csv_log_sensors.log("time,type,v0,v1,v2,v3\n");

    // q0-frame strapdown states + the adaptive wheel efficiencies (mu) and
    // forward accel bias.
    csv_log_trans.log("time,nis,"
                      "px,py,pz,"
                      "vx,vy,vz,"
                      "ax_raw,ay_raw,az_raw,"
                      "pxx,pyy,pxy,"
                      "bax,mur,mul,"
                      "pzz,pvx,pvy,pvz,pmr,pml\n");

    // per-state |K| of the last applied correction, per sensor, plus the
    // last innovation (pre-gate, so rejected updates show too).
    // trans stochastic states: px,py,pz,mur,mul (derived velocity slots log as
    // zero) ; mekf states: thx,thy,thz
    csv_log_gains.log("time,"
                      "uwb_px,uwb_py,uwb_pz,uwb_vx,uwb_vy,uwb_vz,uwb_mr,uwb_ml,"
                      "acc_px,acc_py,acc_pz,acc_vx,acc_vy,acc_vz,acc_mr,acc_ml,"
                      "macc_x,macc_y,macc_z,"
                      "mmag_x,mmag_y,mmag_z,"
                      "mctl_x,mctl_y,mctl_z,"
                      "uwb_r,acc_rx,acc_ry,acc_rz,"
                      "macc_rx,macc_ry,macc_rz,"
                      "mmag_rx,mmag_ry,mmag_rz,mctl_r\n");

    // signed per-state correction dx = K*d_r actually applied (keeps sign).
    // trans stochastic states: px,py,pz,mur,mul (derived velocity slots log as
    // zero) ; mekf attitude: thx,thy,thz
    csv_log_corr.log("time,"
                     "uwb_px,uwb_py,uwb_pz,uwb_vx,uwb_vy,uwb_vz,uwb_mr,uwb_ml,"
                     "acc_px,acc_py,acc_pz,acc_vx,acc_vy,acc_vz,acc_mr,acc_ml,"
                     "macc_x,macc_y,macc_z,"
                     "mmag_x,mmag_y,mmag_z,"
                     "mctl_x,mctl_y,mctl_z\n");

    // full per-step Kalman trace (long format)
    csv_log_kf.log("time,filter,sensor,step,applied,nis,kind,idx,value\n");

    const auto t0 = SteadyClock::now();
    mekf_ = std::make_shared<MEKF>(t0);
    trans_ekf_ = std::make_unique<TRANSEKF>(t0, mekf_, pos0);

    // stream every correction step of both filters to the trace CSV
    mekf_->trace_sink = [this](const ckf::KFTrace &tr) {
      log_kf_trace("mekf", tr);
    };
    trans_ekf_->trace_sink = [this](const ckf::KFTrace &tr) {
      log_kf_trace("trans", tr);
    };

    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data_raw", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::Imu::SharedPtr m) { imu_cb(m); });

    mag_sub_ = create_subscription<sensor_msgs::msg::MagneticField>(
        "/imu/mag", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::MagneticField::SharedPtr m) { mag_cb(m); });

    uwb_sub_ = create_subscription<std_msgs::msg::Float32>(
        "/uwb", rclcpp::SensorDataQoS(),
        [this](std_msgs::msg::Float32::SharedPtr m) { uwb_cb(m); });

    cntrl_sub_ = create_subscription<geometry_msgs::msg::Quaternion>(
        "/smc/control", rclcpp::SensorDataQoS(),
        [this](geometry_msgs::msg::Quaternion::SharedPtr m) { control_cb(m); });

    using namespace std::chrono_literals;
    publisher_timer = create_wall_timer(10ms, [this]() { publisher_cb(); });
    estim_ = create_publisher<nav_msgs::msg::Odometry>("/trans_est", 10);
  }

private:
  // one CSV row per element of every captured vector (long format)
  void log_kf_trace(const char *filter, const ckf::KFTrace &tr) {
    const double t = tr.t_s;
    std::string rows;
    const auto emit = [&](const char *kind, const Eigen::VectorXd &v) {
      for (int i = 0; i < v.size(); ++i)
        rows += string_format("%.9f,%s,%s,%u,%d,%.9e,%s,%d,%.9e\n", t, filter,
                              tr.sensor.c_str(), tr.step, tr.applied ? 1 : 0,
                              tr.nis, kind, i, v(i));
    };
    emit("z", tr.z);
    emit("Hx", tr.Hx);
    emit("r", tr.r);
    emit("S", tr.S_diag);
    emit("R", tr.R_diag);
    emit("dx", tr.dx);
    emit("P", tr.P_diag);
    emit("x_pre", tr.x_pre);
    emit("x_post", tr.x_post);
    // One synchronized write/flush per correction instead of one per element.
    csv_log_kf.log(rows);
  }

  void publisher_cb() {
    auto attitude = mekf_->get_curr_state();
    auto posvel = trans_ekf_->get_curr_state();

    nav_msgs::msg::Odometry msg;

    msg.pose.pose.position.x = posvel(0);
    msg.pose.pose.position.y = posvel(1);
    msg.pose.pose.position.z = posvel(2);

    // rel (q0-frame) orientation: same frame the trans EKF positions and the
    // wheel-odometry yaw live in
    auto o = mekf_->rel_orientation();

    msg.pose.pose.orientation.w = o.w();
    msg.pose.pose.orientation.x = o.x();
    msg.pose.pose.orientation.y = o.y();
    msg.pose.pose.orientation.z = o.z();

    estim_->publish(msg);

    // full trans state on one line, ~1 Hz
    static int tick = 0;
    if (++tick % 100 == 0) {
      const Eigen::VectorXd x = trans_ekf_->get_curr_state();
      const Eigen::Matrix3d R = o.normalized().toRotationMatrix();
      const double roll = std::atan2(R(2, 1), R(2, 2)) * 180.0 / M_PI;
      const double pitch =
          std::asin(std::clamp(-R(2, 0), -1.0, 1.0)) * 180.0 / M_PI;
      const double yaw = std::atan2(R(1, 0), R(0, 0)) * 180.0 / M_PI;
      std::cout << string_format("p[% .2f % .2f % .2f] v[% .2f % .2f % .2f] "
                                 "mu[% .2f % .2f] rpy[% .1f % .1f % .1f]deg\n",
                                 x(0), x(1), x(2), x(3), x(4), x(5), x(6), x(7),
                                 roll, pitch, yaw)
                << std::flush;
    }
  }

  void control_cb(const geometry_msgs::msg::Quaternion::SharedPtr m) {
    // [wr, wl, dwr, dwl]
    const V4 control(m->w, m->x, m->y, m->z);
    const auto now = SteadyClock::now();
    Sensor::SensorTable::put<V4>("control", control, now);
    // mekf_->suppress_magnm = (std::abs(m->w) > 0.05 || std::abs(m->x) > 0.05);

    const double t_s =
        std::chrono::duration<double>(now.time_since_epoch()).count();
    csv_log_sensors.log(string_format("%.9f,control,%.9f,%.9f,%.9f,%.9f\n", t_s,
                                      control(0), control(1), control(2),
                                      control(3)));
    trans_ekf_->tick(now);
  }

  void uwb_cb(const std_msgs::msg::Float32::SharedPtr m) {
    const double range = static_cast<double>(m->data);
    const auto now = SteadyClock::now();
    Sensor::SensorTable::put<double>("uwb", range, now);
    ++n_uwb_samples;

    const double t_s =
        std::chrono::duration<double>(now.time_since_epoch()).count();
    csv_log_sensors.log(string_format("%.9f,uwb,%.9f,,,\n", t_s, range));
    trans_ekf_->tick(now);
  }

  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr m) {
    const V3 w(m->angular_velocity.x, m->angular_velocity.y,
               m->angular_velocity.z);

    const V3 a(m->linear_acceleration.x, m->linear_acceleration.y,
               m->linear_acceleration.z);

    const auto now = SteadyClock::now();

    ++n_gyro_samples;
    ++n_accel_samples;

    Sensor::SensorTable::put<V3>("gyro", w, now);
    Sensor::SensorTable::put<V3>("accel", a, now);

    {
      const double t_s =
          std::chrono::duration<double>(now.time_since_epoch()).count();
      csv_log_sensors.log(string_format("%.9f,gyro,%.9f,%.9f,%.9f,\n", t_s,
                                        w.x(), w.y(), w.z()));
      csv_log_sensors.log(string_format("%.9f,accel,%.9f,%.9f,%.9f,\n", t_s,
                                        a.x(), a.y(), a.z()));
    }

    mekf_->tick(now);

    if (n_accel_samples >= min_accel_samples &&
        n_gyro_samples >= min_gyro_samples &&
        n_magnm_samples >= min_magnm_samples) {
      const Eigen::Quaterniond q = mekf_->orientation().normalized();

      const Eigen::Matrix3d R_I_B = q.toRotationMatrix();

      const double roll = std::atan2(R_I_B(2, 1), R_I_B(2, 2)) * 180.0 / M_PI;
      const double pitch =
          std::asin(std::clamp(-R_I_B(2, 0), -1.0, 1.0)) * 180.0 / M_PI;
      const double yaw = std::atan2(R_I_B(1, 0), R_I_B(0, 0)) * 180.0 / M_PI;

      const V3 bg = mekf_->gyro_bias();

      const V3 gyro_unbiased = w - bg;

      V3 gyro_mean = V3::Zero();

      Sensor::SensorTable::get_mean<V3>("gyro", gyro_mean);

      double time =
          std::chrono::duration<double>(now.time_since_epoch()).count();

      // attitude error-state (dtheta) covariance diagonal, rad^2
      const auto &Pm = mekf_->covariance();

      csv_log.log(string_format("%.9f,%.9f,"
                                "%.9f,%.9f,%.9f,"
                                "%.9f,%.9f,%.9f,"
                                "%.9f,%.9f,%.9f,"
                                "%.9f,%.9f,%.9f,"
                                "%.9f,%.9f,%.9f,"
                                "%.9e,%.9e,%.9e\n",
                                time, mekf_->last_nis(), roll, pitch, yaw,
                                w.x(), w.y(), w.z(), gyro_unbiased.x(),
                                gyro_unbiased.y(), gyro_unbiased.z(), bg.x(),
                                bg.y(), bg.z(), gyro_mean.x(), gyro_mean.y(),
                                gyro_mean.z(), Pm(0, 0), Pm(1, 1), Pm(2, 2)));
    } else
      std::cout << string_format("\raccel: %d, gyro: %d, magnm: %d, uwb: %d",
                                 n_accel_samples, n_gyro_samples,
                                 n_magnm_samples, n_uwb_samples)
                << std::flush;
    // std::cout << "ticking tekf" << std::endl;
    trans_ekf_->tick(now);
    // std::cout << "done" << std::endl;
    // std::cout << "-------------" << std::endl;

    if (n_accel_samples >= min_accel_samples &&
        n_gyro_samples >= min_gyro_samples &&
        n_magnm_samples >= min_magnm_samples &&
        n_uwb_samples >= min_uwb_samples) {
      const V3 p = trans_ekf_->get_position();
      const V3 v = trans_ekf_->get_velocity();

      const double time =
          std::chrono::duration<double>(now.time_since_epoch()).count();

      const auto &cov = trans_ekf_->covariance();
      csv_log_trans.log(string_format(
          "%.9f,%.9f,"
          "%.9f,%.9f,%.9f,"
          "%.9f,%.9f,%.9f,"
          "%.9f,%.9f,%.9f,"
          "%.9e,%.9e,%.9e,"
          "%.9f,%.9f,%.9f,"
          "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
          time, trans_ekf_->last_nis(), p.x(), p.y(), p.z(), v.x(), v.y(),
          v.z(), a.x(), a.y(), a.z(), cov(0, 0), cov(1, 1), cov(0, 1), 0.0,
          trans_ekf_->get_mu_r(), trans_ekf_->get_mu_l(), cov(2, 2), 0.0, 0.0,
          0.0, cov(3, 3), cov(4, 4)));
    }

    if (n_accel_samples >= min_accel_samples &&
        n_gyro_samples >= min_gyro_samples &&
        n_magnm_samples >= min_magnm_samples &&
        n_uwb_samples >= min_uwb_samples) {
      const double time =
          std::chrono::duration<double>(now.time_since_epoch()).count();

      const auto expand_trans = [](const Eigen::VectorXd &in) {
        Eigen::VectorXd out = Eigen::VectorXd::Zero(8);
        out.head<3>() = in.head<3>();
        out.segment<2>(6) = in.segment<2>(3);
        return out;
      };
      const Eigen::VectorXd gu = expand_trans(trans_ekf_->last_gain("uwb", 5));
      const Eigen::VectorXd ga =
          expand_trans(trans_ekf_->last_gain("accel", 5));
      const Eigen::VectorXd ma = mekf_->last_gain("accel", 3);
      const Eigen::VectorXd mm = mekf_->last_gain("magnm", 3);
      const Eigen::VectorXd mc = mekf_->last_gain("control", 3);

      const Eigen::VectorXd ru = trans_ekf_->last_innovation("uwb", 1);
      const Eigen::VectorXd ra = trans_ekf_->last_innovation("accel", 3);
      const Eigen::VectorXd rma = mekf_->last_innovation("accel", 3);
      const Eigen::VectorXd rmm = mekf_->last_innovation("magnm", 3);
      const Eigen::VectorXd rmc = mekf_->last_innovation("control", 1);

      csv_log_gains.log(string_format(
          "%.9f,"
          "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,%.9e\n",
          time, gu(0), gu(1), gu(2), gu(3), gu(4), gu(5), gu(6), gu(7), ga(0),
          ga(1), ga(2), ga(3), ga(4), ga(5), ga(6), ga(7), ma(0), ma(1), ma(2),
          mm(0), mm(1), mm(2), mc(0), mc(1), mc(2), ru(0), ra(0), ra(1), ra(2),
          rma(0), rma(1), rma(2), rmm(0), rmm(1), rmm(2), rmc(0)));

      // signed corrections dx = K*d_r per state (direction, not magnitude)
      const Eigen::VectorXd cu =
          expand_trans(trans_ekf_->last_correction("uwb", 5));
      const Eigen::VectorXd ca =
          expand_trans(trans_ekf_->last_correction("accel", 5));
      const Eigen::VectorXd cma = mekf_->last_correction("accel", 3);
      const Eigen::VectorXd cmm = mekf_->last_correction("magnm", 3);
      const Eigen::VectorXd cmc = mekf_->last_correction("control", 3);

      csv_log_corr.log(string_format(
          "%.9f,"
          "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e,"
          "%.9e,%.9e,%.9e\n",
          time, cu(0), cu(1), cu(2), cu(3), cu(4), cu(5), cu(6), cu(7), ca(0),
          ca(1), ca(2), ca(3), ca(4), ca(5), ca(6), ca(7), cma(0), cma(1),
          cma(2), cmm(0), cmm(1), cmm(2), cmc(0), cmc(1), cmc(2)));
    }
  }

  void mag_cb(const sensor_msgs::msg::MagneticField::SharedPtr m) {
    const V3 magnetic_field(m->magnetic_field.x, m->magnetic_field.y,
                            m->magnetic_field.z);
    const auto now = SteadyClock::now();

    Sensor::SensorTable::put<V3>("magnm", magnetic_field, now);
    ++n_magnm_samples;

    const double t_s =
        std::chrono::duration<double>(now.time_since_epoch()).count();
    csv_log_sensors.log(string_format("%.9f,magnm,%.9f,%.9f,%.9f,\n", t_s,
                                      magnetic_field.x(), magnetic_field.y(),
                                      magnetic_field.z()));
    mekf_->tick(now);
  }
};

} // namespace nav
