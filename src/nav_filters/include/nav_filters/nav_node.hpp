#ifndef NAV_NODE_HPP
#define NAV_NODE_HPP

#include "../../../utils.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>

#include <rclcpp/node.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>

#include <nav_filters/ekf.hpp>
#include <nav_filters/mekf.hpp>
using SteadyClock = std::chrono::steady_clock;

namespace nav {
static Eigen::Vector3d quat_to_rpy(const Eigen::Quaterniond &qn) {
  Eigen::Matrix3d R     = qn.normalized().toRotationMatrix();
  double          roll  = std::atan2(R(2, 1), R(2, 2));
  double          s     = -R(2, 0);
  double          pitch = (std::abs(s) >= 1.0) ? std::copysign(M_PI / 2, s) : std::asin(s);
  double          yaw   = std::atan2(R(1, 0), R(0, 0));
  return {roll, pitch, yaw};
}

class NavigationNode : public rclcpp::Node {
private:
  /// Statistics Generator for the incoming imu data.
  FilteredSampleProducer<Eigen::Vector3d> gyros{Eigen::Vector3d::Zero(), 1.0};
  FilteredSampleProducer<Eigen::Vector3d> accel{Eigen::Vector3d::Zero(), 1.0};
  FilteredSampleProducer<Eigen::Vector3d> magnm{Eigen::Vector3d::Zero(), 1.0};

  // initial values
  double             init_x, init_y, init_z;
  int                min_imu_samples, min_uwb_samples, min_mekf_samples;
  Eigen::Quaterniond init_orientation;

  // mekf values
  std::unique_ptr<MEKF> mekf;
  bool                  initialized     = false;
  bool                  orientation_set = false;

  Logger csv_log{"mekf_log.csv", std::chrono::milliseconds(50)};

  rclcpp::Subscription<sensor_msgs::msg::MagneticField>::SharedPtr mag_sub;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr           imu_sub;
  rclcpp::TimerBase::SharedPtr                                     ekf_timer;

  void mag_cb(const sensor_msgs::msg::MagneticField::SharedPtr m) {
    Eigen::Vector3d mag_vec{m->magnetic_field.x, m->magnetic_field.y, m->magnetic_field.z};
    if (!this->initialized) {
      magnm.put(mag_vec);
      initialize_from_profile();
    } else {
      this->mekf->put_magnm(mag_vec);
      this->mekf->step(SteadyClock::now());
      if (!this->orientation_set)
        initialize_from_profile();
    }
  }

  void imu_cb(const sensor_msgs::msg::Imu::SharedPtr m) {
    Eigen::Vector3d w_vec{m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z};
    Eigen::Vector3d a_vec{m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z};
    if (!this->initialized) {
      this->accel.put(a_vec);
      this->gyros.put(w_vec);
      initialize_from_profile();
    } else {
      this->mekf->put_accel(a_vec);
      this->mekf->put_gyros(w_vec);
      this->mekf->step(SteadyClock::now());
      if (!this->orientation_set)
        initialize_from_profile();
    }
  }

  void initialize_from_profile() {
    // Phase 1: collecting the stationary profile -> nothing built yet, just wait.
    if (!this->initialized) {
      if (this->gyros.get_count() < static_cast<size_t>(this->min_imu_samples)) {
        std::cout << string_format("\r\033[2KCounts: gyro=%zu accel=%zu mag=%zu / target=%d", this->gyros.get_count(),
                                   this->accel.get_count(), this->magnm.get_count(), this->min_imu_samples)
                  << std::flush;
        return;
      }

      Eigen::Quaterniond q0         = Eigen::Quaterniond::FromTwoVectors(this->accel.get_mean().normalized(), Eigen::Vector3d::UnitZ());
      Eigen::Vector3d    gyro_bias  = this->gyros.get_mean();
      Eigen::Vector3d    magnm_bias = Eigen::Vector3d::Zero();
      Eigen::Vector3d    mean_north = q0.toRotationMatrix() * this->magnm.get_mean();

      Eigen::VectorXd x0(10);
      x0 << q0.w(), q0.x(), q0.y(), q0.z(), gyro_bias.x(), gyro_bias.y(), gyro_bias.z(), magnm_bias.x(), magnm_bias.y(), magnm_bias.z();

      Eigen::Matrix<double, 9, 9> P0 = Eigen::Matrix<double, 9, 9>::Zero();
      const double                n  = static_cast<double>(this->gyros.get_count());

      P0.diagonal().segment<3>(0).setConstant(std::pow(0.5 * M_PI / 180.0, 2));
      P0.diagonal().segment<3>(3) = this->gyros.get_variance();
      P0.diagonal().segment<3>(6) = this->magnm.get_variance();

      Eigen::Vector3d gyro_var  = this->gyros.get_variance();
      Eigen::Vector3d accel_var = this->accel.get_variance();
      Eigen::Vector3d magnm_var = this->magnm.get_variance();

      // double Q_MULT = 10.0;
      double R_MULT = 100.0;

      // Eigen::Matrix3d Q_gyro       = (gyro_var * Q_MULT).asDiagonal();
      Eigen::Matrix3d Q_gyro       = Eigen::Matrix3d::Identity() * 1e-5;
      Eigen::Matrix3d Q_bias_gyro  = Eigen::Matrix3d::Identity() * 1e-15;
      Eigen::Matrix3d Q_bias_magnm = Eigen::Matrix3d::Identity() * 1e-15;

      Eigen::Matrix3d R_accel = (accel_var * R_MULT).asDiagonal();
      Eigen::Matrix3d R_magnm = (magnm_var * R_MULT).asDiagonal();

      this->mekf =
          std::make_unique<MEKF>(x0, P0, Q_gyro, Q_bias_gyro, Q_bias_magnm, R_accel, R_magnm, // Pass the newly constructed Matrix3d's here
                                 SteadyClock::now(), mean_north, this->gyros, this->accel, this->magnm);
      this->initialized = true;
      return;
    }

    // Phase 2: MEKF exists and the callbacks are stepping it -> let bias converge.
    if (this->mekf->num_steps() < static_cast<uint32_t>(this->min_mekf_samples)) {
      this->init_orientation    = this->mekf->orientation();
      const Eigen::Vector3d rpy = quat_to_rpy(this->init_orientation) * 180.0 / M_PI;
      std::cout << string_format("\r\033[2KInitial Orientation: roll=%.2f pitch=%.2f yaw=%.2f / steps=%u/%d", rpy.x(), rpy.y(), rpy.z(),
                                 this->mekf->num_steps(), this->min_mekf_samples)
                << std::flush;
      return;
    }

    // Phase 3: settled -> lock the reference orientation, start diagnostics.
    this->orientation_set               = true;
    this->mekf->should_print_diagnostic = true;
  }

  void log_func() {
    if (!this->initialized || !this->mekf)
      return;

    const Eigen::Quaterniond q     = this->mekf->orientation();
    Eigen::Quaterniond       q_rel = this->init_orientation.conjugate() * q;
    q_rel.normalize();

    const Eigen::Vector3d rpy = quat_to_rpy(q_rel); // rad (×180/M_PI for deg)

    const Eigen::Vector3d bg = this->mekf->gyro_bias();
    const Eigen::Vector3d w  = this->gyros.peek();     // raw gyro
    const Eigen::Vector3d wu = w - bg;                 // bias-corrected
    const Eigen::Vector3d wm = this->gyros.get_mean(); // EMA mean

    const double t = std::chrono::duration<double>(SteadyClock::now().time_since_epoch()).count();

    csv_log.log(string_format("%.6f,%.4f,"
                              "%.6f,%.6f,%.6f,"   // roll pitch yaw
                              "%.6e,%.6e,%.6e,"   // w raw   (gx..)
                              "%.6e,%.6e,%.6e,"   // w - bg  (gunb..)
                              "%.6e,%.6e,%.6e,"   // bg      (bg..)
                              "%.6e,%.6e,%.6e\n", // w mean  (gm..)
                              t, this->mekf->last_nis(), rpy.x(), rpy.y(), rpy.z(), w.x(), w.y(), w.z(), wu.x(), wu.y(), wu.z(), bg.x(),
                              bg.y(), bg.z(), wm.x(), wm.y(), wm.z()));
  }

public:
  NavigationNode() : Node("nav_node") {
    this->init_x           = declare_parameter<double>("init_x", 0.0);
    this->init_y           = declare_parameter<double>("init_y", 0.0);
    this->init_z           = declare_parameter<double>("init_z", 0.0);
    this->min_imu_samples  = declare_parameter<int>("min_imu_samples", 3000);
    this->min_uwb_samples  = declare_parameter<int>("min_uwb_samples", 3000);
    this->min_mekf_samples = declare_parameter<int>("min_mekf_samples", 3000);

    this->imu_sub =
        create_subscription<sensor_msgs::msg::Imu>("/imu/data_raw", 10, [this](sensor_msgs::msg::Imu::SharedPtr m) { imu_cb(m); });
    this->mag_sub = create_subscription<sensor_msgs::msg::MagneticField>(
        "/imu/mag", 10, [this](sensor_msgs::msg::MagneticField::SharedPtr m) { mag_cb(m); });

    // logger
    this->ekf_timer = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&NavigationNode::log_func, this));
    this->csv_log.log("time,nis,roll,pitch,yaw,gx,gy,gz,gunbx,gunby,gunbz,bgx,bgy,bgz,gmx,gmy,gmz\n");
  }
};
}; // namespace nav

#endif // NAV_NODE_HPP