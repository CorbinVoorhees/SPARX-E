#pragma once

#include "chuudekf.hpp"
#include "sensor_impl/navigation_sensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace nav {

class NavNode final : public rclcpp::Node {
private:
  inline static constexpr std::size_t gyro_calibration_samples = 3000;
  inline static constexpr std::size_t accel_calibration_samples = 3000;
  inline static constexpr std::size_t magnetometer_calibration_samples = 3000;
  inline static constexpr std::size_t uwb_calibration_samples = 1000;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr state_publisher;
  rclcpp::TimerBase::SharedPtr filter_timer;
  std::size_t state_publish_count = 0;

  // Sensors are destroyed first so their worker and callbacks stop before the
  // filter they reference.
  std::unique_ptr<CHUUDEKF> filter;
  NavigationSensors sensors;

  // void bind_diagnostics() {
  //   sensors.bind<Sensor::UwbSensor>([this](const double range,
  //                                           const Eigen::MatrixXd &,
  //                                           Sensor::SteadyClock::time_point)
  //                                           {
  //     RCLCPP_INFO(get_logger(), "uwb: range=%.3f m", range);
  //   });
  // }

  void publish_state() {
    filter->tick(Sensor::SteadyClock::now());

    const Vector3 position = filter->get_position();
    const Vector3 velocity = filter->get_velocity();
    const Eigen::Quaterniond attitude = filter->rel_orientation().normalized();

    nav_msgs::msg::Odometry message;
    message.header.stamp = now();
    message.header.frame_id = "map";
    message.child_frame_id = "base_link";

    message.pose.pose.position.x = position.x();
    message.pose.pose.position.y = position.y();
    message.pose.pose.position.z = position.z();
    message.pose.pose.orientation.w = attitude.w();
    message.pose.pose.orientation.x = attitude.x();
    message.pose.pose.orientation.y = attitude.y();
    message.pose.pose.orientation.z = attitude.z();

    message.twist.twist.linear.x = velocity.x();
    message.twist.twist.linear.y = velocity.y();
    message.twist.twist.linear.z = velocity.z();
    state_publisher->publish(message);

    if (++state_publish_count % 50 != 0)
      return;

    const Eigen::VectorXd estimate = filter->get_curr_state();
    const Eigen::Matrix3d rotation = attitude.toRotationMatrix();
    constexpr double radians_to_degrees =
        180.0 / 3.141592653589793238462643383279502884;
    const double roll =
        std::atan2(rotation(2, 1), rotation(2, 2)) * radians_to_degrees;
    const double pitch =
        std::asin(std::clamp(-rotation(2, 0), -1.0, 1.0)) * radians_to_degrees;
    const double yaw =
        std::atan2(rotation(1, 0), rotation(0, 0)) * radians_to_degrees;

    RCLCPP_INFO(get_logger(),
                "estimate rpy_deg=[% .2f % .2f % .2f] "
                "p=[% .6f % .6f % .6f] v=[% .6f % .6f % .6f] "
                "mu_r=% .6f mu_l=% .6f",
                roll, pitch, yaw, estimate(4), estimate(5), estimate(6),
                estimate(7), estimate(8), estimate(9), filter->get_mu_r(),
                filter->get_mu_l());
  }

public:
  NavNode() : Node("nav_node") {
    // bind_diagnostics();

    const Eigen::Vector3d initial_position =
        Eigen::Vector3d(39.0, 0.0, -36.0) * 0.0254;
    filter = std::make_unique<CHUUDEKF>(sensors, Sensor::SteadyClock::now(),
                                        initial_position);

    sensors.register_sensor<GyroscopeSensor>(
        {Vector3::Zero(), gyro_calibration_samples, ""}, *this);
    sensors.register_sensor<AccelerometerSensor>({Vector3::Zero(),
                                                  accel_calibration_samples,
                                                  "accel_calibration.txt"},
                                                 *this);
    sensors.register_sensor<MagnetometerSensor>(
        {Vector3::Zero(), magnetometer_calibration_samples,
         "magnm_calibration.txt"},
        *this);
    sensors.register_sensor<ControlSensor>({Vector4::Zero(), 0, ""}, *this);

    Sensor::UwbConfig uwb_config;
    uwb_config.serial.device =
        declare_parameter<std::string>("uwb_device", uwb_config.serial.device);
    sensors.register_sensor<Sensor::UwbSensor>(
        {0.0, uwb_calibration_samples, ""}, std::move(uwb_config));

    state_publisher =
        create_publisher<nav_msgs::msg::Odometry>("/trans_est", 10);

    using namespace std::chrono_literals;
    filter_timer = create_wall_timer(20ms, [this] { publish_state(); });
  }
};

} // namespace nav
