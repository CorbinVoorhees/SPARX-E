#pragma once

#include "topic_sensor.hpp"
#include "uwb_sensor.hpp"

#include <Eigen/Core>
#include <geometry_msgs/msg/quaternion.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/magnetic_field.hpp>

namespace nav {

struct GyroscopeTag : Sensor::ISensorTag {};
struct AccelerometerTag : Sensor::ISensorTag {};
struct MagnetometerTag : Sensor::ISensorTag {};
struct ControlTag : Sensor::ISensorTag {};

using Vector3 = Eigen::Vector3d;
using Vector4 = Eigen::Vector4d;

class GyroscopeSensor final
    : public Sensor::TopicSensor<sensor_msgs::msg::Imu, GyroscopeTag, Vector3,
                                 std::monostate> {
protected:
  Vector3
  extract(sensor_msgs::msg::Imu::ConstSharedPtr message) const override {
    return Vector3(message->angular_velocity.x, message->angular_velocity.y,
                   message->angular_velocity.z);
  }

public:
  explicit GyroscopeSensor(rclcpp::Node &node)
      : TopicSensor(node, "/imu/data_raw", rclcpp::SensorDataQoS()) {}
};

class AccelerometerSensor final
    : public Sensor::TopicSensor<sensor_msgs::msg::Imu, AccelerometerTag,
                                 Vector3, std::monostate> {
protected:
  Vector3
  extract(sensor_msgs::msg::Imu::ConstSharedPtr message) const override {
    return Vector3(message->linear_acceleration.x,
                   message->linear_acceleration.y,
                   message->linear_acceleration.z);
  }

public:
  explicit AccelerometerSensor(rclcpp::Node &node)
      : TopicSensor(node, "/imu/data_raw", rclcpp::SensorDataQoS()) {}
};

class MagnetometerSensor final
    : public Sensor::TopicSensor<sensor_msgs::msg::MagneticField,
                                 MagnetometerTag, Vector3, std::monostate> {
protected:
  Vector3 extract(sensor_msgs::msg::MagneticField::ConstSharedPtr message)
      const override {
    return Vector3(message->magnetic_field.x, message->magnetic_field.y,
                   message->magnetic_field.z);
  }

public:
  explicit MagnetometerSensor(rclcpp::Node &node)
      : TopicSensor(node, "/imu/mag", rclcpp::SensorDataQoS()) {}
};

class ControlSensor final
    : public Sensor::TopicSensor<geometry_msgs::msg::Quaternion, ControlTag,
                                 Vector4, std::monostate> {
protected:
  Vector4
  extract(geometry_msgs::msg::Quaternion::ConstSharedPtr message) const override {
    return Vector4(message->w, message->x, message->y, message->z);
  }

public:
  explicit ControlSensor(rclcpp::Node &node)
      : TopicSensor(node, "/smc/control", rclcpp::SensorDataQoS()) {}
};

using NavigationSensors =
    Sensor::SensorTable<GyroscopeSensor, AccelerometerSensor,
                        MagnetometerSensor, ControlSensor, Sensor::UwbSensor>;

} // namespace nav
