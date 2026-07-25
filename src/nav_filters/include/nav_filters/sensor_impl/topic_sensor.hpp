#pragma once

#include "../sensor/sensor.hpp"

#include <rclcpp/rclcpp.hpp>

#include <string>

namespace Sensor {

template <typename Message, typename Tag, typename InputType,
          typename OutputType>
class TopicSensor : public ISensor<Tag, InputType, OutputType> {
protected:
  virtual InputType extract(typename Message::ConstSharedPtr message) const = 0;

public:
  TopicSensor(rclcpp::Node &node, const std::string &topic,
              const rclcpp::QoS &quality)
      : subscription(node.create_subscription<Message>(
            topic, quality, [this](typename Message::ConstSharedPtr message) {
              InputType input = this->extract(message);
              this->read(input);
            })) {}

private:
  typename rclcpp::Subscription<Message>::SharedPtr subscription;
};

} // namespace Sensor
