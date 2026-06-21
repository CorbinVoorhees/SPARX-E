#ifndef UWB_BRIDGE_HPP
#define UWB_BRIDGE_HPP

#include "bridge.hpp"
#include <rclcpp/logging.hpp>
#include <std_msgs/msg/float32.hpp>

#include <cstdio>
#include <string>

/**
 * @class UWBBridge
 * @brief Bridges UWB serial range data into ROS 2 topics.
 */
class UWBBridge : public BridgeBase {
private:
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr uwb_publisher;
  rclcpp::TimerBase::SharedPtr read_timer;

  /**
   * Gets the serial monitor data, parses it from str to float, publishes it.
   */
  void read_uwb() {
    auto serial_data = this->read_serial();

    if (serial_data.empty()) {
      RCLCPP_INFO(this->get_logger(), "No Value!");
      return;
    }

    try {
      std_msgs::msg::Float32 msg;
      
      // find "range_m"
      const std::string find_this = "range_m=";

      // split the string by spaces
      std::stringstream ss_tknzr(serial_data);

      std::string token;
      while(ss_tknzr >> token)
        if (token.find(find_this) != std::string::npos)
            break;

      msg.data = std::stof(token.substr(find_this.length()));
      uwb_publisher->publish(msg);
    } catch (const std::exception &e) {
    }
  }

public:
  /**
   * @brief Constructor for the UWB bridge node.
   */
  UWBBridge() : BridgeBase("uwb_bridge_node", "/dev/ttyACM1", 115200, 500) {
    uwb_publisher = this->create_publisher<std_msgs::msg::Float32>("/uwb", 10);

    using namespace std::chrono_literals;

    // WALL TIMER PERIOD MUST MATCH THE UWB RATE
    read_timer = this->create_wall_timer(20ms, std::bind(&UWBBridge::read_uwb, this));
  }
};

#endif