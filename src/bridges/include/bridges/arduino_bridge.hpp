#ifndef ARDUINO_BRIDGE
#define ARDUINO_BRIDGE

#include "bridge.hpp"

/**
 * @class ArduinoBridge
 * @brief Bridges Arduino serial data into ROS 2 topics.
 *
 * @author Corbin Voorhees
 * @author Krish Sridhar
 * @author Adam Ben Youssef
 * @date May 28, 2026
 */
class ArduinoBridge : public BridgeBase
{
private:
    // Initialize shared pointer of subscription object string message (publisher: Arduino serial monitor)
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr arduino_subscription;
public:
    /**
     * @brief Constructor for the arduino bridge node: "arduino_bridge_node". No idea how long the char string is.
     */
    ArduinoBridge() : BridgeBase("arduino_bridge_node", "/dev/ttyACM0", 115200, 1000)
    {
        // Subscription that pulls any message pushed into the arduino command topic
        arduino_subscription = this->create_subscription<std_msgs::msg::String>(
            "/arduino_cmd",
            10,
            std::bind(&BridgeBase::write_serial, this, std::placeholders::_1));
    }
};

#endif
