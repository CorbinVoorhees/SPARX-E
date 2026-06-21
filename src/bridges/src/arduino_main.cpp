#include <bridges/arduino_bridge.hpp>

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArduinoBridge>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}