
#include <motor_commander/smc_node.hpp>
#include <rclcpp/rclcpp.hpp>

// Responsible for spinning up the MotorCommander node
int main(int argc, char* argv[]) 
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SmcNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}