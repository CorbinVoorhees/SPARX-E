

// Responsible for spinning up the uwbBridge node
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<UWBBridge>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
