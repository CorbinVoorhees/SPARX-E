#include <nav_filters/nav_node.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<nav::NavNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}