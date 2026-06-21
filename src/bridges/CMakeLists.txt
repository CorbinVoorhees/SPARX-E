cmake_minimum_required(VERSION 3.20)
project(bridges)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rcpputils REQUIRED)

add_executable(arduino_bridge_node src/arduino_main.cpp)
target_include_directories(arduino_bridge_node PRIVATE include)
target_link_libraries(arduino_bridge_node PUBLIC
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  rcpputils::rcpputils
)

add_executable(uwb_bridge_node src/uwb_main.cpp)
target_include_directories(uwb_bridge_node PRIVATE include)
target_link_libraries(uwb_bridge_node PUBLIC
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  rcpputils::rcpputils
)

install(TARGETS
  arduino_bridge_node
  uwb_bridge_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()