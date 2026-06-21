cmake_minimum_required(VERSION 3.20)
project(lidar_processor)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- 1. FIND DEPENDENCIES ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

#idk what this shit is
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Obstacle.msg"
  "msg/Env.msg"
)

# Added PCL specific dependencies
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)

# --- 2. ADD EXECUTABLE ---
# Tells CMake to build an executable named 'lidar_obstacle_processor' from main.cpp
# Note: If main.cpp is not in a 'src' folder, change this to just main.cpp
add_executable(lidar_obstacle_processor src/main.cpp)

# --- 3. LINK PCL ---
# PCL requires standard CMake linking macros
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")

target_include_directories(lidar_obstacle_processor PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(lidar_obstacle_processor ${PCL_LIBRARIES})
target_compile_definitions(lidar_obstacle_processor PUBLIC ${PCL_DEFINITIONS})
target_link_libraries(lidar_obstacle_processor "${cpp_typesupport_target}")

# --- 4. LINK ROS2 DEPENDENCIES ---
# Ament macro to link standard ROS2 and Eigen libraries
target_link_libraries(lidar_obstacle_processor
  rclcpp::rclcpp
  ${std_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${message_filters_TARGETS}
  Eigen3::Eigen
  ${pcl_conversions_TARGETS}
)

# --- 5. INSTALLATION ---
# Installs the executable so you can use 'ros2 run lidar_processor lidar_obstacle_processor'
install(TARGETS lidar_obstacle_processor
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY msg
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()