#pragma once

#include <Eigen/Dense>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/node.hpp>
#include <std_msgs/msg/string.hpp>

#include "../../utils.h" // GEOM: rw, B, mu_r, mu_l

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>

using namespace std::chrono_literals;

class SmcNode : public rclcpp::Node {
private:
  // generate a publisher of data
  rclcpp::Publisher<geometry_msgs::msg::Quaternion>::SharedPtr
      control_publisher;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr nav_subscriber;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr autonomy_subscriber;

  rclcpp::TimerBase::SharedPtr control_publisher_timer;
  static Eigen::Matrix<double, 3, 2> b_jacobian(double r_w, double B,
                                                double mu_R, double mu_L,
                                                Eigen::Vector3d &orientation) {
    // hardcode the jacobian matrix to be returned
    Eigen::Matrix<double, 3, 2> dBdu;

    // set all values to zero (they're random)
    dBdu.setZero();

    // unpack all of the orientation angles
    double roll = orientation[0];  // rad
    double pitch = orientation[1]; // rad
    double yaw = orientation[2];   // rad

    // actually input values for jacobian matrix (derivative of X_dot WRT
    // control)
    dBdu << (r_w / 2) * mu_R * cos(pitch) * cos(yaw),
        (r_w / 2) * mu_L * cos(pitch) * cos(yaw),
        (r_w / 2) * mu_R * cos(pitch) * sin(yaw),
        (r_w / 2) * mu_L * cos(pitch) * sin(yaw),
        // z-down frame: wr > wl = CCW from above = negative yaw
        (-r_w / B) * mu_R * cos(roll) * cos(pitch),
        (r_w / B) * mu_L * cos(roll) * cos(pitch);

    // return
    return dBdu;
  }

  // function to compute B matrix evaluated at current state (and previous
  // control input)
  static Eigen::Vector3d B_Xu(double r_w, double B, double mu_R, double mu_L,
                              Eigen::Vector3d &orientation,
                              Eigen::Vector4d &input) {

    // hardcode the B matrix to be returned
    Eigen::Vector3d B_vec;

    // set all values to zero (they're random)
    B_vec.setZero();

    // unpack all of the orientation angles
    double roll = orientation[0];  // rad
    double pitch = orientation[1]; // rad
    double yaw = orientation[2];   // rad

    // unpack control inputs
    double wR = input[0]; // rad/s
    double wL = input[1]; // rad/s
    // input[2], input[3] (dwR, dwL) currently unused in the dynamics

    // actually input values for B matrix
    B_vec << (r_w / 2) * (mu_R * wR + mu_L * wL) * cos(pitch) * cos(yaw), // m/s
        (r_w / 2) * (mu_R * wR + mu_L * wL) * cos(pitch) * sin(yaw),      // m/s
        (r_w / B) * (mu_L * wL - mu_R * wR) * cos(roll) * cos(pitch); // rad/s

    // return
    return B_vec;
  }

  static Eigen::Vector3d comp_tanh(const Eigen::Vector3d &z, double epsilon) {
    return (z.array() / (2.0 * epsilon)).tanh();
  }

  std::mutex state_mutex;
  Eigen::Vector4d control;
  Eigen::VectorXd latest_state_estimate;
  Eigen::VectorXd latest_state_desired;

  // CALLER MUST HOLD state_mutex.
  // Returns the control increment du = [d_wR, d_wL] from the SMC law.
  Eigen::Vector2d compute_control_increment(double r_w, double B, double mu_r,
                                            double mu_l) {
    Eigen::Vector3d eulers(this->latest_state_estimate[2],
                           this->latest_state_estimate[3],
                           this->latest_state_estimate[4]);

    // matrix inversions -- materialized to a concrete type so we don't
    // hold a lazy expression that references a destroyed temporary
    Eigen::Matrix<double, 2, 3> dbdx_pseudo_inv =
        b_jacobian(r_w, B, mu_r, mu_l, eulers)
            .completeOrthogonalDecomposition()
            .pseudoInverse();

    Eigen::Vector3d B_prev = B_Xu(r_w, B, mu_r, mu_l, eulers, this->control);

    // tracking-relevant state slice: [x, y, yaw]
    Eigen::Vector3d x_track(this->latest_state_estimate[0],
                            this->latest_state_estimate[1],
                            this->latest_state_estimate[4]);
    Eigen::Vector3d s = this->latest_state_desired.head<3>() - x_track;

    // yaw error must be wrapped to (-pi, pi] or the controller takes the long
    // way around / flips sign across the +-180 deg line
    s(2) = std::atan2(std::sin(s(2)), std::cos(s(2)));
    Eigen::Vector3d xdot_d = this->latest_state_desired.tail<3>();

    const double epsilon = 3.0;
    Eigen::Vector3d K(2.0, 2.0, 4.0);

    Eigen::Vector3d Ks = (K.array() * comp_tanh(s, epsilon).array()).matrix();
    // target closed-loop velocity is  xdot_d + K*tanh(s)  (s = desired - x, so
    // positive error must command positive velocity); solving J*du = residual
    // sets B(u+du) ~= B_prev + residual
    Eigen::Vector3d residual = Ks + xdot_d - B_prev;

    return dbdx_pseudo_inv * residual;
  }

  // function to publish the control input calculations
  void publish_callback() {
    auto msg = geometry_msgs::msg::Quaternion();
    auto log = std_msgs::msg::String();

    if (!this->drive_motors) {
      msg.w = 0;
      msg.x = 0;
      msg.y = 0;
      msg.z = 0;
      this->control_publisher->publish(msg);
      return;
    }

    // direct wheel-speed drive: publish [wr, wl] and their rates, skip the SMC
    if (this->get_parameter("direct_drive").as_bool()) {
      const auto wheels = this->get_parameter("wheels").as_double_array();
      const double wr = wheels.size() > 0 ? wheels[0] : 0.0;
      const double wl = wheels.size() > 1 ? wheels[1] : 0.0;

      {
        std::lock_guard<std::mutex> lk(this->state_mutex);
        this->control[2] = wr - this->control[0];
        this->control[3] = wl - this->control[1];
        this->control[0] = wr;
        this->control[1] = wl;

        msg.w = this->control[0];
        msg.x = this->control[1];
        msg.y = this->control[2];
        msg.z = this->control[3];
      }

      this->control_publisher->publish(msg);
      return;
    }

    Eigen::Vector4d ctrl_snapshot;
    {
      std::lock_guard<std::mutex> lk(this->state_mutex);

      // fixed setpoint from parameters (settable live via `ros2 param set`)
      auto desired_state =
          this->get_parameter("desired_state").as_double_array();

      this->latest_state_desired.head<3>() =
          Eigen::Vector3d{desired_state[0], desired_state[1], desired_state[2]};
      this->latest_state_desired.tail<3>().setZero();

      Eigen::Vector2d du =
          compute_control_increment(GEOM.rw, GEOM.B, GEOM.mu_r, GEOM.mu_l);
      this->control[0] += du[0];
      this->control[1] += du[1];
      this->control[2] = du[0];
      this->control[3] = du[1];

      ctrl_snapshot = this->control;
    }

    msg.w = std::clamp(ctrl_snapshot[0], -1.0, 1.0);
    msg.x = std::clamp(ctrl_snapshot[1], -1.0, 1.0);
    msg.y = ctrl_snapshot[2];
    msg.z = ctrl_snapshot[3];
    if (abs(msg.w) < 0.2) {
      msg.w = 0;
      msg.y = 0;
    }
    if (abs(msg.x) < 0.2) {
      msg.x = 0;
      msg.z = 0;
    }
    this->control_publisher->publish(msg);
  }

  // function that runs when nav node gives us state estimation
  void nav_callback(nav_msgs::msg::Odometry odom) {
    // extract state estimate from nav callback (do math outside the lock)
    auto position = odom.pose.pose.position;
    auto orientation = odom.pose.pose.orientation;

    const double MY_PI = 3.141592653589;

    Eigen::Quaterniond quat(orientation.w, orientation.x, orientation.y,
                            orientation.z);
    quat.normalize();
    auto R = quat.toRotationMatrix();
    double roll = std::atan2(R(2, 1), R(2, 2));
    double s = -R(2, 0);
    double pitch =
        (std::abs(s) >= 1.0) ? std::copysign(MY_PI / 2.0, s) : std::asin(s);
    double yaw = std::atan2(R(1, 0), R(0, 0));

    std::lock_guard<std::mutex> lk(this->state_mutex);
    this->latest_state_estimate << position.x, position.y, roll, pitch, yaw;
  }

  bool drive_motors = true;

  void autonomy_callback(nav_msgs::msg::Odometry des) {
    if (des.pose.pose.orientation.x == 1.0) {
      this->drive_motors = false;
      return;
    }
    this->drive_motors = true;

    std::lock_guard<std::mutex> lk(this->state_mutex);
    this->latest_state_desired << des.pose.pose.position.x,
        des.pose.pose.position.y, des.pose.pose.position.z,
        des.twist.twist.linear.x, des.twist.twist.linear.y,
        des.twist.twist.linear.z;
  }

public:
  SmcNode() : Node("smc_node") {
    this->declare_parameter("direct_drive", true);
    this->declare_parameter("wheels", std::vector<double>{0.0, 0.0});

    // default value should be the initial position! (look at nav_node)
    this->declare_parameter("desired_state",
                            std::vector<double>{0.0, 0.0, 0.0});

    this->control_publisher =
        this->create_publisher<geometry_msgs::msg::Quaternion>("/smc/control",
                                                               10);

    // when nav publishes, nav callback runs
    this->nav_subscriber = this->create_subscription<nav_msgs::msg::Odometry>(
        "/trans_est", 10,
        std::bind(&SmcNode::nav_callback, this, std::placeholders::_1));

    this->autonomy_subscriber =
        this->create_subscription<nav_msgs::msg::Odometry>(
            "/auto_next_state", 10,
            std::bind(&SmcNode::autonomy_callback, this,
                      std::placeholders::_1));

    this->control_publisher_timer =
        this->create_wall_timer(std::chrono::duration<double>(CTRL_PERIOD_S),
                                std::bind(&SmcNode::publish_callback, this));

    // x y orientation
    this->latest_state_estimate = Eigen::VectorXd(5);
    this->latest_state_estimate.setZero();

    this->latest_state_desired = Eigen::VectorXd(6);
    this->latest_state_desired.setZero();

    this->control.setZero();
  }
};
