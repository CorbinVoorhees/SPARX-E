/*
For the file to run:
- ROS2 needs to be running
- A driver is publishing to /unilidar/cloud
- The node is built in a ROS2 package and is actually running (i.e. ros2 run
<package_name> <node_name>)
*/

/* Summary:
Converts /unilidar/cloud PointCloud2 data into XYZ points, stores them in a
class, then processes them into:
1. filtered obstacle-candidate points
2. KD-tree
3. obstacle clusters
4. Gaussian obstacle cost model

More assumptions
- The LiDAR topic is /unilidar/cloud
- The topic type is sensor_msgs/msg/PointCloud2
- PCL is available (point cloud library)
- Rover body frame convention:
    x = forward
    y = left
    z = up
- LiDAR points initially arrive in the LiDAR frame
- LiDAR mounting error is modeled as a yaw rotation about z then 180 deg roll
about x
- Points close to ground are ignored as non-obstacles
- Obstacles are found by clustering remaining points
- Each obstacle cluster becomes one Gaussian cost source
*/

#include <lidar_processor/msg/env.hpp>
#include <lidar_processor/msg/obstacle.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>

#include <cmath>
#include <fstream>
#include <mutex>
#include <sstream>
#include <vector>

using namespace std::chrono_literals;

class LidarObstacleProcessor : public rclcpp::Node {
public:
  LidarObstacleProcessor() : Node("lidar_obstacle_processor") {
    raw_cloud_lidar_ =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    accumulated_cloud_ =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    cloud_body_ =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    filtered_cloud_ =
        pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

    kd_tree_ = pcl::search::KdTree<pcl::PointXYZ>::Ptr(
        new pcl::search::KdTree<pcl::PointXYZ>);

    R_body_lidar_ = computeMountRotation();

    // Connects to LiDAR data stream
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/unilidar/cloud", 10,
        std::bind(&LidarObstacleProcessor::pointCloudCallback, this,
                  std::placeholders::_1));

    publication_ =
        this->create_publisher<lidar_processor::msg::Env>("/obstacles", 10);

    // Timer to process the accumulated cloud and write CSV every 0.5 seconds
    // (500ms)
    processing_timer_ = this->create_wall_timer(
        2000ms, std::bind(&LidarObstacleProcessor::timerCallback, this));

    this->estimation_subscription =
        this->create_subscription<nav_msgs::msg::Odometry>(
            "/trans_est", 10,
            std::bind(&LidarObstacleProcessor::estimation_callback, this,
                      std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "LidarObstacleProcessor started.");
  }

  void process() {
    if (cloud_body_->empty()) {
      RCLCPP_WARN(this->get_logger(), "No point cloud data stored yet.");
      return;
    }

    filterPoints();
    buildKdTree();
    computeObstacleClusters();
    computeGaussianCosts();

    RCLCPP_INFO(
        this->get_logger(),
        "Processed accumulated cloud: raw=%zu, filtered=%zu, obstacles=%zu",
        cloud_body_->size(), filtered_cloud_->size(), obstacles_.size());
  }

private:
  struct Obstacle {
    Eigen::Vector2f center_xy;
    float height;
    float sigma;
    Eigen::Matrix2f covariance;
    Eigen::Matrix2f inv_covariance;
    float amplitude;
    int num_points;
  };

  // PARAMETERS

  float mount_offset_angle_rad_ = -0.27f;

  float min_x_ = 0.05f;
  float max_x_ = 2.0f;
  float max_abs_y_ = 2.0f;
  float min_z_ = 0.0f;
  float max_z_ = 2.0f;

  float ground_threshold_ = 0.1f;

  float cluster_tolerance_ = 0.2f;
  int min_cluster_size_ = 10;
  int max_cluster_size_ = 2000;

  float base_sigma_ = 0.40f;
  float sigma_per_height_ = 0.50f;
  float base_amplitude_ = 1.0f;
  float amplitude_per_height_ = 5.0f;

  // STORED DATA

  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud_lidar_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr
      accumulated_cloud_; // New buffer for accumulating points
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_body_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree_;

  std::vector<Obstacle> obstacles_;

  std::mutex cloud_mutex_; // Mutex to protect read/writes to accumulated_cloud_
  std::mutex obstacles_mutex_; // Mutex to protect read/writes to obstacles_

  Eigen::Matrix3f R_body_lidar_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<lidar_processor::msg::Env>::SharedPtr publication_;
  rclcpp::TimerBase::SharedPtr processing_timer_; // Timer object

  // get a nav_filters object
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      estimation_subscription;

  // callback
  std::mutex estimation_mutex;
  Eigen::Vector3d pos;
  Eigen::Vector3d orientation;
  Eigen::Vector3d velocity;
  void estimation_callback(const nav_msgs::msg::Odometry::SharedPtr est_msg) {
    std::lock_guard<std::mutex> lg(estimation_mutex);
    // get current position, orientation, velocity.
    auto &p = est_msg->pose.pose;
    pos[0] = p.position.x;
    pos[1] = p.position.y;
    pos[2] = p.position.z;

    // get the orientation
    Eigen::Quaterniond quat(p.orientation.w, p.orientation.x, p.orientation.y,
                            p.orientation.z);

    static constexpr double MY_PI = 3.141592653589793238462643383279502884;
    auto R = quat.toRotationMatrix();
    double roll = std::atan2(R(2, 1), R(2, 2));
    double s = -R(2, 0);
    double pitch =
        (std::abs(s) >= 1.0) ? std::copysign(MY_PI / 2.0, s) : std::asin(s);
    double yaw = std::atan2(R(1, 0), R(0, 0));

    orientation[0] = roll;
    orientation[1] = pitch;
    orientation[2] = yaw;

    // get velos
    auto &t = est_msg->twist.twist;
    velocity[0] = t.linear.x;
    velocity[1] = t.linear.y;
    velocity[2] = t.linear.z;
  }

  // CALLBACK - raw ROS data to XYZ
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    raw_cloud_lidar_->clear();
    pcl::fromROSMsg(*msg, *raw_cloud_lidar_);

    // Process transformations outside the lock to minimize blocking
    pcl::PointCloud<pcl::PointXYZ> temp_cloud;

    for (const auto &p_lidar : raw_cloud_lidar_->points) {
      Eigen::Vector3f p_L(p_lidar.x, p_lidar.y, p_lidar.z);
      Eigen::Vector3f p_B = R_body_lidar_ * p_L;

      pcl::PointXYZ p_body;
      p_body.x = p_B.x();
      p_body.y = p_B.y();
      p_body.z = p_B.z();

      temp_cloud.points.push_back(p_body);
    }

    // Lock mutex and append the newly transformed points to the accumulator
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      *accumulated_cloud_ +=
          temp_cloud; // PCL supports direct point cloud concatenation
      accumulated_cloud_->width = accumulated_cloud_->points.size();
      accumulated_cloud_->height = 1;
      accumulated_cloud_->is_dense = false;
    }
  }

  // TIMER CALLBACK - Processes the accumulated cloud and writes to CSV at 2Hz

  void timerCallback() {
    // 1. Grab the accumulated data safely
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);

      if (accumulated_cloud_->empty()) {
        return; // No data arrived in the last 0.5s, skip processing
      }

      // Copy the accumulated data into the processing cloud
      *cloud_body_ = *accumulated_cloud_;

      // Clear the accumulator to start fresh for the next 0.5 seconds
      accumulated_cloud_->clear();
      accumulated_cloud_->width = 0;
      accumulated_cloud_->height = 1;
    }

    // 2. Process the collated 0.5 seconds worth of data
    process();

    // 3. Write results to CSV safely
    {
      std::lock_guard<std::mutex> lock(obstacles_mutex_);

      std::stringstream storage;
      std::ofstream fuckass("fuck_gemini.csv", std::ios::out | std::ios::trunc);

      if (!fuckass.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open output file!");
        return;
      }

      for (const auto &i : obstacles_) {
        storage << i.center_xy.x() << "," << i.center_xy.y() << ","
                << i.covariance(0, 0) << "," << i.covariance(0, 1) << ","
                << i.covariance(1, 0) << "," << i.covariance(1, 1);

        storage << "," << pos[0] << "," << pos[1] << "," << pos[2];
        storage << "," << orientation[0] << "," << orientation[1] << ","
                << orientation[2];
        storage << "," << velocity[0] << "," << velocity[1] << ","
                << velocity[2];

        storage << "\n";
      }

      fuckass << storage.str();
      fuckass.close();
      publish_obstacles();
    }
  }

  // Rotation helper

  Eigen::Matrix3f computeMountRotation() {
    Eigen::Matrix3f R_yaw;
    R_yaw << std::cos(mount_offset_angle_rad_),
        -std::sin(mount_offset_angle_rad_), 0.0f,
        std::sin(mount_offset_angle_rad_), std::cos(mount_offset_angle_rad_),
        0.0f, 0.0f, 0.0f, 1.0f;

    Eigen::Matrix3f R_upside_down;
    R_upside_down << 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, -1.0f;

    return R_yaw * R_upside_down;
  }

  // FILTERING STUFF

  void filterPoints() {
    filtered_cloud_->clear();

    float projector_height = 4.0f;
    float rover_height = 0.245f;
    float ground_z = -rover_height;

    for (const auto &p : cloud_body_->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        continue;
      }

      if (p.x < min_x_ || p.x > max_x_)
        continue;
      if (std::abs(p.y) > max_abs_y_)
        continue;

      if (p.z <= (ground_z + ground_threshold_)) {
        continue;
      }

      if (p.z > projector_height)
        continue;

      if (p.z > 1.0f) {
        RCLCPP_WARN(this->get_logger(), "High point detected: z=%.2f", p.z);
      }

      filtered_cloud_->points.push_back(p);
    }

    filtered_cloud_->width = filtered_cloud_->points.size();
    filtered_cloud_->height = 1;
    filtered_cloud_->is_dense = false;
  }

  void buildKdTree() {
    if (filtered_cloud_->empty()) {
      return;
    }

    kd_tree_->setInputCloud(filtered_cloud_);
  }

  // Clustering

  void computeObstacleClusters() {
    if (filtered_cloud_->empty()) {
      std::lock_guard<std::mutex> lock(obstacles_mutex_);
      obstacles_.clear();
      return;
    }

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(kd_tree_);
    ec.setInputCloud(filtered_cloud_);
    ec.extract(cluster_indices);

    // Compute temporarily, so we hold the lock for as little time as possible
    std::vector<Obstacle> new_obstacles;

    for (const auto &cluster : cluster_indices) {
      Obstacle obs;
      obs.center_xy = Eigen::Vector2f(0.0f, 0.0f);
      obs.height = -std::numeric_limits<float>::infinity();
      obs.num_points = static_cast<int>(cluster.indices.size());

      for (int idx : cluster.indices) {
        const auto &p = filtered_cloud_->points[idx];

        obs.center_xy.x() += p.x;
        obs.center_xy.y() += p.y;

        if (p.z > obs.height) {
          obs.height = p.z;
        }
      }

      obs.center_xy /= static_cast<float>(obs.num_points);

      Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();

      for (int idx : cluster.indices) {
        const auto &p = filtered_cloud_->points[idx];

        Eigen::Vector2f d;
        d << p.x - obs.center_xy.x(), p.y - obs.center_xy.y();

        cov += d * d.transpose();
      }

      cov /= (static_cast<float>(obs.num_points) - 1);

      float min_sigma = 0.25f;
      cov += Eigen::Matrix2f::Identity() * min_sigma * min_sigma;

      obs.covariance = cov;
      obs.inv_covariance = cov.inverse();
      obs.amplitude = base_amplitude_ * (1.0f + 0.10f * obs.height);

      new_obstacles.push_back(obs);
    }

    // Lock mutex, update real container safely
    {
      std::lock_guard<std::mutex> lock(obstacles_mutex_);
      obstacles_ = std::move(new_obstacles);
    }
  }

  void computeGaussianCosts() {
    std::lock_guard<std::mutex> lock(obstacles_mutex_);
    for (auto &obs : obstacles_) {
      obs.amplitude = base_amplitude_ * (1.0f + 0.10f * obs.height);
    }
  }

  float gaussianCostAtPoint(float x, float y) const {
    float total_cost = 0.0f;

    Eigen::Vector2f p;
    p << x, y;

    for (const auto &obs : obstacles_) {
      Eigen::Vector2f d = p - obs.center_xy;
      float exponent = -0.5f * d.transpose() * obs.inv_covariance * d;
      float cost = obs.amplitude * std::exp(exponent);
      total_cost += cost;
    }

    return total_cost;
  }

  float pathCostStraightLine(float heading_rad, float path_length) const {
    int num_samples = 50;
    float total_cost = 0.0f;

    for (int i = 1; i <= num_samples; i++) {
      float s =
          path_length * static_cast<float>(i) / static_cast<float>(num_samples);
      float x = s * std::cos(heading_rad);
      float y = s * std::sin(heading_rad);

      total_cost += gaussianCostAtPoint(x, y);
    }

    return total_cost;
  }

  void publish_obstacles() {
    auto message = lidar_processor::msg::Env();

    for (Obstacle obs : obstacles_) {
      lidar_processor::msg::Obstacle item;
      item.center[0] = obs.center_xy.x();
      item.center[1] = obs.center_xy.y();
      item.covariance[0] = obs.covariance(0, 0);
      item.covariance[1] = obs.covariance(0, 1);
      item.covariance[2] = obs.covariance(1, 0);
      item.covariance[3] = obs.covariance(1, 1);
      message.environment.push_back(item);
    }

    this->publication_->publish(message);
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LidarObstacleProcessor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}