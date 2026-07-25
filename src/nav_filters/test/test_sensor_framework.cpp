#include <nav_filters/sensor/sensor.hpp>
#include <nav_filters/sensor_impl/topic_sensor.hpp>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <variant>

namespace {

using namespace std::chrono_literals;

struct PullTag : Sensor::ISensorTag {};
struct PushTag : Sensor::ISensorTag {};

struct PullState {
  std::atomic_bool first_read{true};
  std::atomic_bool poll_active{false};
};

class PullSensor final
    : public Sensor::ISensor<PullTag, double, std::monostate> {
private:
  std::shared_ptr<PullState> state;

public:
  inline static constexpr bool pull_driven = true;

  explicit PullSensor(std::shared_ptr<PullState> input_state)
      : state(std::move(input_state)) {}

  int read(double &value) noexcept override {
    state->poll_active.store(true);
    std::this_thread::sleep_for(10ms);
    state->poll_active.store(false);

    if (state->first_read.exchange(false)) {
      value = 42.0;
      return 1;
    }
    return 0;
  }
};

using PushSensor =
    Sensor::TopicSensor<std_msgs::msg::Float64, PushTag, double,
                        std::monostate>;
using Table = Sensor::SensorTable<PullSensor, PushSensor>;

class SensorTableTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    int argc = 0;
    rclcpp::init(argc, nullptr);
  }

  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST_F(SensorTableTest, StartsPullWorkerEmitsAndJoinsAtDestruction) {
  auto state = std::make_shared<PullState>();
  std::mutex mutex;
  std::condition_variable received;
  double value = 0.0;

  {
    Table table;
    table.bind<PullSensor>(
        [&](const double input, const Eigen::MatrixXd &,
            Sensor::SteadyClock::time_point) {
          {
            std::lock_guard<std::mutex> lock(mutex);
            value = input;
          }
          received.notify_one();
        });

    EXPECT_TRUE(table.register_sensor<PullSensor>({0.0, 0, ""}, state));
    EXPECT_FALSE(table.register_sensor<PullSensor>({0.0, 0, ""}, state));

    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(received.wait_for(lock, 1s, [&] { return value == 42.0; }));
  }

  EXPECT_FALSE(state->poll_active.load());
}

TEST_F(SensorTableTest, TopicSensorsOwnSubscriptionAndFeedTheSameHook) {
  const auto sensor_node =
      std::make_shared<rclcpp::Node>("sensor_framework_subscription_test");
  const auto publisher_node =
      std::make_shared<rclcpp::Node>("sensor_framework_publisher_test");

  Table table;
  double value = 0.0;
  std::size_t initialized_count = 0;

  table.bind<PushSensor>(
      [&](const double input, const Eigen::MatrixXd &,
          Sensor::SteadyClock::time_point) { value = input; },
      [&](const Sensor::DataPrefilter<double>::Stamped &prefiltered) {
        initialized_count = prefiltered.count;
      });
  ASSERT_TRUE(table.register_sensor<PushSensor>(
      {0.0, 0, ""}, *sensor_node, "/sensor_framework/value", rclcpp::QoS(10)));

  const auto publisher = publisher_node->create_publisher<std_msgs::msg::Float64>(
      "/sensor_framework/value", 10);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(sensor_node);

  const auto discovery_deadline = std::chrono::steady_clock::now() + 1s;
  while (publisher->get_subscription_count() == 0 &&
         std::chrono::steady_clock::now() < discovery_deadline) {
    executor.spin_some();
    std::this_thread::sleep_for(10ms);
  }
  ASSERT_GT(publisher->get_subscription_count(), 0U);

  std_msgs::msg::Float64 message;
  message.data = 7.0;
  publisher->publish(message);

  const auto delivery_deadline = std::chrono::steady_clock::now() + 1s;
  while (value != 7.0 && std::chrono::steady_clock::now() < delivery_deadline) {
    executor.spin_some();
    std::this_thread::sleep_for(10ms);
  }

  EXPECT_DOUBLE_EQ(value, 7.0);
  EXPECT_EQ(initialized_count, 1U);
}

} // namespace
