#pragma once

#include <Eigen/Dense>

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

#include "data_prefilter.hpp"
#include "operations.hpp"
#include "sensor_hook.hpp"

namespace Sensor {
using SteadyClock = std::chrono::steady_clock;

using CorrectFn =
    std::function<void(const Eigen::MatrixXd &H, const Eigen::MatrixXd &R,
                       const Eigen::VectorXd &r)>;

struct ISensorTag {};

template <typename Tag, typename InputType, typename OutputType>
class ISensor : protected Operations::Read<InputType>,
                protected Operations::Write<OutputType> {
  static_assert(std::is_base_of_v<ISensorTag, Tag>,
                "Sensor tags must derive from ISensorTag");

public:
  using SensorInputType = InputType;
  using SensorOutputType = OutputType;

  virtual ~ISensor() = default;
};

template <typename... Sensors> class SensorTable final {
  static_assert((std::is_class_v<Sensors> && ...),
                "SensorTable entries must be sensor classes");

private:
  template <typename ISensorType> class SensorSlot final {
  public:
    using InputType = typename ISensorType::SensorInputType;

  private:
    SensorHook<InputType> hook;
    std::optional<DataPrefilter<InputType>> producer;
    std::optional<ISensorType> sensor;

    std::mutex processing_mutex;
    bool initialized = false;
    std::atomic_bool stop_requested{false};
    std::thread worker;

    void accept(const InputType &input, SteadyClock::time_point timestamp) {
      std::lock_guard<std::mutex> lock(processing_mutex);
      const auto filtered = producer->put(input, timestamp);
      if (!filtered.has_value())
        return;

      const Eigen::MatrixXd measurement_covariance =
          DataPrefilter<InputType>::covariance(filtered->variance);

      if (!initialized) {
        initialized = true;
        hook.initialize(*filtered);
      }

      hook.emit(filtered->data, measurement_covariance, filtered->timestamp);
    }

    void run() noexcept {
      while (!stop_requested.load(std::memory_order_acquire)) {
        InputType input{};
        const int result = sensor->read(input);

        if (result > 0)
          accept(input, SteadyClock::now());
        else if (result < 0)
          return;
      }
    }

    void stop() noexcept {
      stop_requested.store(true, std::memory_order_release);
      if (worker.joinable())
        worker.join();
    }

  public:
    SensorSlot() = default;
    SensorSlot(const SensorSlot &) = delete;
    SensorSlot &operator=(const SensorSlot &) = delete;

    ~SensorSlot() { stop(); }

    template <typename... Args>
    bool emplace(PrefilterConfig<InputType> sample_config, Args &&...args) {
      if (sensor.has_value())
        return false;

      producer.emplace(std::move(sample_config));
      sensor.emplace(std::forward<Args>(args)...);
      worker = std::thread([this] { run(); });
      return true;
    }

    void bind(typename SensorHook<InputType>::SnapshotCallback snapshot,
              typename SensorHook<InputType>::InitializeCallback initialize) {
      hook.bind(std::move(snapshot), std::move(initialize));
    }
  };

  std::tuple<SensorSlot<Sensors>...> slots;

  template <typename Sensor> SensorSlot<Sensor> &slot() {
    return std::get<SensorSlot<Sensor>>(slots);
  }

  template <typename Sensor> const SensorSlot<Sensor> &slot() const {
    return std::get<SensorSlot<Sensor>>(slots);
  }

public:
  SensorTable() = default;
  SensorTable(const SensorTable &) = delete;
  SensorTable &operator=(const SensorTable &) = delete;

  template <typename Sensor, typename... Args>
  bool register_sensor(
      PrefilterConfig<typename Sensor::SensorInputType> sample_config,
      Args &&...args) {
    return slot<Sensor>().emplace(std::move(sample_config),
                                  std::forward<Args>(args)...);
  }

  template <typename Sensor>
  void
  bind(typename SensorHook<typename Sensor::SensorInputType>::SnapshotCallback
           snapshot,
       typename SensorHook<typename Sensor::SensorInputType>::InitializeCallback
           initialize =
               SensorHook<typename Sensor::SensorInputType>::NoInitialize) {
    slot<Sensor>().bind(std::move(snapshot), std::move(initialize));
  }
};

} // namespace Sensor
