#ifndef SENSOR_HOOK_HPP
#define SENSOR_HOOK_HPP

#include "data_prefilter.hpp"
#include <functional>
#include <vector>
namespace Sensor {
template <typename InputType> class SensorHook final {
public:
  using SnapshotCallback = std::function<void(
      const InputType &, const Eigen::MatrixXd &, SteadyClock::time_point)>;
  using InitializeCallback =
      std::function<void(const typename DataPrefilter<InputType>::Stamped &)>;

  inline static const InitializeCallback NoInitialize =
      [](const typename DataPrefilter<InputType>::Stamped &) {};

private:
  std::mutex mutex;
  std::vector<SnapshotCallback> snapshot_callbacks;
  std::vector<InitializeCallback> initialize_callbacks;

public:
  void bind(SnapshotCallback snapshot,
            InitializeCallback initialize = NoInitialize) {
    std::lock_guard<std::mutex> lock(mutex);
    snapshot_callbacks.emplace_back(std::move(snapshot));
    initialize_callbacks.emplace_back(std::move(initialize));
  }

  void
  initialize(const typename DataPrefilter<InputType>::Stamped &prefiltered) {
    std::vector<InitializeCallback> callbacks;
    {
      std::lock_guard<std::mutex> lock(mutex);
      callbacks.swap(initialize_callbacks);
    }

    for (auto &callback : callbacks)
      callback(prefiltered);
  }

  void emit(const InputType &data, const Eigen::MatrixXd &covariance,
            SteadyClock::time_point timestamp) {
    std::vector<SnapshotCallback> callbacks;
    {
      std::lock_guard<std::mutex> lock(mutex);
      callbacks = snapshot_callbacks;
    }

    for (auto &callback : callbacks)
      callback(data, covariance, timestamp);
  }
};
}; // namespace Sensor
#endif
