#pragma once

#include "../../../../utils.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Sensor {

template <typename InputType> struct PrefilterConfig {
  InputType initial{};
  std::size_t minimum_samples = 0;
  std::string calibration_path;
};

template <typename InputType> class DataPrefilter {
public:
  struct Stamped {
    InputType data;
    SteadyClock::time_point timestamp;
    InputType mean;
    InputType variance;
    std::size_t count;
  };

private:
  mutable std::mutex mutex;
  const PrefilterConfig<InputType> config;
  Stamped latest;

  InputType variance_accumulator;

  Eigen::MatrixXd calibration_matrix;
  Eigen::VectorXd calibration_bias;

  static std::size_t input_dimension(const InputType &input) {
    std::size_t dimension;
    IF_CONSTEXPR_ASSIGN(dimension, InputType, 1,
                        static_cast<std::size_t>(input.size()));
    return dimension;
  }

  static InputType product_like(const InputType &left, const InputType &right) {
    InputType result;
    IF_CONSTEXPR_ASSIGN(result, InputType, left * right,
                        left.cwiseProduct(right));
    return result;
  }

public:
  explicit DataPrefilter(PrefilterConfig<InputType> input_config)
      : config(std::move(input_config)),
        latest{config.initial, {}, config.initial, config.initial, 0},
        variance_accumulator(config.initial - config.initial),
        calibration_matrix(Eigen::MatrixXd::Identity(
            input_dimension(config.initial), input_dimension(config.initial))),
        calibration_bias(
            Eigen::VectorXd::Zero(input_dimension(config.initial))) {
    if (config.calibration_path.empty())
      return;

    const Eigen::Index dimension =
        static_cast<Eigen::Index>(input_dimension(config.initial));
    const std::size_t expected_values =
        static_cast<std::size_t>(dimension + dimension * dimension);
    std::vector<double> values;
    if (!load_doubles(config.calibration_path, values, expected_values) ||
        !std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); })) {
      std::cerr << "[DataPrefilter] invalid calibration file "
                << config.calibration_path
                << " -- using identity/zero calibration\n";
      return;
    }

    calibration_bias =
        Eigen::Map<const Eigen::VectorXd>(values.data(), dimension);
    calibration_matrix =
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>(
            values.data() + dimension, dimension, dimension);
  }

  static Eigen::MatrixXd covariance(const InputType &variance) {
    Eigen::MatrixXd result;
    IF_CONSTEXPR_ASSIGN(result, InputType,
                        Eigen::MatrixXd::Constant(1, 1, variance),
                        variance.asDiagonal());
    return result;
  }

  std::optional<Stamped>
  put(const InputType &input,
      SteadyClock::time_point sample_time = SteadyClock::now()) {
    std::lock_guard<std::mutex> lock(mutex);

    InputType filtered;
    IF_CONSTEXPR_ASSIGN(filtered, InputType,
                        calibration_matrix(0, 0) *
                            (input - calibration_bias(0)),
                        calibration_matrix * (input - calibration_bias));

    ++latest.count;

    if (latest.count <= config.minimum_samples) {
      if (latest.count == 1) {
        latest.mean = filtered;
      } else {
        const double count = static_cast<double>(latest.count);
        const InputType delta = filtered - latest.mean;
        latest.mean += delta / count;
        variance_accumulator += product_like(delta, filtered - latest.mean);
      }

      const InputType variance =
          variance_accumulator / static_cast<double>(latest.count);
      IF_CONSTEXPR_ASSIGN(latest.variance, InputType, std::max(variance, 0.0),
                          variance.cwiseMax(0.0));
    }

    latest.data = filtered;
    latest.timestamp = sample_time;

    if (latest.count <= config.minimum_samples)
      return std::nullopt;

    return latest;
  }

  Stamped peek() const {
    std::lock_guard<std::mutex> lock(mutex);
    return latest;
  }
};

} // namespace Sensor
