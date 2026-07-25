#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

using SteadyClock = std::chrono::steady_clock;
/**
 * @brief
 *
 * @tparam Args
 * @param fmt
 * @param args
 * @return std::string
 */
template <typename... Args>
std::string string_format(const char *fmt, Args... args) {
  // obtain theoretical exact container size required for all variable
  // substitutions
  int size = std::snprintf(nullptr, 0, fmt, args...);
  if (size <= 0)
    return "";

  // initialize return string with exact sizing and perform argument substition
  std::string result(size, '\0');
  std::snprintf(result.data(), result.size() + 1, fmt, args...);

  return result;
}

// perf A/B: uncomment to make every Logger a no-op
// #define NO_LOGGING

/**
 * @brief Meyer's singleton used on a logging instance.
 *
 */
class Logger {
private:
  using Clock = std::chrono::steady_clock;

  std::string log_file_name;
  std::ofstream out;

  Clock::time_point last_log_time;
  std::chrono::milliseconds rate;

public:
  Logger(std::string file_name, std::chrono::milliseconds app_rate)
      : log_file_name(std::move(file_name)),
        out{log_file_name, std::ios::trunc},
        last_log_time(Clock::now() - app_rate), // first log happens immediately
        rate(app_rate) {}

  void log(const std::string &str) {
#ifdef NO_LOGGING
    return;
#endif
    const auto now = Clock::now();
    if (now - this->last_log_time >= this->rate) {
      this->out << str;
      this->out.flush();
      this->last_log_time = now;
    }
  }
};

// single source of truth for rover geometry + wheel slip efficiencies —
// used by the SMC, the TRANS-EKF wheel model, and the MEKF yaw odometry.
struct RoverGeometry {
  double rw = 0.175 / 2.0; // wheel radius [m]
  double B = 0.400;        // track width [m]
  double mu_r = 0.95;      // right wheel slip efficiency [-]
  double mu_l = 0.95;      // left wheel slip efficiency [-]
};
inline constexpr RoverGeometry GEOM{};

// smc control publish period [s]; dwr/dwl are command differences per tick,
// so dw / CTRL_PERIOD_S is the commanded wheel acceleration.
inline constexpr double CTRL_PERIOD_S = 0.067;

#define IF_CONSTEXPR_ASSIGN(M, T, A, B)                                        \
  do {                                                                         \
    if constexpr (std::is_arithmetic_v<T>)                                     \
      (M) = (A);                                                               \
    else                                                                       \
      (M) = (B);                                                               \
  } while (0)

#define WITH_LOCK(m) std::lock_guard<std::mutex> lk(m);
#define WITH_UQ_LOCK(m) std::unique_lock<std::shared_mutex> lk(m);
#define WITH_SH_LOCK(m) std::shared_lock<std::shared_mutex> lk(m);

// Reads `expected_count` whitespace/newline-separated doubles from `path`.
// Returns false (leaving `out` untouched) if the file is missing or short.
inline bool load_doubles(const std::string &path, std::vector<double> &out,
                         size_t expected_count) {
  std::ifstream in(path);
  if (!in.is_open())
    return false;

  std::vector<double> vals;
  vals.reserve(expected_count);
  double v;
  while (vals.size() < expected_count && (in >> v))
    vals.push_back(v);

  if (vals.size() != expected_count)
    return false;

  out = std::move(vals);
  return true;
}

// Convert a calibrated body-frame accelerometer sample into the start-relative
// translation frame and remove the gravity vector captured at startup.
inline Eigen::Vector3d
linear_accel_in_start_frame(const Eigen::Quaterniond &start_from_body,
                            const Eigen::Vector3d &accel_body,
                            const Eigen::Vector3d &gravity_start) {
  return start_from_body.normalized() * accel_body - gravity_start;
}
