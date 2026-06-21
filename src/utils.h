#ifndef __SRC_UTILS__
#define __SRC_UTILS__

#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

/**
 * @brief
 *
 * @tparam Args
 * @param fmt
 * @param args
 * @return std::string
 */
template <typename... Args> std::string string_format(const char *fmt, Args... args) {
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

/**
 * @brief Meyer's singleton used on a logging instance.
 *
 */
class Logger {
private:
  using Clock = std::chrono::steady_clock;

  std::string   log_file_name;
  std::ofstream out;

  Clock::time_point         last_log_time;
  std::chrono::milliseconds rate;

public:
  Logger(std::string file_name, std::chrono::milliseconds app_rate)
      : log_file_name(std::move(file_name)), out{log_file_name, std::ios::app},
        last_log_time(Clock::now() - app_rate), // first log happens immediately
        rate(app_rate) {}

  void log(const std::string &str) {
    const auto now = Clock::now();
    if (now - this->last_log_time >= this->rate) {
      this->out << str;
      this->out.flush();
      this->last_log_time = now;
    }
  }
};

struct RoverGeometry {
  double rw = 0.175 / 2;
  double B  = 0.381;
};
inline constexpr RoverGeometry GEOM{};

#define IF_CONSTEXPR_ASSIGN(M, T, A, B)                                                                                                    \
  do {                                                                                                                                     \
    if constexpr (std::is_arithmetic_v<T>)                                                                                                 \
      (M) = (A);                                                                                                                           \
    else                                                                                                                                   \
      (M) = (B);                                                                                                                           \
  } while (0)
#define WITH_LOCK(m) std::lock_guard<std::mutex> lk(m);

template <typename T> class FilteredSampleProducer {
private:
  using Clock = std::chrono::steady_clock;

  mutable std::mutex m;
  T                  latest;
  T                  mean;
  T                  var;
  size_t             count = 0;
  double_t           alpha;
  Clock::time_point  timestamp{};
  bool               is_fresh = false;

  static T square_like(const T &x) {
    T dummy;
    IF_CONSTEXPR_ASSIGN(dummy, T, x * x, x.cwiseProduct(x));
    return dummy;
  }

  static T product_like(const T &a, const T &b) {
    T result;

    IF_CONSTEXPR_ASSIGN(result, T, a * b, a.cwiseProduct(b));

    return result;
  }

  static T sqrt_like(const T &x) {
    T dummy;
    IF_CONSTEXPR_ASSIGN(dummy, T, sqrt(x), x.array().sqrt().matrix());
    return dummy;
  }

public:
  explicit FilteredSampleProducer(const T &zero, double alpha_param) : latest(zero), mean(zero), var(zero), alpha(alpha_param) {};

  void put(const T &x) {
    std::lock_guard<std::mutex> lg(this->m);

    this->latest    = x;
    this->timestamp = Clock::now();
    this->count++;

    if (this->count == 1) {
      this->mean = x;
      return;
    }

    const double n     = static_cast<double>(this->count);
    const T      delta = x - this->mean;

    this->mean += delta / n;

    const T delta2 = x - this->mean;

    this->var      = ((n - 2.0) / (n - 1.0)) * this->var + product_like(delta, delta2) / (n - 1.0);
    this->is_fresh = true;
  }

  bool take(T &out, Clock::time_point &when) {
    std::lock_guard<std::mutex> lg(this->m);

    out            = this->latest;
    when           = this->timestamp;
    this->is_fresh = false;

    return true;
  }

  bool              has_fresh() const { WITH_LOCK(this->m) return this->is_fresh; }
  T                 peek() const { WITH_LOCK(this->m) return this->latest; }
  T                 get_mean() const { WITH_LOCK(m) return this->mean; }
  T                 get_variance() const { WITH_LOCK(m) return this->var; }
  T                 get_stddev() const { WITH_LOCK(m) return FilteredSampleProducer::sqrt_like(this->var); }
  std::size_t       get_count() const { WITH_LOCK(m) return this->count; }
  Clock::time_point stamp() const { WITH_LOCK(m) return this->timestamp; }
};

#endif