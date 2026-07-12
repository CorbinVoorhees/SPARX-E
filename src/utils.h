#pragma once

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
  double rw = 0.175 / 2; // wheel radius [m]
  double B = 0.381;      // track width [m]
  double mu_r = 0.95;    // right wheel slip efficiency [-]
  double mu_l = 0.95;    // left wheel slip efficiency [-]
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

template <typename T> class FilteredSampleProducer {
private:
  using Clock = std::chrono::steady_clock;

  mutable std::mutex m;
  T latest;
  T mean; // lifetime Welford mean (used by calibration hooks)
  T var;
  size_t count = 0;
  Clock::time_point timestamp{};

  // Sliding-window variance -> R. A fixed window over PAST samples keeps R
  // independent of the current update (no adaptive-R / gain coupling), and the
  // windowed mean + mean-of-squares are both maintained recursively (O(1)):
  //   var = mean_sq_win - mean_win^2   over the last win_size samples.
  static constexpr size_t win_size = 50;
  std::vector<T> win_buf; // ring buffer of recent samples
  size_t win_head = 0;    // next write slot
  size_t win_count = 0;   // filled entries (<= win_size)
  T mean_win;             // recursive windowed mean
  T mean_sq_win;          // recursive windowed mean of squares

  static T square_like(const T &x) {
    T dummy;
    IF_CONSTEXPR_ASSIGN(dummy, T, x * x, x.cwiseProduct(x));
    return dummy;
  }

  static T sqrt_like(const T &x) {
    T dummy;
    IF_CONSTEXPR_ASSIGN(dummy, T, sqrt(x), x.array().sqrt().matrix());
    return dummy;
  }

  // T apply_biquad(const T &x) {
  //   const T y = coeffs_.b0 * x + this->z1_;
  //   this->z1_ = coeffs_.b1 * x - coeffs_.a1 * y + this->z2_;
  //   this->z2_ = coeffs_.b2 * x - coeffs_.a2 * y;
  //   return y;
  // }

public:
  explicit FilteredSampleProducer(const T &zero)
      : latest(zero), mean(zero), var(zero), win_buf(win_size, zero),
        mean_win(zero), mean_sq_win(zero){};

  // void set_filter(BiquadCoeffs c) {
  //   WITH_LOCK(this->m)
  //   this->coeffs_ = c;
  //   this->filter_enabled_ = true;
  // }

  void put(const T &x_in, Clock::time_point sample_time = Clock::now()) {
    std::lock_guard<std::mutex> lg(this->m);

    // const T x = this->filter_enabled_ ? this->apply_biquad(x_in) : x_in;
    const T x = x_in;

    this->latest = x;
    this->timestamp = sample_time;
    this->count++;

    if (this->count == 1) {
      this->mean = x;
      return;
    }

    const double n = static_cast<double>(this->count);
    const T delta = x - this->mean;
    this->mean += delta / n; // lifetime Welford mean, unchanged (calib uses it)

    // --- sliding-window variance over the ring buffer (recursive, O(1)) ---
    const T x_sq = square_like(x);
    if (this->win_count < win_size) {
      // window still filling: incremental mean of the first win_count+1 samples
      const double m = static_cast<double>(this->win_count) + 1.0;
      this->mean_win += (x - this->mean_win) / m;
      this->mean_sq_win += (x_sq - this->mean_sq_win) / m;
      ++this->win_count;
    } else {
      // window full: swap oldest for newest, mean shifts by (new - old)/N
      const T x_old = this->win_buf[this->win_head];
      const double N = static_cast<double>(win_size);
      this->mean_win += (x - x_old) / N;
      this->mean_sq_win += (x_sq - square_like(x_old)) / N;
    }

    this->win_buf[this->win_head] = x;
    this->win_head = (this->win_head + 1) % win_size;

    // population variance of the window: E[x^2] - E[x]^2, floored at 0
    const T v = this->mean_sq_win - square_like(this->mean_win);
    IF_CONSTEXPR_ASSIGN(this->var, T, std::max(v, 0.0), v.cwiseMax(0.0));
  }

  void get_latest(T &out, Clock::time_point &when) {
    WITH_LOCK(this->m)
    out = this->latest;
    when = this->timestamp;
  }

  T peek() const { WITH_LOCK(this->m) return this->latest; }
  T get_mean() const { WITH_LOCK(m) return this->mean; }
  T get_variance() const { WITH_LOCK(m) return this->var; }
  T get_stddev() const {
    WITH_LOCK(m) return FilteredSampleProducer::sqrt_like(this->var);
  }
  std::size_t get_count() const { WITH_LOCK(m) return this->count; }
  Clock::time_point stamp() const { WITH_LOCK(m) return this->timestamp; }
};
