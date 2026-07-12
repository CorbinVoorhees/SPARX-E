#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <map>
#include <queue>
#include <vector>

/**
 * @brief Expected accel (specific-force) reading at rest (m/s^2).
 * Frame is x-forward, y-right, z-down: gravity is +z, so the accel reads -g.
 */
inline const Eigen::Vector3d grav_vec{0.0, 0.0, -9.80655};
inline double MIN_PREDICT_INTVL = 1e-3;

/**
 * @brief helper function takes an R3 vector and generates the skew symmetric
 * matrix.
 *
 * @param v
 * @return Eigen::Matrix3d
 */
inline Eigen::Matrix3d skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return S;
}
// useful reassignment
using SteadyClock = std::chrono::steady_clock;

/**
 * @brief Common Kalman Filter (CKF) is a set of functions but allow for users
 * to interface with the sensor pipeline and implement asynchronous updates.
 */
namespace ckf {

// one Kalman correction step captured whole, for offline debugging
struct KFTrace {
  double t_s;
  std::string sensor;
  uint32_t step;
  bool applied;
  double nis;
  Eigen::VectorXd z, Hx, r, S_diag, R_diag, dx, P_diag, x_pre, x_post;
};


/**
 * @brief A class containing basic Kalman filtering functions, and an important
 * inheritance-based framework for to do async estimation and updates on sparse,
 * infrequent data. CommonKF controls the process and order of operations for
 * the filter! The user can only imeplement specific break-out functions listed
 * below that controls the method of error application, predict step and a few
 * @ref Sensor related properties.
 */
class CommonKF {
private:
  /**
   * @brief wrapper that's used to queue tasks into the CKF's internal task
   * queue, based on queue time.
   */
  struct CorrectionTask {
    // queue time
    SteadyClock::time_point t;
    // task to complete
    std::function<void()> callback;
    // time-based sorting!
    bool operator>(const CorrectionTask &o) const { return t > o.t; }
  };

  // internal queue sorts on queue time to avoid predictions occurring
  // non-chronologically.
  std::priority_queue<CorrectionTask, std::vector<CorrectionTask>,
                      std::greater<CorrectionTask>>
      queue;

  // internal clock tracks time of most-recent accomplished task
  SteadyClock::time_point t_state;

  // how many steps did we complete?
  uint32_t steps = 0;

  // some data collection necessities
  std::map<std::string, Eigen::VectorXd> last_K;
  std::map<std::string, Eigen::VectorXd> last_residual;
  // signed per-state correction dx = K*d_r actually applied (keeps direction,
  // unlike the last_K row-norm)
  std::map<std::string, Eigen::VectorXd> last_dx;

  /**
   * @brief Corrects based on the basic Kalman blending step. uses Joseph form,
   * also prints a diagnostic message to the terminal.
   *
   * @param H Observation matrix
   * @param R Measurement noise matrix
   * @param d_r Measurement residual
   * @param name name of the sensor used to do the correction (Displays in the
   * @ref print_diagnostic panel)
   */
  void correct(const Eigen::MatrixXd &H, const Eigen::MatrixXd &R,
               const Eigen::VectorXd &d_r, const Eigen::VectorXd &z,
               std::string name = "") {
    // innovation covariance
    const Eigen::MatrixXd Pz = H * P * H.transpose();
    const Eigen::MatrixXd S = Pz + R;

    // Kalman gain
    const Eigen::MatrixXd K = P * H.transpose() * S.inverse();

    // Estimation update
    const Eigen::VectorXd dx = K * d_r;

    // calculate the Normalized Innovation Squared
    // (TODO: this may be the wrong metric to use... and might also have to be
    // sensor dependent...) we may have to switch it...
    nis = (d_r.transpose() * S.inverse() * d_r)(0, 0);

    // full-pipeline trace (emitted whether or not the update is applied)
    KFTrace tr;
    tr.t_s = std::chrono::duration<double>(t_state.time_since_epoch()).count();
    tr.sensor = name;
    tr.step = this->steps;
    tr.nis = nis;
    tr.z = z;
    tr.Hx = z - d_r; // h(x) reconstructed from residual (r = z - h(x))
    tr.r = d_r;
    tr.S_diag = S.diagonal();
    tr.R_diag = R.diagonal();
    tr.dx = dx;
    tr.P_diag = P.diagonal(); // pre-update, before the (I-KH) shrink below
    tr.x_pre = x;

    // stashed pre-gate so rejected updates still show their innovation
    last_residual[name] = d_r;

    // do a one-tailed chi-score test
    // double chi_sq_threshold = 100.0;
    // // lowered the ci from 95 -> 99.9
    // // double chi_sq_threshold = 0;
    // switch (d_r.size()) {
    // case 1:
    //   chi_sq_threshold = 10.828;
    //   break;
    // case 2:
    //   chi_sq_threshold = 13.816;
    //   break;
    // case 3:
    //   chi_sq_threshold = 16.266;
    //   break;
    // default:
    //   chi_sq_threshold = 100;
    // }
    // if (nis > chi_sq_threshold) {
    //   tr.applied = false;
    //   tr.x_post = x; // unchanged
    //   this->trace_step(tr);
    //   return; // probably didn't happen...
    // }

    // per-state gain magnitude of the applied correction (row-wise norm of K)
    last_K[name] = K.rowwise().norm();
    // signed correction actually applied to each state this step
    last_dx[name] = dx;

    // print the diagnostic to the termianl
    this->print_diagnostic(S, Pz, R, d_r, name);

    // Update the covariance matrix.
    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
    const Eigen::MatrixXd IKH = I - K * H;
    P = IKH * P * IKH.transpose() + K * R * K.transpose();

    // and appy the error...
    this->apply_error(dx);

    // full state after the correction, then emit the trace
    tr.applied = true;
    tr.x_post = x;
    this->trace_step(tr);

    // increment the error
    ++this->steps;
  }

  /**
   * @brief predict step of the Kalman Update
   *
   * @param target target time point to advance to.
   * @return true succesfully predicted
   * @return false no prediction occurred (time was late to arrive to queue).
   */
  bool advance_to(SteadyClock::time_point target) {
    double dt = std::chrono::duration<double>(target - t_state).count();
    if (dt < 0.0)
      return false; // already past it (late/equal arrival)

    // if my dt is really large, might be a good idea to do multiple predict
    // steps sequentially to reduce error...
    while (dt > MIN_PREDICT_INTVL) {
      this->predict(MIN_PREDICT_INTVL); // do a predict per millisecond
      dt -= MIN_PREDICT_INTVL;
      t_state += std::chrono::duration_cast<SteadyClock::duration>(
          std::chrono::duration<double>(MIN_PREDICT_INTVL));
    }
    if (dt > 0.0)
      this->predict(dt);

    // advance the current time
    this->t_state = target;
    return true;
  }

protected:
  /**
   * @brief Normalized innovation squared i.e., chi-score.
   * @warning Will probably be altered to another value, since the asynchronous
   * structure may invalidate the point of this variable.
   */
  double nis = 0.0;
  /**
   * @brief State covariance matrix.
   */
  Eigen::MatrixXd P;
  /**
   * @brief state variable.
   */
  Eigen::VectorXd x;

  /**
   * @brief Neat snapshot struct that allows users to just store fresh important
   * data outside the filter framework.
   */
  typedef struct snapshot {
    SteadyClock::time_point t;
    Eigen::VectorXd z;
    bool fresh{false};
  } snapshot;

  /**
   * @brief Construct a new CommonKF object
   *
   * @param P0 Initial covariance
   * @param x0 Initial state
   * @param t0 Initial time
   */
  CommonKF(const Eigen::MatrixXd &P0, const Eigen::VectorXd &x0,
           SteadyClock::time_point t0)
      : t_state(t0), P(P0), x(x0) {}

  /**
   * @brief Construct a new CommonKF object
   *
   * @param state_len length of the initial state.
   * @param cov_len lenght of the covariance matrix
   * @param t0 initial time
   */
  CommonKF(Eigen::Index state_len, Eigen::Index cov_len,
           SteadyClock::time_point t0)
      : t_state(t0), P(Eigen::MatrixXd::Zero(cov_len, cov_len)),
        x(Eigen::VectorXd::Zero(state_len)) {}

  /**
   * @brief Predict step of the Kalman Filter. Derived classes should implement
   * this.
   * @param dt time increment.
   */
  virtual void predict(double dt) = 0;
  /**
   * @brief Blending step of the Kalman Filter.
   *
   * @param dx Measurement residual
   */
  virtual void apply_error(const Eigen::VectorXd &dx) = 0;
  /**
   * @brief Simple print diagnostic panel that breaks out innovation and
   * covariance values for the user.
   *
   * @param S innovation covariance
   * @param Pz covariance mapped to the measurement space
   * @param R measurement noise
   * @param d_r measurement residual
   * @param name name of the sensor (title of the panel)
   */
  virtual void print_diagnostic(const Eigen::MatrixXd &S,
                                const Eigen::MatrixXd &Pz,
                                const Eigen::MatrixXd &R,
                                const Eigen::VectorXd &d_r,
                                std::string name) = 0;

  /**
   * @brief Full-pipeline trace sink, called once per correction step with
   * every intermediate quantity. Default is a no-op; a derived filter (or its
   * owner) can override / route it to a CSV. See @ref KFTrace.
   */
  virtual void trace_step(const KFTrace &tr) {
    if (trace_sink)
      trace_sink(tr);
  }

public:
  /**
   * @brief Optional external sink for the full per-step trace. Set this from
   * the owning node to stream every correction to a CSV. See @ref KFTrace.
   */
  std::function<void(const KFTrace &)> trace_sink;

protected:

  /**
   * @brief Derived classes implement a form of this in the @ref SensorTable and
   * is required by the filter.
   */
  // Fills H (observation), r (residual = z - h(x)), and z (raw measurement).
  // z lets the trace log the true measurement and reconstruct Hx = z - r.
  using ComputeHrFn = std::function<bool(
      Eigen::MatrixXd &H, Eigen::VectorXd &r, Eigen::VectorXd &z)>;

  /**
   * @brief Pushes a task to be completed by the filter into the queue.
   *
   * @param t queue time
   * @param callback task to complete
   */
  void queue_task(SteadyClock::time_point t, std::function<void()> callback) {
    queue.push({t, std::move(callback)});
  }

  /**
   * @brief Pushes a filtering task into the queue.
   *
   * @param t queue time
   * @param R Measurement noise
   * @param compute_Hr function used to get the observation and measurement
   * residual off of the derived class
   * @param name sensor id
   */
  void queue_correction(SteadyClock::time_point t, const Eigen::MatrixXd &R,
                        ComputeHrFn compute_Hr, std::string name) {

    // queue a task with a callback that executes the Kalman steps
    queue_task(t, [this, R, name = std::move(name),
                   compute_Hr = std::move(compute_Hr)]() mutable {
      Eigen::MatrixXd H;
      Eigen::VectorXd dr;
      Eigen::VectorXd z;

      if (!compute_Hr(H, dr, z))
        return;

      this->correct(H, R, dr, z, name);
    });
  }

public:
  /**
   * @brief Get the covariance matrix.
   *
   * @return Eigen::MatrixXd
   */
  Eigen::MatrixXd get_cov_matrix() const { return this->P; }
  virtual ~CommonKF() = default;

  /**
   * @brief Steps the CommonKF. MP compatible.
   *
   * @param now current time step.
   */
  void tick(SteadyClock::time_point now) {
    // process the entire queue build up until I'm done.
    while (!queue.empty() && queue.top().t <= now) {
      CorrectionTask task = queue.top(); // get that corrective task
      queue.pop();
      if (!advance_to(task.t))
        continue;      // predict up to the measurement time...
      task.callback(); // ...then correct
    }

    advance_to(now);
  }

  /**
   * @brief Some Helper function that gets the gains because I need to debug.
   * Last Kalman Gain computed.
   *
   * @param name name of the filter
   * @param n length of the state.
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd last_gain(const std::string &name, Eigen::Index n) const {
    auto it = last_K.find(name);
    return it == last_K.end() ? Eigen::VectorXd::Zero(n).eval() : it->second;
  }

  /**
   * @brief Signed per-state correction dx = K*d_r last applied by this sensor.
   * Unlike @ref last_gain (a row-norm), this keeps the direction of the nudge.
   *
   * @param name sensor id
   * @param n length of the state
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd last_correction(const std::string &name,
                                  Eigen::Index n) const {
    auto it = last_dx.find(name);
    return it == last_dx.end() ? Eigen::VectorXd::Zero(n).eval() : it->second;
  }

  /**
   * @brief Gets the last innovation covariance matrix computed.
   *
   * @param name sensor id
   * @param m length of the measurement residual
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd last_innovation(const std::string &name,
                                  Eigen::Index m) const {
    auto it = last_residual.find(name);
    return it == last_residual.end() ? Eigen::VectorXd::Zero(m).eval()
                                     : it->second;
  }

  /**
   * @brief Number of steps completed
   *
   * @return uint32_t
   */
  uint32_t num_steps() const { return steps; }
  /**
   * @brief Gets the covariance matrix
   *
   * @return const Eigen::MatrixXd&
   */
  const Eigen::MatrixXd &covariance() const { return P; }
  /**
   * @brief Gets the normalized innovation squared for the filter.
   * @warning maybe not a good measurement of filter health, or performance.
   *
   * @return double
   */
  double last_nis() const { return nis; }
  /**
   * @brief Get the current estimate.
   *
   * @return Eigen::VectorXd
   */
  Eigen::VectorXd get_curr_state() const { return x; }
};
} // namespace ckf
