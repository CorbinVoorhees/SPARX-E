#ifndef EKF_HPP
#define EKF_HPP

#include "../../utils.h"
#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

inline static const Eigen::Vector3d grav_vec{0.0, 0.0, 9.81};
inline static Eigen::Matrix3d       skew(const Eigen::Vector3d &v) {
  Eigen::Matrix3d S;
  S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return S;
}

using SteadyClock = std::chrono::steady_clock;

namespace ekf {

class ISensor {
private:
  Eigen::MatrixXd H;
  Eigen::MatrixXd R;
  Eigen::VectorXd r;

public:
  virtual ~ISensor()                        = default;
  virtual void        update_HRr_matrices() = 0;
  virtual bool        has_fresh()           = 0;
  virtual std::string get_name()            = 0;

  auto &get_H() { return this->H; }
  auto &get_R() { return this->R; }
  auto &get_r() { return this->r; }
};

template <typename T> class Sensor final : public ISensor {
private:
  FilteredSampleProducer<T>                       &prod;
  std::function<void(FilteredSampleProducer<T> &)> update_HRr;
  std::string                                      name;

public:
  Sensor(const T &init, double tau, std::function<void(FilteredSampleProducer<T> &)> update_HRr_fn, std::string name_in = "")
      : prod{init, tau}, update_HRr(std::move(update_HRr_fn)), name(name_in) {};

  Sensor(FilteredSampleProducer<T> &producer, std::function<void(FilteredSampleProducer<T> &)> update_HRr_fn, std::string name_in = "")
      : prod(producer), update_HRr(std::move(update_HRr_fn)), name(name_in) {};

  const FilteredSampleProducer<T> &producer() const { return prod; }
  FilteredSampleProducer<T>       &producer() { return prod; }

  void        put(const T &data) { this->prod.put(data); }
  void        update_HRr_matrices() override { this->update_HRr(this->prod); }
  bool        has_fresh() override { return this->producer().has_fresh(); }
  std::string get_name() override { return this->name; }
};

class EKF {
private:
  std::vector<std::shared_ptr<ISensor>> sensors;
  SteadyClock::time_point               last_called_time;
  uint32_t                              steps = 0;

protected:
  double          nis;
  Eigen::MatrixXd P;
  Eigen::MatrixXd W;
  Eigen::VectorXd x;

  explicit EKF(const Eigen::MatrixXd &P0, const Eigen::MatrixXd &W0, const Eigen::VectorXd &x0, SteadyClock::time_point t0)
      : nis(0.0), last_called_time(t0), P(P0), W(W0), x(x0) {}
  virtual void    add_sensor(const std::shared_ptr<ISensor> &s) final { this->sensors.emplace_back(s); }
  Eigen::VectorXd correct(const Eigen::MatrixXd &H, const Eigen::MatrixXd &R, const Eigen::VectorXd &d_r, std::string name = "") {
    const Eigen::MatrixXd Pz = H * this->P * H.transpose();
    const Eigen::MatrixXd S  = Pz + R;
    const Eigen::MatrixXd K  = this->P * H.transpose() * S.inverse();
    const Eigen::VectorXd dx = K * d_r;

    this->nis = (d_r.transpose() * S.inverse() * d_r)(0, 0);

    // if (this->nis > 14.16)
    //   return Eigen::VectorXd::Zero(dx.rows());

    this->print_diagnostic(S, Pz, R, d_r, name);

    const Eigen::MatrixXd I   = Eigen::MatrixXd::Identity(this->P.rows(), this->P.cols());
    const Eigen::MatrixXd IKH = I - K * H;

    this->P = IKH * this->P * IKH.transpose() + K * R * K.transpose();
    return dx;
  }
  virtual void predict_timepoint(SteadyClock::time_point t) final {
    double dt              = std::chrono::duration<double>(t - this->last_called_time).count();
    this->last_called_time = t;
    this->predict(dt);
  }

  virtual void predict(double dt)                 = 0;
  virtual void apply_error(Eigen::VectorXd dx)    = 0;
  virtual void print_diagnostic(const Eigen::MatrixXd &S, const Eigen::MatrixXd &Pz, const Eigen::MatrixXd &R, const Eigen::VectorXd &d_r,
                                std::string name) = 0;

public:
  virtual uint32_t num_steps() final { return this->steps; }
  virtual void     step(SteadyClock::time_point t) final {
    double dt              = std::chrono::duration<double>(t - this->last_called_time).count();
    this->last_called_time = t;

    this->predict(dt);

    for (auto &s : sensors) {
      if (!s->has_fresh())
        continue;

      s->update_HRr_matrices();
      const Eigen::VectorXd dx = correct(s->get_H(), s->get_R(), s->get_r(), s->get_name());
      this->apply_error(dx);
      this->steps++;
    }
  }

  virtual const Eigen::MatrixXd &covariance() const final { return this->P; }
  virtual double                 last_nis() const final { return this->nis; }
  virtual Eigen::VectorXd        get_curr_state() const final { return this->x; }
};
}; // namespace ekf
#endif // EKF_HPP