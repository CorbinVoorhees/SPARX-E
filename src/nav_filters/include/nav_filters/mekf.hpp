#ifndef MEKF_HPP
#define MEKF_HPP

using namespace ekf;
static const Eigen::Vector3d gravity(0, 0, 9.81);
class MEKF : public EKF {

private:
  // create sensor hooks
  std::shared_ptr<Sensor<Eigen::Vector3d>> gyros;
  std::shared_ptr<Sensor<Eigen::Vector3d>> accel; // 3 input, 3 inertial frame accel, 3 bias
  std::shared_ptr<Sensor<Eigen::Vector3d>> magnm;

  // state getters
  auto q() { return this->x.segment<4>(0); }
  auto bg() { return this->x.segment<3>(4); }
  auto bm() { return this->x.segment<3>(7); }

  // Out-of-the-box characteristics
  Eigen::Matrix3d Qw;
  Eigen::Matrix3d Qbw;
  Eigen::Matrix3d Qbm;

  void apply_error(Eigen::VectorXd err) override {
    const Eigen::Vector3d dtheta = err.segment<3>(0);
    const Eigen::Vector3d dbg    = err.segment<3>(3);
    const Eigen::Vector3d dbm    = err.segment<3>(6);

    const Eigen::Vector4d o = this->q();
    Eigen::Quaterniond    q_curr(o(0), o(1), o(2), o(3));
    Eigen::Quaterniond    dq(1.0, 0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z());
    Eigen::Quaterniond    q_new = q_curr * dq;
    q_new.normalize();

    if (q_new.w() < 0.0)
      q_new.coeffs() *= -1.0;

    this->q() << q_new.w(), q_new.x(), q_new.y(), q_new.z();
    this->bg() += dbg;
    this->bm() += dbm;
  }
  void print_diagnostic(const Eigen::MatrixXd &S, const Eigen::MatrixXd &Pz, const Eigen::MatrixXd &R, const Eigen::VectorXd &d_r,
                        std::string name) override {
    if (!this->should_print_diagnostic)
      return;

    using clock            = std::chrono::steady_clock;
    static auto last_print = clock::now();
    static bool first_run  = true;

    const auto now        = clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count();

    if (elapsed_ms < 1000)
      return;

    last_print = now;

    const double Sx = S(0, 0);
    const double Sy = S(1, 1);
    const double Sz = S(2, 2);

    const double Pzx = Pz(0, 0);
    const double Pzy = Pz(1, 1);
    const double Pzz = Pz(2, 2);

    const double Rx = R(0, 0);
    const double Ry = R(1, 1);
    const double Rz = R(2, 2);

    const double nrx = d_r(0) / std::sqrt(Sx);
    const double nry = d_r(1) / std::sqrt(Sy);
    const double nrz = d_r(2) / std::sqrt(Sz);

    const Eigen::Vector4d q_curr  = this->q();
    const Eigen::Vector3d bg_curr = this->bg();
    const Eigen::Vector3d bm_curr = this->bm();

    Eigen::Vector3d         w_raw;
    SteadyClock::time_point t;
    this->gyros->producer().take(w_raw, t);

    const Eigen::Vector3d w_corrected = w_raw - bg_curr;
    const Eigen::Vector3d w_mean      = this->gyros->producer().get_mean();

    Eigen::Quaterniond q_obj(q_curr(0), q_curr(1), q_curr(2), q_curr(3));
    q_obj.normalize();

    const Eigen::Matrix3d Rot = q_obj.toRotationMatrix();

    const double roll  = std::atan2(Rot(2, 1), Rot(2, 2)) * 180.0 / M_PI;
    const double pitch = std::asin(std::max(-1.0, std::min(1.0, -Rot(2, 0)))) * 180.0 / M_PI;
    const double yaw   = std::atan2(Rot(1, 0), Rot(0, 0)) * 180.0 / M_PI;

    if (!first_run) {
      std::cout << "\033[19A";
    } else {
      first_run = false;
    }

    std::cout << string_format("\033[2K============================= %s DASHBOARD =============================\n"
                               "\033[2KNIS  : %9.4f (Expected Mean: %d)\n"
                               "\033[2K--------------------------------------------------------------------------\n"
                               "\033[2KAttitude (Euler Angles)          | State Trajectory Estimates\n"
                               "\033[2K  Roll : %8.3f deg               |   q  : [%.4f, %.4f, %.4f, %.4f]\n"
                               "\033[2K  Pitch: %8.3f deg               |   bg : [%.6f, %.6f, %.6f] rad/s\n"
                               "\033[2K  Yaw  : %8.3f deg               |   bm : [%.6f, %.6f, %.6f]\n"
                               "\033[2K--------------------------------------------------------------------------\n"
                               "\033[2KGyroscope Channels (rad/s)       | Measurement Innovation Residuals\n"
                               "\033[2K  w raw  : [%.5f, %.5f, %.5f]  |   r  : [%.6f, %.6f, %.6f]\n"
                               "\033[2K  w-bg   : [%.5f, %.5f, %.5f]  |   nr : [%.4f, %.4f, %.4f]\n"
                               "\033[2K  w mean : [%.5f, %.5f, %.5f]  |\n"
                               "\033[2K--------------------------------------------------------------------------\n"
                               "\033[2KUncertainty & Sensor Noise Balances (Axis: X, Y, Z)\n"
                               "\033[2K  S  total innov var :  %.4e,  %.4e,  %.4e\n"
                               "\033[2K  Pz state contrib   :  %.4e,  %.4e,  %.4e  (%5.1f%%, %5.1f%%, %5.1f%%)\n"
                               "\033[2K  R  sensor noise    :  %.4e,  %.4e,  %.4e  (%5.1f%%, %5.1f%%, %5.1f%%)\n"
                               "\033[2K==========================================================================\n",
                               name.c_str(), this->last_nis(), (int)d_r.rows(),

                               roll, q_curr(0), q_curr(1), q_curr(2), q_curr(3), pitch, bg_curr(0), bg_curr(1), bg_curr(2), yaw, bm_curr(0),
                               bm_curr(1), bm_curr(2),

                               w_raw.x(), w_raw.y(), w_raw.z(), d_r(0), d_r(1), d_r(2),

                               w_corrected.x(), w_corrected.y(), w_corrected.z(), nrx, nry, nrz,

                               w_mean.x(), w_mean.y(), w_mean.z(),

                               Sx, Sy, Sz, Pzx, Pzy, Pzz, 100.0 * Pzx / Sx, 100.0 * Pzy / Sy, 100.0 * Pzz / Sz, Rx, Ry, Rz, 100.0 * Rx / Sx,
                               100.0 * Ry / Sy, 100.0 * Rz / Sz)
              << std::flush;
  }

  void predict(double dt) override {
    if (dt <= 0.0)
      return;

    Eigen::Vector3d         w_corrected;
    SteadyClock::time_point time;
    this->gyros->producer().take(w_corrected, time);
    w_corrected -= this->bg();

    const Eigen::Vector4d    o = this->q();
    const Eigen::Quaterniond prior_q(o(0), o(1), o(2), o(3));
    const Eigen::Quaterniond omega_quat(0.0, w_corrected.x(), w_corrected.y(), w_corrected.z());

    Eigen::Quaterniond q_dot;
    q_dot.coeffs() = 0.5 * (prior_q * omega_quat).coeffs();

    Eigen::Quaterniond q_pred;
    q_pred.coeffs() = prior_q.coeffs() + q_dot.coeffs() * dt;
    q_pred.normalize();

    if (q_pred.w() < 0.0)
      q_pred.coeffs() *= -1.0;

    this->q() << q_pred.w(), q_pred.x(), q_pred.y(), q_pred.z();

    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Zero();
    F.block<3, 3>(0, 0)           = -skew(w_corrected);
    F.block<3, 3>(0, 3)           = -Eigen::Matrix3d::Identity();

    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;

    Eigen::Matrix<double, 9, 9> Qk = Eigen::Matrix<double, 9, 9>::Zero();

    Qk.block<3, 3>(0, 0) = this->Qw * dt + this->Qbw * (dt3 / 3.0);
    Qk.block<3, 3>(0, 3) = -this->Qbw * (dt2 / 2.0);
    Qk.block<3, 3>(3, 0) = -this->Qbw * (dt2 / 2.0);
    Qk.block<3, 3>(3, 3) = this->Qbw * dt;
    Qk.block<3, 3>(6, 6) = this->Qbm * dt;

    const Eigen::Matrix<double, 9, 9> Phi = Eigen::Matrix<double, 9, 9>::Identity() + F * dt;

    this->P = Phi * this->P * Phi.transpose() + Qk;
  }

public:
  bool should_print_diagnostic = false;

  MEKF(const Eigen::VectorXd &x0, const Eigen::Matrix<double, 9, 9> &P0, const Eigen::Matrix3d &Q_gyro, const Eigen::Matrix3d &Q_bias_gyro,
       const Eigen::Matrix3d &Q_bias_magnm, const Eigen::Matrix3d &R_accel, const Eigen::Matrix3d &R_magnm, // <-- ADDED R MATRICES HERE
       SteadyClock::time_point t0, const Eigen::Vector3d &mean_north, FilteredSampleProducer<Eigen::Vector3d> &gyros_producer,
       FilteredSampleProducer<Eigen::Vector3d> &accel_producer, FilteredSampleProducer<Eigen::Vector3d> &magnm_producer)
      : EKF(P0, Eigen::MatrixXd::Zero(P0.rows(), P0.cols()), x0, t0), Qbw(Q_bias_gyro), Qw(Q_gyro), Qbm(Q_bias_magnm) {

    this->gyros =
        std::make_shared<Sensor<Eigen::Vector3d>>(gyros_producer, [this](FilteredSampleProducer<Eigen::Vector3d> &p) {}, "Gyroscope");

    this->accel = std::make_shared<Sensor<Eigen::Vector3d>>(
        accel_producer,
        [this, R_accel](FilteredSampleProducer<Eigen::Vector3d> &p) { // <-- CAPTURE R_accel BY VALUE
          // update the H,R, and measurment error vector (r) here.
          // get the rotation matrix from the orientation quat
          const Eigen::Vector4d    o = q();
          const Eigen::Quaterniond orientation(o(0), o(1), o(2), o(3));
          const Eigen::Vector3d    a_measured = p.peek();

          // rotate gravity vector
          const Eigen::Vector3d grav_body = orientation.normalized().toRotationMatrix().transpose() * gravity;

          // update HRr
          this->accel->get_H()                   = Eigen::MatrixXd::Zero(3, 9);
          this->accel->get_H().block<3, 3>(0, 0) = skew(grav_body);
          this->accel->get_R()                   = R_accel; // <-- ASSIGN HERE
          this->accel->get_r()                   = a_measured - grav_body;
        },
        "Accelerometer");

    this->magnm = std::make_shared<Sensor<Eigen::Vector3d>>(
        magnm_producer,
        [this, mean_north, R_magnm](FilteredSampleProducer<Eigen::Vector3d> &p) { // <-- CAPTURE R_magnm BY VALUE
          // update the H,R, and measurment error vector (r) here.
          const Eigen::Vector4d    o = this->q();
          const Eigen::Quaterniond orientation(o(0), o(1), o(2), o(3));
          const Eigen::Vector3d    m_measured = p.peek();

          // rotate magnetometer vector
          const Eigen::Vector3d magnetic_body = orientation.normalized().toRotationMatrix().transpose() * mean_north;

          // update HRr
          this->magnm->get_H()                   = Eigen::MatrixXd::Zero(3, 9);
          this->magnm->get_H().block<3, 3>(0, 0) = skew(magnetic_body);
          this->magnm->get_H().block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
          this->magnm->get_R()                   = R_magnm; // <-- ASSIGN HERE
          this->magnm->get_r()                   = m_measured - this->bm() - magnetic_body;
        },
        "Magnetometer");

    this->add_sensor(this->accel);
    this->add_sensor(this->magnm);
  }

  void put_gyros(Eigen::Vector3d data) { this->gyros->put(data); }
  void put_accel(Eigen::Vector3d data) { this->accel->put(data); }
  void put_magnm(Eigen::Vector3d data) { this->magnm->put(data); }

  Eigen::Quaterniond orientation() const { return Eigen::Quaterniond(x(0), x(1), x(2), x(3)).normalized(); }
  Eigen::Vector3d    gyro_bias() const { return x.segment<3>(4); }
};

#endif