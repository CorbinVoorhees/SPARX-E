#include "../../utils.h"
#include <nav_filters/sensor/data_prefilter.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

namespace {

class CalibrationFile {
public:
  explicit CalibrationFile(const std::string &contents) {
    const auto suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("sparxe_calibration_" + std::to_string(suffix) + ".txt");
    std::ofstream out(path);
    out << contents;
  }

  ~CalibrationFile() { std::filesystem::remove(path); }

  std::string filename() const { return path.string(); }

private:
  std::filesystem::path path;
};

TEST(DataPrefilterTest, AppliesBiasThenFullMatrix) {
  const CalibrationFile calibration(
      "1 2 3\n"
      "1 2 0\n"
      "0 1 3\n"
      "4 0 1\n");

  Sensor::DataPrefilter<Eigen::Vector3d> producer(
      {Eigen::Vector3d::Zero(), 0, calibration.filename()});
  producer.put(Eigen::Vector3d(2.0, 4.0, 6.0));

  EXPECT_TRUE(producer.peek().data.isApprox(
      Eigen::Vector3d(5.0, 11.0, 7.0), 1e-12));
}

TEST(DataPrefilterTest, RecordedRestSampleKeepsGravityMagnitude) {
  const CalibrationFile calibration(
      "-1.884560533848281304e-03 1.125813625003250704e-02 "
      "6.331115305596568543e-02\n"
      "1.001392117494123823e+00 3.558085168311501956e-02 "
      "-1.554752250580663164e-03\n"
      "0 1.000900255498943459e+00 7.543068692760565705e-04\n"
      "0 0 1.001503898533010695e+00\n");

  Sensor::DataPrefilter<Eigen::Vector3d> producer(
      {Eigen::Vector3d::Zero(), 0, calibration.filename()});
  producer.put(
      Eigen::Vector3d(0.15764686899749203, -1.0456998481451723,
                      9.795763687964696));

  const Eigen::Vector3d expected(0.10701449826488466, -1.050568260830824,
                                 9.747089155998577);
  EXPECT_TRUE(producer.peek().data.isApprox(expected, 1e-12));
  EXPECT_NEAR(producer.peek().data.norm(), 9.804126314388704, 1e-12);
}

TEST(DataPrefilterTest, MissingCalibrationIsNoOp) {
  const std::string missing =
      (std::filesystem::temp_directory_path() /
       "sparxe_definitely_missing_calibration.txt")
          .string();
  std::filesystem::remove(missing);

  Sensor::DataPrefilter<Eigen::Vector3d> producer(
      {Eigen::Vector3d::Zero(), 0, missing});
  const Eigen::Vector3d sample(1.0, -2.0, 3.0);
  producer.put(sample);

  EXPECT_TRUE(producer.peek().data.isApprox(sample, 1e-12));
}

TEST(DataPrefilterTest, CalibratesScalarInput) {
  const CalibrationFile calibration(
      "2\n" // bias
      "3\n" // scale
  );
  Sensor::DataPrefilter<double> prefilter(
      {0.0, 0, calibration.filename()});

  const std::optional<Sensor::DataPrefilter<double>::Stamped> filtered =
      prefilter.put(5.0);
  ASSERT_TRUE(filtered.has_value());
  EXPECT_DOUBLE_EQ(filtered->data, 9.0);
}

TEST(DataPrefilterTest, IdentityCalibrationSupportsOtherEigenVectorSizes) {
  Sensor::DataPrefilter<Eigen::Vector4d> prefilter(
      {Eigen::Vector4d::Zero(), 0, ""});
  const Eigen::Vector4d input(1.0, 2.0, 3.0, 4.0);

  const std::optional<Sensor::DataPrefilter<Eigen::Vector4d>::Stamped> filtered =
      prefilter.put(input);
  ASSERT_TRUE(filtered.has_value());
  EXPECT_TRUE(filtered->data.isApprox(input, 1e-12));
}

TEST(DataPrefilterTest, ReleasesStampedDataAfterConfiguredMinimum) {
  Sensor::DataPrefilter<double> prefilter({0.0, 1, ""});
  const SteadyClock::time_point first_time = SteadyClock::now();
  const SteadyClock::time_point second_time =
      first_time + std::chrono::milliseconds(10);

  EXPECT_FALSE(prefilter.put(1.0, first_time).has_value());

  const std::optional<Sensor::DataPrefilter<double>::Stamped> released =
      prefilter.put(2.0, second_time);
  ASSERT_TRUE(released.has_value());
  EXPECT_DOUBLE_EQ(released->data, 2.0);
  EXPECT_EQ(released->timestamp, second_time);

  const Sensor::DataPrefilter<double>::Stamped latest = prefilter.peek();
  EXPECT_DOUBLE_EQ(latest.data, released->data);
  EXPECT_EQ(latest.timestamp, released->timestamp);

  const Eigen::MatrixXd covariance =
      Sensor::DataPrefilter<double>::covariance(released->variance);
  ASSERT_EQ(covariance.rows(), 1);
  ASSERT_EQ(covariance.cols(), 1);
  EXPECT_DOUBLE_EQ(covariance(0, 0), released->variance);
  EXPECT_DOUBLE_EQ(released->mean, 1.0);
  EXPECT_DOUBLE_EQ(released->variance, 0.0);
  EXPECT_EQ(released->count, 2U);
}

TEST(DataPrefilterTest, FreezesWelfordVarianceAfterInitialization) {
  Sensor::DataPrefilter<double> prefilter({0.0, 4, ""});

  EXPECT_FALSE(prefilter.put(1.0).has_value());
  EXPECT_FALSE(prefilter.put(2.0).has_value());
  EXPECT_FALSE(prefilter.put(3.0).has_value());
  EXPECT_FALSE(prefilter.put(4.0).has_value());

  const std::optional<Sensor::DataPrefilter<double>::Stamped> released =
      prefilter.put(100.0);

  ASSERT_TRUE(released.has_value());
  EXPECT_DOUBLE_EQ(released->mean, 2.5);
  EXPECT_DOUBLE_EQ(released->variance, 1.25);

  const std::optional<Sensor::DataPrefilter<double>::Stamped> next =
      prefilter.put(-100.0);
  ASSERT_TRUE(next.has_value());
  EXPECT_DOUBLE_EQ(next->mean, 2.5);
  EXPECT_DOUBLE_EQ(next->variance, 1.25);
}

TEST(StartFrameAccelerationTest, CancelsGravityAtAnyRelativeOrientation) {
  const Eigen::Vector3d gravity_start(0.107, -1.051, 9.747);

  EXPECT_TRUE(linear_accel_in_start_frame(Eigen::Quaterniond::Identity(),
                                          gravity_start, gravity_start)
                  .isZero(1e-12));

  const Eigen::Quaterniond start_from_body(
      Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitY()));
  const Eigen::Vector3d gravity_body =
      start_from_body.conjugate() * gravity_start;

  EXPECT_TRUE(linear_accel_in_start_frame(start_from_body, gravity_body,
                                          gravity_start)
                  .isZero(1e-12));
}

} // namespace
