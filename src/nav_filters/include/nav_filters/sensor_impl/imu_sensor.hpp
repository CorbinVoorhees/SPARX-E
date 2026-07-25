#include "../sensor/sensor.hpp"

namespace Sensor {

struct IMUTag : ISensorTag {};
// imu gives me accel, gyro, magnm so its a vector 9
using V9 = Eigen::Vector<double, 9>;
class IMUSensor : public ISensor<IMUTag, V9, std::monostate> {
  // no write
public:
  int read(V9 &in) noexcept override { return 0; }
  // we implement read though
};

}; // namespace Sensor
