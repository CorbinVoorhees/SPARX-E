#pragma once

#include "../sensor/serial_sensor.hpp"

#include <string>

namespace Sensor {

struct OdometryTag : ISensorTag {};

class OdometrySensor final
    : public ISerialSensor<OdometryTag, std::monostate, std::string> {
public:
  explicit OdometrySensor(
      SerialConfig config = {"/dev/ttyACM0", B115200, -1})
      : ISerialSensor(std::move(config)) {}
};

} // namespace Sensor
