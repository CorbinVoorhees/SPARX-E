#pragma once

#include "../sensor/serial_sensor.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <variant>

namespace Sensor {

struct UwbTag : ISensorTag {};

struct UwbConfig {
  SerialConfig serial{"/dev/ttyACM1", B115200, 200};
};

class UwbSensor final : public ISerialSensor<UwbTag, double, std::monostate> {
private:
  static constexpr std::size_t maximum_buffer_size = 4096;
  static constexpr const char *range_prefix = "range_m=";

  std::string receive_buffer;

  static bool parse_line(const std::string &line, double &range) noexcept {
    const std::size_t prefix = line.find(range_prefix);
    if (prefix == std::string::npos)
      return false;

    const char *begin =
        line.c_str() + prefix + std::char_traits<char>::length(range_prefix);
    char *end = nullptr;
    errno = 0;
    const double value = std::strtod(begin, &end);
    if (begin == end || errno == ERANGE || !std::isfinite(value))
      return false;

    range = value;
    return true;
  }

  bool parse_buffer(double &range) {
    for (;;) {
      const std::size_t newline = receive_buffer.find('\n');
      if (newline == std::string::npos)
        return false;

      const std::string line = receive_buffer.substr(0, newline);
      receive_buffer.erase(0, newline + 1);
      if (parse_line(line, range))
        return true;
    }
  }

protected:
  std::optional<double>
  convert(const void *bytes, std::size_t size) noexcept override {
    receive_buffer.append(static_cast<const char *>(bytes), size);
    if (receive_buffer.size() > maximum_buffer_size) {
      const std::size_t newline = receive_buffer.rfind('\n');
      if (newline == std::string::npos)
        receive_buffer.clear();
      else
        receive_buffer.erase(0, newline + 1);
    }

    double range;
    if (!parse_buffer(range))
      return std::nullopt;
    return range;
  }

public:
  explicit UwbSensor(UwbConfig config = {})
      : ISerialSensor(std::move(config.serial)) {}
};

} // namespace Sensor
