#pragma once

#include "sensor.hpp"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <iterator>
#include <optional>
#include <poll.h>
#include <string>
#include <system_error>
#include <termios.h>
#include <type_traits>
#include <unistd.h>
#include <utility>

namespace Sensor {

struct SerialConfig {
  std::string device;
  speed_t baud = B115200;
  int poll_timeout_ms = -1;
};

template <typename Tag, typename InputType, typename OutputType>
class ISerialSensor : public ISensor<Tag, InputType, OutputType> {
private:
  inline static constexpr std::size_t read_buffer_size = 4096;

  int fd = -1;
  SerialConfig config;

  int wait(short events, int timeout_ms = -1) const noexcept {
    pollfd descriptor{fd, events, 0};
    const int timeout = timeout_ms < 0 ? config.poll_timeout_ms : timeout_ms;

    int result;
    do {
      result = ::poll(&descriptor, 1, timeout);
    } while (result < 0 && errno == EINTR);

    if (result <= 0)
      return result;

    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      errno = EIO;
      return -1;
    }

    return (descriptor.revents & events) != 0 ? 1 : 0;
  }

  ssize_t read_some(void *destination, std::size_t capacity) noexcept {
    ssize_t result;
    do {
      result = ::read(fd, destination, capacity);
    } while (result < 0 && errno == EINTR);
    return result;
  }

  int write_all(const void *source, std::size_t size) noexcept {
    const auto *bytes = static_cast<const std::uint8_t *>(source);
    std::size_t written = 0;

    while (written < size) {
      if (wait(POLLOUT) <= 0)
        return -1;

      ssize_t result;
      do {
        result = ::write(fd, bytes + written, size - written);
      } while (result < 0 && errno == EINTR);

      if (result < 0) {
        if (errno == EAGAIN) // same as blocking error
          continue;
        return -1;
      }

      written += static_cast<std::size_t>(result);
    }

    return 0;
  }

protected:
  virtual std::optional<InputType> convert(const void *, std::size_t) noexcept {
    return std::nullopt;
  }

public:
  explicit ISerialSensor(SerialConfig input_config)
      : config(std::move(input_config)) {
    fd = ::open(config.device.c_str(),
                O_RDWR | O_NOCTTY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0)
      throw std::system_error(errno, std::generic_category(),
                              "open " + config.device);

    termios tty{};
    if (::tcgetattr(fd, &tty) < 0) {
      const int error = errno;
      ::close(fd);
      fd = -1;
      throw std::system_error(error, std::generic_category(), "tcgetattr");
    }

    ::cfmakeraw(&tty);
    ::cfsetispeed(&tty, config.baud);
    ::cfsetospeed(&tty, config.baud);

    tty.c_cflag &= ~(CSIZE | PARENB | CSTOPB | CRTSCTS);
    tty.c_cflag |= CS8 | CLOCAL | CREAD;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;

    if (::tcsetattr(fd, TCSANOW, &tty) < 0) {
      const int error = errno;
      ::close(fd);
      fd = -1;
      throw std::system_error(error, std::generic_category(), "tcsetattr");
    }

    ::tcflush(fd, TCIOFLUSH);
  }

  // yeah? you can't get around me.
  ISerialSensor() = delete;
  ISerialSensor(const ISerialSensor &) = delete;
  ISerialSensor &operator=(const ISerialSensor &) = delete;
  ISerialSensor(ISerialSensor &&) = delete;
  ISerialSensor &operator=(ISerialSensor &&) = delete;

  int read(InputType &input) noexcept override {
    if constexpr (std::is_same_v<InputType, std::monostate>) {
      return 0;
    } else {
      const int ready = wait(POLLIN);
      if (ready <= 0)
        return ready;

      std::array<std::uint8_t, read_buffer_size> bytes;
      const ssize_t count = read_some(bytes.data(), bytes.size());
      if (count < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
          return 0;
        return -1;
      }
      if (count == 0)
        return 0;

      std::optional<InputType> converted =
          convert(bytes.data(), static_cast<std::size_t>(count));
      if (!converted.has_value())
        return 0;

      input = std::move(*converted);
      return 1;
    }
  }

  int write(OutputType &output) noexcept override {
    if constexpr (std::is_same_v<OutputType, std::monostate>) {
      return 0;
    } else {
      int result;
      if constexpr (std::is_trivially_copyable_v<OutputType>) {
        result = write_all(&output, sizeof(output));
      } else {
        const auto *data = std::data(output);
        result =
            write_all(data, std::size(output) *
                                sizeof(std::remove_pointer_t<decltype(data)>));
      }

      if (result < 0)
        return result;

      do {
        result = ::tcdrain(fd);
      } while (result < 0 && errno == EINTR);
      return result;
    }
  }

  ~ISerialSensor() override {
    if (fd >= 0)
      ::close(fd);
  }
};

} // namespace Sensor
