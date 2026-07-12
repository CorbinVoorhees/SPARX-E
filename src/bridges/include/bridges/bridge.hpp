#pragma once

#include <cassert>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <termios.h>
#include <unistd.h>
#include <vector>

#include "../../../utils.h"
#include <rclcpp/rclcpp.hpp>
#include <rcpputils/asserts.hpp>
#include <std_msgs/msg/string.hpp>

/**
 * @brief Bridge is a base class that defines serial monitor communication for
 * peripherals that use TTY.
 *
 * @author Krish Sridhar
 */
class BridgeBase : public rclcpp::Node {

protected:
  // Initialize shared pointer of subscription object string message
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr
      serial_monitor_subscription;

  // Initialize serial file descriptor.
  int serial_fd;

  // String determining serial port address.
  std::string serial_port;

  // Actual baud rate of the Arduino Serial Monitor.
  speed_t baud_rate;

  // buffer initialization
  struct Buffer
  {
    size_t length;
    size_t idx = 0;
    std::vector<char> buffer;
  } buf;

  // target character
  static constexpr char target = '\n';
  
private:
  void open_serial_port() {
    // initialize termial communication protocol struct and zero-out struct
    // bytes
    struct termios tty;
    memset(&tty, 0, sizeof tty);

    // initialize the serial file descriptor, and catastrophically fail if
    // connection fails
    serial_fd = open(this->serial_port.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    rcpputils::assert_true(
        this->serial_fd > 0,
        string_format("Failed to open Connection Port! Open returned code %d", serial_fd));

    // force checking config attr validity
    rcpputils::assert_true(
        tcgetattr(this->serial_fd, &tty) == 0,
        string_format("Error from tcgetattr: %s", std::strerror(errno)));

    // set the speed from the baud rate in thte tty termois struct.
    // we'll use these configs to read from the terminal.
    cfsetospeed(&tty, this->baud_rate);
    cfsetispeed(&tty, this->baud_rate);

    // tty
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag |= ICANON;
    tty.c_lflag &= ~(ECHO | ECHOE | ECHONL); 
    
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    // force setting of the attribute to be vaild.
    rcpputils::assert_true(
        tcsetattr(serial_fd, TCSANOW, &tty) == 0,
        string_format("Error from tcsetattr: %s", std::strerror(errno)));

    // log the result and wait for data to populate.
    RCLCPP_INFO(this->get_logger(),
                "Serial port %s opened at specified baud rate!",
                serial_port.c_str());
    sleep(3);

    // automatically detect refresh rate of the peripheral using an anchoring
    // character... nominally this is /r/n, or just /n
    RCLCPP_INFO(this->get_logger(), "Checking Refresh Rate...");
  }

public:
  BridgeBase(const BridgeBase &) = delete;
  BridgeBase &operator=(const BridgeBase &) = delete;
  BridgeBase &operator=(BridgeBase &&) = delete;
  BridgeBase(BridgeBase &&) = delete;

  /**
   * @brief Base constructor for Bridges
   *
   */
  BridgeBase(std::string node_name, std::string serial_port_str,
             int baud_rate_int, size_t buflen)
      : Node(node_name), serial_fd(-1) {
    // Allows different baud rate and serial port to be set up during launch if
    // desired
    this->declare_parameter<std::string>("serial_port", serial_port_str);
    this->declare_parameter<int>("baud_rate", baud_rate_int);

    // set the baud_rate
    switch (this->get_parameter("baud_rate").as_int()) {
    case 9600:
      this->baud_rate = B9600;
      break;
    case 19200:
      this->baud_rate = B19200;
      break;
    case 38400:
      this->baud_rate = B38400;
      break;
    case 57600:
      this->baud_rate = B57600;
      break;
    case 115200:
      this->baud_rate = B115200;
      break;
    default:
      throw std::invalid_argument("Unsupported baud rate");
    }

    // Set the serial port and baud rate from the ROS start up sequence
    this->serial_port = this->get_parameter("serial_port").as_string();
    this->buf.length = buflen;
    this->buf.buffer.resize(this->buf.length);

    // Open the serial port
    open_serial_port();
  }

  virtual ~BridgeBase() {
    if (serial_fd >= 0)
      close(serial_fd);
  }

  /**
   * @brief Generic write serial function.
   *
   * @param msg String message shared pointer. @see std_msgs::msg::String
   */
  void write_serial(const std_msgs::msg::String::SharedPtr msg) {
    rcpputils::assert_true(this->serial_fd >= 0,
                           "Serial port not open. Failed to send command.");

    // Sets up string to be sent an appends a new line for arduino parsing
    std::string command = msg->data;
    command.append("\n");
    ssize_t bytes_written = write(serial_fd, command.c_str(), command.size());
    tcdrain(serial_fd);

    rcpputils::assert_true(
        bytes_written >= 0,
        string_format("Error writing to serial port: %s", strerror(errno)));
  }

  /**
   * @brief Generic read from terminal
   */
  std::string read_serial() {
    // ensure serial communication is successful, fail if not
    rcpputils::assert_true(this->serial_fd >= 0,
                           "Serial port not open. Failed to read command.");
    
    // read into a long buffer
    ssize_t bytes_read = read(this->serial_fd, this->buf.buffer.data(), this->buf.length - 1);
    this->buf.buffer[bytes_read] = '\0';

    if (bytes_read <= 0) return "";
    
    auto s = std::string(this->buf.buffer.data());

    return s;
  }
};
