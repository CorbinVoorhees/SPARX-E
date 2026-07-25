#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include <variant>
namespace Sensor {
namespace Operations {
template <typename Input> class Read {
public:
  virtual int read(Input &input) noexcept = 0;
};

template <> class Read<std::monostate> {
public:
  virtual int read(std::monostate &) noexcept { return 0; }
};

template <typename Output> class Write {
public:
  virtual int write(Output &output) noexcept = 0;
};

template <> class Write<std::monostate> {
public:
  virtual int write(std::monostate &) noexcept { return 0; }
};
}; // namespace Operations
}; // namespace Sensor

#endif
