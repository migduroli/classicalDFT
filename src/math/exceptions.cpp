#include "dft/math/exceptions.h"

#include <utility>

namespace dft::exception {
  NegativeParameterException::NegativeParameterException(std::string msg) : WrongParameterException(std::move(msg)) {}
}  // namespace dft::exception