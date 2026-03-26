#include "classicaldft_bits/exceptions/parameter.h"

#include <utility>

namespace dft_core::exception {
  NegativeParameterException::NegativeParameterException(std::string msg) : WrongParameterException(std::move(msg)) {}
}  // namespace dft_core::exception