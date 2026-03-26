#include "dft_lib/exceptions/parameter.h"

#include <utility>

namespace dft_core {
  namespace exception {
    NegativeParameterException::NegativeParameterException(std::string msg) : WrongParameterException(std::move(msg)) {}
  }  // namespace exception
}  // namespace dft_core