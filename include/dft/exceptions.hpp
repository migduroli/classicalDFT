#ifndef DFT_EXCEPTIONS_HPP
#define DFT_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>

namespace dft::exception {

  class WrongParameterException : public std::runtime_error {
   public:
    explicit WrongParameterException(const std::string& msg) : std::runtime_error(msg) {}
  };

  class NegativeParameterException : public WrongParameterException {
   public:
    explicit NegativeParameterException(const std::string& msg)
        : WrongParameterException("Negative parameter: " + msg) {}
  };

}  // namespace dft::exception

#endif  // DFT_EXCEPTIONS_HPP
