#ifndef DFT_PLOTTING_EXCEPTIONS_HPP
#define DFT_PLOTTING_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>
#include <utility>

namespace dft::exception {

  class GraceException : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
  };

  class GraceNotOpenedException : public GraceException {
   public:
    GraceNotOpenedException() : GraceException("No grace subprocess currently connected.") {}
  };

  class GraceCommunicationFailedException : public GraceException {
   public:
    GraceCommunicationFailedException() : GraceException("There was a problem while communicating with Grace.") {}

    explicit GraceCommunicationFailedException(std::string msg) : GraceException(std::move(msg)) {}
  };

} // namespace dft::exception

#endif // DFT_PLOTTING_EXCEPTIONS_HPP
