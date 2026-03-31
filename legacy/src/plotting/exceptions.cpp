#include "dft/plotting/exceptions.h"

#include <utility>

namespace dft::exception {
  GraceNotOpenedException::GraceNotOpenedException() : GraceException(std::string()) {
    set_error_message("No grace subprocess currently connected.");
  }

  GraceCommunicationFailedException::GraceCommunicationFailedException() : GraceException(std::string()) {
    set_error_message("There was a problem while communicating with Grace.");
  }

  GraceCommunicationFailedException::GraceCommunicationFailedException(std::string msg)
      : GraceException(std::move(msg)) {}
}  // namespace dft::exception
