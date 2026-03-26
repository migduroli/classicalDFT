#include "dft_lib/exceptions/grace.h"

#include <utility>

namespace dft_core {
  namespace exception {
    GraceNotOpenedException::GraceNotOpenedException() : GraceException(std::string()) {
      set_error_message("No grace subprocess currently connected.");
    }

    GraceCommunicationFailedException::GraceCommunicationFailedException() : GraceException(std::string()) {
      set_error_message("There was a problem while communicating with Grace.");
    }

    GraceCommunicationFailedException::GraceCommunicationFailedException(std::string msg)
        : GraceException(std::move(msg)) {}
  }  // namespace exception
}  // namespace dft_core
