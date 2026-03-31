#include "dft/plotting/exceptions.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace dft::exception;

TEST_CASE("GraceException stores message via what()", "[exceptions]") {
  GraceException ex("test error");
  CHECK(std::string(ex.what()) == "test error");
}

TEST_CASE("GraceException is throwable and catchable", "[exceptions]") {
  REQUIRE_THROWS_AS(throw GraceException("error"), std::runtime_error);
}

TEST_CASE("GraceNotOpenedException has default message", "[exceptions]") {
  GraceNotOpenedException ex;
  CHECK(std::string(ex.what()) == "No grace subprocess currently connected.");
}

TEST_CASE("GraceNotOpenedException is a GraceException", "[exceptions]") {
  REQUIRE_THROWS_AS(throw GraceNotOpenedException(), GraceException);
}

TEST_CASE("GraceCommunicationFailedException has default message", "[exceptions]") {
  GraceCommunicationFailedException ex;
  CHECK(std::string(ex.what()) == "There was a problem while communicating with Grace.");
}

TEST_CASE("GraceCommunicationFailedException accepts custom message", "[exceptions]") {
  GraceCommunicationFailedException ex("custom error");
  CHECK(std::string(ex.what()) == "custom error");
}

TEST_CASE("GraceCommunicationFailedException is a GraceException", "[exceptions]") {
  REQUIRE_THROWS_AS(throw GraceCommunicationFailedException(), GraceException);
}
