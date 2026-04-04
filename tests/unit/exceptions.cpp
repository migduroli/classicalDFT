#include "dft/exceptions.hpp"

#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace dft::exception;

TEST_CASE("WrongParameterException stores message", "[exceptions]") {
  WrongParameterException e("bad value");
  CHECK(std::string(e.what()) == "bad value");
}

TEST_CASE("WrongParameterException is a runtime_error", "[exceptions]") {
  REQUIRE_THROWS_AS(throw WrongParameterException("msg"), std::runtime_error);
}

TEST_CASE("NegativeParameterException prepends prefix", "[exceptions]") {
  NegativeParameterException e("temperature");
  CHECK(std::string(e.what()) == "Negative parameter: temperature");
}

TEST_CASE("NegativeParameterException is a WrongParameterException", "[exceptions]") {
  REQUIRE_THROWS_AS(throw NegativeParameterException("x"), WrongParameterException);
}

TEST_CASE("NegativeParameterException is a runtime_error", "[exceptions]") {
  REQUIRE_THROWS_AS(throw NegativeParameterException("x"), std::runtime_error);
}
