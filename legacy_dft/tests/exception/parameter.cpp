#include "classicaldft_bits/exception/parameter.h"

#include <gtest/gtest.h>

TEST(general_exceptions, wrong_parameter_exception_cttor_test) {
  std::string msg = "new exception";
  auto exception = dft::exception::WrongParameterException(msg);
  EXPECT_THROW(throw exception, dft::exception::WrongParameterException);
  ASSERT_STREQ(exception.error_message().c_str(), msg.c_str());
}

TEST(general_exceptions, negative_parameter_exception_cttor_test) {
  std::string msg = "new exception";
  auto exception = dft::exception::NegativeParameterException(msg);
  EXPECT_THROW(throw exception, dft::exception::NegativeParameterException);
  ASSERT_STREQ(exception.error_message().c_str(), msg.c_str());
  ASSERT_STREQ(exception.what(), msg.c_str());
}