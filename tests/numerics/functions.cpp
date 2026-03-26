#include "classicaldft_bits/numerics/functions.h"

#include <gtest/gtest.h>

namespace example {
  class TestClass {
   public:
    TestClass() = default;
    double fn_cubic(double x) const { return x * x * x; }
  };

  double fn_sqr(double x) {
    return x * x;
  }
}  // namespace example

using namespace dft_core::utils::functions;

// region Methods
TEST(functions, apply_vector_wise_fn_works_ok) {
  auto x_vec = std::vector<double>{1.0, 2.0, 3.0, 1.765};
  auto actual = apply_vector_wise<double>(&example::fn_sqr, x_vec);
  auto expected = std::vector<double>{1.0, 4.0, 9.0, 3.115225};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(expected[i], actual[i]);
  }
}

TEST(functions, apply_vector_wise_method_works_ok) {
  auto x_vec = std::vector<double>{1.0, 2.0, 3.0, 1.765};
  auto obj = example::TestClass();
  auto method = [](const example::TestClass& o, double x) { return o.fn_cubic(x); };
  auto actual = apply_vector_wise<example::TestClass, double>(obj, method, x_vec);
  auto expected = std::vector<double>{1.0, 8.0, 27.0, 5.498372125};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(expected[i], actual[i]);
  }
}
// endregion