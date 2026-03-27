#include "classicaldft_bits/numerics/functions.h"

#include <cmath>
#include <gtest/gtest.h>

namespace functions = dft_core::utils::functions;

namespace {
  double square(double x) {
    return x * x;
  }

  class TestObject {
   public:
    double cube(double x) const { return x * x * x; }
  };
}  // namespace

// ── apply_vector_wise (free function overload) ──────────────────────────────

struct ApplyFreeFunctionTestCase {
  std::string name;
  std::vector<double> input;
  std::vector<double> expected;
};

class ApplyFreeFunctionTest : public testing::TestWithParam<ApplyFreeFunctionTestCase> {};

TEST_P(ApplyFreeFunctionTest, ProducesCorrectOutput) {
  auto [name, input, expected] = GetParam();
  auto actual = functions::apply_vector_wise<double>(&square, input);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(expected[i], actual[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ApplyVectorWise,
    ApplyFreeFunctionTest,
    testing::Values(
        ApplyFreeFunctionTestCase{"integers", {1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}},
        ApplyFreeFunctionTestCase{"empty", {}, {}},
        ApplyFreeFunctionTestCase{"single", {5.0}, {25.0}},
        ApplyFreeFunctionTestCase{"fractional", {1.5, 2.5}, {2.25, 6.25}}
    ),
    [](const auto& info) { return info.param.name; }
);

// ── apply_vector_wise (member function overload) ────────────────────────────

struct ApplyMethodTestCase {
  std::string name;
  std::vector<double> input;
  std::vector<double> expected;
};

class ApplyMethodTest : public testing::TestWithParam<ApplyMethodTestCase> {};

TEST_P(ApplyMethodTest, ProducesCorrectOutput) {
  auto [name, input, expected] = GetParam();
  auto obj = TestObject();
  auto method = [](const TestObject& o, double x) { return o.cube(x); };
  auto actual = functions::apply_vector_wise<TestObject, double>(obj, method, input);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_DOUBLE_EQ(expected[i], actual[i]);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ApplyVectorWise,
    ApplyMethodTest,
    testing::Values(
        ApplyMethodTestCase{"integers", {1.0, 2.0, 3.0}, {1.0, 8.0, 27.0}},
        ApplyMethodTestCase{"empty", {}, {}},
        ApplyMethodTestCase{"single", {4.0}, {64.0}}
    ),
    [](const auto& info) { return info.param.name; }
);
