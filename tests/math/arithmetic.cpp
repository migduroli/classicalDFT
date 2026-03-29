#include "dft/math/arithmetic.h"

#include <cmath>
#include <gtest/gtest.h>
#include <sstream>

using namespace dft::math::arithmetic;

// ── Free functions: parameterized across summation algorithms ────────────────

struct FreeSumTestCase {
  std::string name;
  std::vector<double> input;
  double expected;
};

class KahanBabuskaTest : public testing::TestWithParam<FreeSumTestCase> {};
class KahanBabuskaNeumaierTest : public testing::TestWithParam<FreeSumTestCase> {};
class KahanBabuskaKleinTest : public testing::TestWithParam<FreeSumTestCase> {};

TEST_P(KahanBabuskaTest, SumProducesCorrectResult) {
  auto [name, input, expected] = GetParam();
  auto [sum, err] = kahan_babuska_sum(input);
  EXPECT_DOUBLE_EQ(expected, sum);
}

TEST_P(KahanBabuskaNeumaierTest, SumProducesCorrectResult) {
  auto [name, input, expected] = GetParam();
  auto [sum, err] = kahan_babuska_neumaier_sum(input);
  EXPECT_DOUBLE_EQ(expected, sum);
}

TEST_P(KahanBabuskaKleinTest, SumProducesCorrectResult) {
  auto [name, input, expected] = GetParam();
  auto [sum, err] = kahan_babuska_klein_sum(input);
  EXPECT_DOUBLE_EQ(expected, sum);
}

auto sum_test_cases = testing::Values(
    FreeSumTestCase{"simple", {1.0, 2.0, 4.0, 3.0}, 10.0},
    FreeSumTestCase{"single", {42.0}, 42.0},
    FreeSumTestCase{"negative", {-1.0, -2.0, -3.0, -4.0}, -10.0},
    FreeSumTestCase{"mixed", {1e15, 1.0, -1e15}, 1.0}
);

// 1e16+1 loses precision in double; only Neumaier and Klein recover it
auto compensated_test_cases = testing::Values(
    FreeSumTestCase{"simple", {1.0, 2.0, 4.0, 3.0}, 10.0},
    FreeSumTestCase{"single", {42.0}, 42.0},
    FreeSumTestCase{"negative", {-1.0, -2.0, -3.0, -4.0}, -10.0},
    FreeSumTestCase{"precision_loss", {1e16, 1.0, -1e16}, 1.0}
);

INSTANTIATE_TEST_SUITE_P(Summation, KahanBabuskaTest, sum_test_cases, [](const auto& info) { return info.param.name; });
INSTANTIATE_TEST_SUITE_P(Summation, KahanBabuskaNeumaierTest, compensated_test_cases, [](const auto& info) {
  return info.param.name;
});
INSTANTIATE_TEST_SUITE_P(Summation, KahanBabuskaKleinTest, compensated_test_cases, [](const auto& info) {
  return info.param.name;
});

TEST(KahanBabuska, ErrorBoundedByEpsilon) {
  auto input = std::vector<double>{1.0, 2.0, 4.0, 3.0};
  auto [sum, err] = kahan_babuska_sum(input);
  EXPECT_LE(std::abs(err.front()), 2 * DBL_EPSILON * 10);
}

TEST(KahanBabuskaNeumaier, ErrorBoundedByEpsilon) {
  auto input = std::vector<double>{1.0, 2.0, 4.0, 3.0};
  auto [sum, err] = kahan_babuska_neumaier_sum(input);
  EXPECT_LE(std::abs(err.front()), 2 * DBL_EPSILON * 10);
}

TEST(KahanBabuskaKlein, ErrorBoundedByEpsilon) {
  auto input = std::vector<double>{1.0, 2.0, 4.0, 3.0};
  auto [sum, err] = kahan_babuska_klein_sum(input);
  EXPECT_LE(std::abs(err[0]) + std::abs(err[1]), 2 * DBL_EPSILON * 10);
}

TEST(KahanBabuska, ContinuedSummation) {
  auto part1 = std::vector<double>{1.0, 2.0};
  auto [sum1, err1] = kahan_babuska_sum(part1);
  auto part2 = std::vector<double>{3.0, 4.0};
  auto [sum2, err2] = kahan_babuska_sum(part2, sum1, err1);
  EXPECT_DOUBLE_EQ(10.0, sum2);
}

TEST(KahanBabuskaNeumaier, ContinuedSummation) {
  auto part1 = std::vector<double>{1.0, 2.0};
  auto [sum1, err1] = kahan_babuska_neumaier_sum(part1);
  auto part2 = std::vector<double>{3.0, 4.0};
  auto [sum2, err2] = kahan_babuska_neumaier_sum(part2, sum1, err1);
  EXPECT_DOUBLE_EQ(10.0, sum2);
}

TEST(KahanBabuskaKlein, ContinuedSummation) {
  auto part1 = std::vector<double>{1.0, 2.0};
  auto [sum1, err1] = kahan_babuska_klein_sum(part1);
  auto part2 = std::vector<double>{3.0, 4.0};
  auto [sum2, err2] = kahan_babuska_klein_sum(part2, sum1, err1);
  EXPECT_DOUBLE_EQ(10.0, sum2);
}

// ── standard_vector_sum ─────────────────────────────────────────────────────

TEST(StandardVectorSum, ComputesCorrectSum) {
  auto input = std::vector<double>{1.0, 2.0, 3.0, 4.0};
  EXPECT_DOUBLE_EQ(standard_vector_sum(input), 10.0);
}

TEST(StandardVectorSum, EmptyVector) {
  auto input = std::vector<double>{};
  EXPECT_DOUBLE_EQ(standard_vector_sum(input), 0.0);
}

TEST(StandardVectorSum, FloatVersion) {
  auto input = std::vector<float>{1.0f, 2.0f, 3.0f};
  EXPECT_FLOAT_EQ(standard_vector_sum(input), 6.0f);
}

// ── CompensatedSum class: parameterized across types ────────────────────────

struct CompensatedSumTypeTestCase {
  std::string name;
  Type type;
};

class CompensatedSumTypeTest : public testing::TestWithParam<CompensatedSumTypeTestCase> {};

TEST_P(CompensatedSumTypeTest, SumViaVectorPlusEquals) {
  auto [name, type] = GetParam();
  auto input = std::vector<double>{1.0, 2.0, 4.0, 3.0};
  auto cs = CompensatedSum(type);
  cs += input;
  EXPECT_DOUBLE_EQ(10.0, cs.sum());
}

TEST_P(CompensatedSumTypeTest, SumViaScalarPlusEquals) {
  auto [name, type] = GetParam();
  auto cs = CompensatedSum(type);
  cs += 1.0;
  cs += 2.0;
  cs += 4.0;
  cs += 3.0;
  EXPECT_DOUBLE_EQ(10.0, cs.sum());
}

TEST_P(CompensatedSumTypeTest, SubtractViaScalarMinusEquals) {
  auto [name, type] = GetParam();
  auto cs = CompensatedSum(type);
  cs += 10.0;
  cs -= 3.0;
  EXPECT_NEAR(7.0, cs.sum(), 1e-14);
}

TEST_P(CompensatedSumTypeTest, SubtractViaVectorMinusEquals) {
  auto [name, type] = GetParam();
  auto cs = CompensatedSum(type);
  cs += 10.0;
  cs -= std::vector<double>{3.0, 2.0};
  EXPECT_NEAR(5.0, cs.sum(), 1e-14);
}

TEST_P(CompensatedSumTypeTest, ConversionToDouble) {
  auto [name, type] = GetParam();
  auto cs = CompensatedSum(type);
  cs += std::vector<double>{1.0, 2.0, 3.0};
  EXPECT_DOUBLE_EQ(6.0, static_cast<double>(cs));
}

TEST_P(CompensatedSumTypeTest, OperatorPlusScalar) {
  auto [name, type] = GetParam();
  auto cs = CompensatedSum(type);
  cs + 5.0;
  EXPECT_NEAR(5.0, cs.sum(), 1e-14);
}

TEST_P(CompensatedSumTypeTest, OperatorMinusScalar) {
  auto [name, type] = GetParam();
  auto cs = CompensatedSum(type);
  cs += 10.0;
  cs - 3.0;
  EXPECT_NEAR(7.0, cs.sum(), 1e-14);
}

TEST_P(CompensatedSumTypeTest, AssignmentFromDouble) {
  auto [name, type] = GetParam();
  auto cs = CompensatedSum(type);
  cs += std::vector<double>{1.0, 2.0, 3.0};
  cs = 42.0;
  EXPECT_DOUBLE_EQ(42.0, cs.sum());
}

INSTANTIATE_TEST_SUITE_P(
    AllTypes,
    CompensatedSumTypeTest,
    testing::Values(
        CompensatedSumTypeTestCase{"KB", Type::KahanBabuska},
        CompensatedSumTypeTestCase{"KBN", Type::KahanBabuskaNeumaier},
        CompensatedSumTypeTestCase{"KBK", Type::KahanBabuskaKlein}
    ),
    [](const auto& info) { return info.param.name; }
);

// ── CompensatedSum: inter-object operators ──────────────────────────────────

TEST(CompensatedSum, PlusEqualsOtherSum) {
  auto a = CompensatedSum(Type::KahanBabuskaNeumaier);
  auto b = CompensatedSum(Type::KahanBabuskaNeumaier);
  a += 3.0;
  b += 7.0;
  a += b;
  EXPECT_NEAR(10.0, a.sum(), 1e-14);
}

TEST(CompensatedSum, MinusEqualsOtherSum) {
  auto a = CompensatedSum(Type::KahanBabuskaNeumaier);
  auto b = CompensatedSum(Type::KahanBabuskaNeumaier);
  a += 10.0;
  b += 3.0;
  a -= b;
  EXPECT_NEAR(7.0, a.sum(), 1e-14);
}

TEST(CompensatedSum, OperatorPlusOtherSum) {
  auto a = CompensatedSum(Type::KahanBabuskaNeumaier);
  auto b = CompensatedSum(Type::KahanBabuskaNeumaier);
  a += 3.0;
  b += 7.0;
  a + b;
  EXPECT_NEAR(10.0, a.sum(), 1e-14);
}

TEST(CompensatedSum, OperatorMinusOtherSum) {
  auto a = CompensatedSum(Type::KahanBabuskaNeumaier);
  auto b = CompensatedSum(Type::KahanBabuskaNeumaier);
  a += 10.0;
  b += 3.0;
  a - b;
  EXPECT_NEAR(7.0, a.sum(), 1e-14);
}

// ── CompensatedSum: inspectors ──────────────────────────────────────────────

TEST(CompensatedSum, TypeInspector) {
  auto cs = CompensatedSum(Type::KahanBabuskaKlein);
  EXPECT_EQ(cs.type(), Type::KahanBabuskaKlein);
}

TEST(CompensatedSum, ErrorInspector) {
  auto cs = CompensatedSum(Type::KahanBabuskaNeumaier);
  EXPECT_EQ(cs.error().size(), 1U);

  auto cs2 = CompensatedSum(Type::KahanBabuskaKlein);
  EXPECT_EQ(cs2.error().size(), 2U);
}

TEST(CompensatedSum, DefaultTypeIsNeumaier) {
  auto cs = CompensatedSum();
  EXPECT_EQ(cs.type(), Type::KahanBabuskaNeumaier);
}

// ── CompensatedSum: ostream ─────────────────────────────────────────────────

TEST(CompensatedSum, StreamOutput) {
  auto cs = CompensatedSum();
  cs += 42.0;
  std::ostringstream ss;
  ss << cs;
  EXPECT_EQ(ss.str(), "42");
}

TEST(CompensatedSum, UnknownTypeDefaultsToNeumaier) {
  // Construct with an out-of-range enum to exercise the else branch
  auto cs = CompensatedSum(static_cast<Type>(99));
  cs += std::vector<double>{1.0, 2.0, 3.0};
  EXPECT_DOUBLE_EQ(6.0, cs.sum());
}
