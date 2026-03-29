#include "cdft/numerics/math.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

namespace cdft::numerics {

  TEST(IntegrationTest, QagsLinear) {
    auto result = integrate_qags([](double x) { return x; }, 0.0, 1.0);
    EXPECT_NEAR(result.value, 0.5, 1e-10);
  }

  TEST(IntegrationTest, QagsSine) {
    auto result = integrate_qags([](double x) { return std::sin(x); }, 0.0, std::numbers::pi);
    EXPECT_NEAR(result.value, 2.0, 1e-10);
  }

  TEST(IntegrationTest, QngGaussian) {
    auto result = integrate_qng([](double x) { return std::exp(-x * x); }, -3.0, 3.0);
    EXPECT_NEAR(result.value, std::sqrt(std::numbers::pi), 0.01);
  }

  TEST(CompensatedSumTest, BasicAccumulation) {
    CompensatedSum cs;
    cs += 1.0;
    cs += 2.0;
    cs += 3.0;
    EXPECT_NEAR(cs.value(), 6.0, 1e-15);
  }

  TEST(CompensatedSumTest, CancellationProtection) {
    CompensatedSum cs;
    cs += 1e16;
    cs += 1.0;
    cs += -1e16;
    EXPECT_NEAR(cs.value(), 1.0, 1e-6);
  }

  TEST(SumResultTest, VectorSum) {
    std::vector<double> vals = {1.0, 2.0, 3.0, 4.0};
    auto result = compensated_sum(vals);
    EXPECT_NEAR(result.sum, 10.0, 1e-15);
  }

  TEST(IntegratorLegacyTest, DefiniteIntegral) {
    struct Problem {
      double f(double x) const { return x * x; }
    };
    Problem p;
    Integrator<Problem> integrator(p, &Problem::f);
    double result = integrator.definite_integral(0.0, 1.0);
    EXPECT_NEAR(result, 1.0 / 3.0, 1e-10);
  }

}  // namespace cdft::numerics
