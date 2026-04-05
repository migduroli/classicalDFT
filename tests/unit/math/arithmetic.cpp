#include "dft/math/arithmetic.hpp"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <vector>

using namespace dft::math;

TEST_CASE("kahan_sum of empty span returns 0", "[arithmetic]") {
  std::vector<double> v;
  CHECK(kahan_sum(v) == 0.0);
}

TEST_CASE("kahan_sum of single element returns that element", "[arithmetic]") {
  std::array<double, 1> v{42.0};
  CHECK(kahan_sum(v) == 42.0);
}

TEST_CASE("kahan_sum accumulates small values accurately", "[arithmetic]") {
  constexpr int N = 10'000'000;
  std::vector<double> v(N, 1e-7);
  double expected = static_cast<double>(N) * 1e-7;
  CHECK(kahan_sum(v) == Catch::Approx(expected).epsilon(1e-12));
}

TEST_CASE("neumaier_sum of empty span returns 0", "[arithmetic]") {
  std::vector<double> v;
  CHECK(neumaier_sum(v) == 0.0);
}

TEST_CASE("neumaier_sum handles large-then-small pattern", "[arithmetic]") {
  // Pattern where naive sum loses precision: big + small - big
  std::vector<double> v{1e16, 1.0, -1e16};
  CHECK(neumaier_sum(v) == 1.0);
}

TEST_CASE("neumaier_sum accumulates small values accurately", "[arithmetic]") {
  constexpr int N = 10'000'000;
  std::vector<double> v(N, 1e-7);
  double expected = static_cast<double>(N) * 1e-7;
  CHECK(neumaier_sum(v) == Catch::Approx(expected).epsilon(1e-12));
}

TEST_CASE("klein_sum of empty span returns 0", "[arithmetic]") {
  std::vector<double> v;
  CHECK(klein_sum(v) == 0.0);
}

TEST_CASE("klein_sum handles large-then-small pattern", "[arithmetic]") {
  std::vector<double> v{1e16, 1.0, -1e16};
  CHECK(klein_sum(v) == 1.0);
}

TEST_CASE("klein_sum accumulates small values accurately", "[arithmetic]") {
  constexpr int N = 10'000'000;
  std::vector<double> v(N, 1e-7);
  double expected = static_cast<double>(N) * 1e-7;
  CHECK(klein_sum(v) == Catch::Approx(expected).epsilon(1e-12));
}

TEST_CASE("all three sums agree on well-conditioned input", "[arithmetic]") {
  std::vector<double> v{1.0, 2.0, 3.0, 4.0, 5.0};
  CHECK(kahan_sum(v) == 15.0);
  CHECK(neumaier_sum(v) == 15.0);
  CHECK(klein_sum(v) == 15.0);
}

TEST_CASE("compensated sums handle negative values", "[arithmetic]") {
  std::vector<double> v{-1.0, -2.0, -3.0, -4.0, -5.0};
  CHECK(kahan_sum(v) == -15.0);
  CHECK(neumaier_sum(v) == -15.0);
  CHECK(klein_sum(v) == -15.0);
}
