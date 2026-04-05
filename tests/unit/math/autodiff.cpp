#include "dft/math/autodiff.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::math;

TEST_CASE("derivatives_up_to_1 of x^2", "[autodiff]") {
  auto [f, df] = derivatives_up_to_1([](dual x) -> dual { return x * x; }, 3.0);
  CHECK(f == Catch::Approx(9.0));
  CHECK(df == Catch::Approx(6.0));
}

TEST_CASE("derivatives_up_to_1 of exp(x)", "[autodiff]") {
  auto [f, df] = derivatives_up_to_1([](dual x) -> dual { return autodiff::detail::exp(x); }, 1.0);
  CHECK(f == Catch::Approx(std::exp(1.0)).epsilon(1e-12));
  CHECK(df == Catch::Approx(std::exp(1.0)).epsilon(1e-12));
}

TEST_CASE("derivatives_up_to_2 of x^3", "[autodiff]") {
  auto [f, df, d2f] = derivatives_up_to_2([](dual2nd x) -> dual2nd { return x * x * x; }, 2.0);
  CHECK(f == Catch::Approx(8.0));
  CHECK(df == Catch::Approx(12.0));
  CHECK(d2f == Catch::Approx(12.0));
}

TEST_CASE("derivatives_up_to_2 of sin(x) at pi/4", "[autodiff]") {
  double x0 = M_PI / 4.0;
  auto [f, df, d2f] = derivatives_up_to_2([](dual2nd x) -> dual2nd { return autodiff::detail::sin(x); }, x0);
  CHECK(f == Catch::Approx(std::sin(x0)).epsilon(1e-10));
  CHECK(df == Catch::Approx(std::cos(x0)).epsilon(1e-10));
  CHECK(d2f == Catch::Approx(-std::sin(x0)).epsilon(1e-10));
}

TEST_CASE("derivatives_up_to_3 of x^4", "[autodiff]") {
  auto [f, df, d2f, d3f] = derivatives_up_to_3([](dual3rd x) -> dual3rd { return x * x * x * x; }, 1.0);
  CHECK(f == Catch::Approx(1.0));
  CHECK(df == Catch::Approx(4.0));
  CHECK(d2f == Catch::Approx(12.0));
  CHECK(d3f == Catch::Approx(24.0));
}

TEST_CASE("derivatives_up_to_3 of polynomial", "[autodiff]") {
  // f(x) = x^4 + x^3 + x^2 + x + 1
  // f(1) = 5, f'(1) = 4+3+2+1 = 10, f''(1) = 12+6+2 = 20, f'''(1) = 24+6 = 30
  auto [f, df, d2f, d3f] = derivatives_up_to_3(
      [](dual3rd x) -> dual3rd { return x * x * x * x + x * x * x + x * x + x + dual3rd(1.0); },
      1.0
  );
  CHECK(f == Catch::Approx(5.0));
  CHECK(df == Catch::Approx(10.0));
  CHECK(d2f == Catch::Approx(20.0));
  CHECK(d3f == Catch::Approx(30.0));
}

TEST_CASE("derivatives_up_to_1 of constant returns zero derivative", "[autodiff]") {
  auto [f, df] = derivatives_up_to_1([](dual) -> dual { return dual(5.0); }, 99.0);
  CHECK(f == Catch::Approx(5.0));
  CHECK(df == Catch::Approx(0.0).margin(1e-15));
}
