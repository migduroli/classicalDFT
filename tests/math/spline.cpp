#include "dft/math/spline.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

using namespace dft::math;

TEST_CASE("CubicSpline interpolates linear function exactly", "[spline]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<double> y = {0.0, 2.0, 4.0, 6.0, 8.0};
  CubicSpline s(x, y);
  CHECK(s(0.5) == Catch::Approx(1.0).epsilon(1e-10));
  CHECK(s(2.5) == Catch::Approx(5.0).epsilon(1e-10));
}

TEST_CASE("CubicSpline derivative of linear function is constant", "[spline]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<double> y = {0.0, 2.0, 4.0, 6.0, 8.0};
  CubicSpline s(x, y);
  CHECK(s.derivative(1.5) == Catch::Approx(2.0).epsilon(1e-8));
}

TEST_CASE("CubicSpline second derivative of linear function is zero", "[spline]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<double> y = {0.0, 2.0, 4.0, 6.0, 8.0};
  CubicSpline s(x, y);
  CHECK(s.derivative2(2.0) == Catch::Approx(0.0).margin(1e-8));
}

TEST_CASE("CubicSpline integrate gives correct area", "[spline]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<double> y = {0.0, 2.0, 4.0, 6.0, 8.0};
  CubicSpline s(x, y);
  // integral of 2x from 0 to 4 = 16
  CHECK(s.integrate(0.0, 4.0) == Catch::Approx(16.0).epsilon(1e-8));
}

TEST_CASE("CubicSpline reports x_min and x_max", "[spline]") {
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {1.0, 4.0, 9.0, 16.0, 25.0};
  CubicSpline s(x, y);
  CHECK(s.x_min() == 1.0);
  CHECK(s.x_max() == 5.0);
  CHECK(s.size() == 5);
}

TEST_CASE("CubicSpline interpolates quadratic well", "[spline]") {
  std::vector<double> x, y;
  for (int i = 0; i <= 20; ++i) {
    double xi = 0.5 * i;
    x.push_back(xi);
    y.push_back(xi * xi);
  }
  CubicSpline s(x, y);
  CHECK(s(3.7) == Catch::Approx(3.7 * 3.7).epsilon(1e-4));
  CHECK(s.derivative(3.0) == Catch::Approx(6.0).epsilon(1e-3));
}

TEST_CASE("CubicSpline throws for mismatched sizes", "[spline]") {
  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> y = {1.0, 2.0};
  REQUIRE_THROWS_AS(CubicSpline(x, y), std::invalid_argument);
}

TEST_CASE("CubicSpline throws for too few points", "[spline]") {
  std::vector<double> x = {1.0, 2.0};
  std::vector<double> y = {1.0, 2.0};
  REQUIRE_THROWS_AS(CubicSpline(x, y), std::invalid_argument);
}

TEST_CASE("CubicSpline is move-constructible", "[spline]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> y = {0.0, 1.0, 4.0, 9.0};
  CubicSpline s1(x, y);
  CubicSpline s2(std::move(s1));
  CHECK(s2(1.5) == Catch::Approx(2.25).epsilon(0.1));
}

TEST_CASE("BivariateSpline interpolates bilinear function", "[spline]") {
  // f(x, y) = x + 2*y on a 4x4 grid (bicubic needs >= 4 points per axis)
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
  // GSL expects z in column-major order: z[j*nx + i] = f(x[i], y[j])
  std::vector<double> z;
  for (double yj : y) {
    for (double xi : x) {
      z.push_back(xi + 2.0 * yj);
    }
  }
  BivariateSpline s(x, y, z);
  CHECK(s(0.5, 0.5) == Catch::Approx(1.5).epsilon(1e-4));
  CHECK(s(1.0, 1.0) == Catch::Approx(3.0).epsilon(1e-6));
}

TEST_CASE("BivariateSpline derivatives of f(x,y) = x + 2y", "[spline]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> z;
  for (double yj : y) {
    for (double xi : x) {
      z.push_back(xi + 2.0 * yj);
    }
  }
  BivariateSpline s(x, y, z);
  CHECK(s.deriv_x(1.5, 1.5) == Catch::Approx(1.0).epsilon(1e-4));
  CHECK(s.deriv_y(1.5, 1.5) == Catch::Approx(2.0).epsilon(1e-4));
}

TEST_CASE("BivariateSpline second derivatives of bilinear are zero", "[spline]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> z;
  for (double yj : y) {
    for (double xi : x) {
      z.push_back(xi + 2.0 * yj);
    }
  }
  BivariateSpline s(x, y, z);
  CHECK(s.deriv_xx(1.5, 1.5) == Catch::Approx(0.0).margin(1e-8));
  CHECK(s.deriv_yy(1.5, 1.5) == Catch::Approx(0.0).margin(1e-8));
  CHECK(s.deriv_xy(1.5, 1.5) == Catch::Approx(0.0).margin(1e-8));
}

TEST_CASE("BivariateSpline throws for wrong z size", "[spline]") {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> y = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> z = {1.0, 2.0};  // wrong size
  REQUIRE_THROWS_AS(BivariateSpline(x, y, z), std::invalid_argument);
}

TEST_CASE("BivariateSpline throws for empty x", "[spline]") {
  std::vector<double> x;
  std::vector<double> y = {0.0, 1.0};
  std::vector<double> z;
  REQUIRE_THROWS_AS(BivariateSpline(x, y, z), std::invalid_argument);
}
