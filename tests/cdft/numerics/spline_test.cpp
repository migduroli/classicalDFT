#include "cdft/numerics/spline.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace cdft::numerics {

  TEST(CubicSplineTest, LinearFunction) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0, 4.0};

    CubicSpline spline(x, y);
    EXPECT_NEAR(spline(0.5), 0.5, 1e-10);
    EXPECT_NEAR(spline(2.5), 2.5, 1e-10);
  }

  TEST(CubicSplineTest, Derivative) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0, 4.0};

    CubicSpline spline(x, y);
    EXPECT_NEAR(spline.derivative(2.0), 1.0, 1e-8);
  }

  TEST(CubicSplineTest, Integration) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 1.0, 2.0, 3.0, 4.0};

    CubicSpline spline(x, y);
    EXPECT_NEAR(spline.integrate(0.0, 4.0), 8.0, 1e-8);
  }

  TEST(CubicSplineTest, MinMax) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 1.0, 4.0, 9.0, 16.0};

    CubicSpline spline(x, y);
    EXPECT_DOUBLE_EQ(spline.x_min(), 0.0);
    EXPECT_DOUBLE_EQ(spline.x_max(), 4.0);
    EXPECT_EQ(spline.size(), 5u);
  }

  TEST(CubicSplineTest, MoveSemantics) {
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 1.0, 4.0, 9.0, 16.0};

    CubicSpline s1(x, y);
    CubicSpline s2 = std::move(s1);
    EXPECT_NEAR(s2(2.0), 4.0, 1e-8);
  }

  TEST(CubicSplineTest, TooFewPointsThrows) {
    std::vector<double> x = {0.0, 1.0};
    std::vector<double> y = {0.0, 1.0};
    EXPECT_THROW(CubicSpline(x, y), std::invalid_argument);
  }

}  // namespace cdft::numerics
