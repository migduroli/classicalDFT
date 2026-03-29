#include "classicaldft_bits/numerics/spline.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace spline = dft::numerics::spline;

// ── CubicSpline ─────────────────────────────────────────────────────────────

TEST(CubicSpline, InterpolateSinAtMidpoints) {
  const int n = 100;
  std::vector<double> x(n);
  std::vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n - 1);
    y[i] = std::sin(x[i]);
  }

  spline::CubicSpline s(x, y);

  for (int i = 0; i < n - 1; ++i) {
    double xmid = 0.5 * (x[i] + x[i + 1]);
    EXPECT_NEAR(s(xmid), std::sin(xmid), 1e-5);
  }
}

TEST(CubicSpline, DerivativeOfSin) {
  const int n = 200;
  std::vector<double> x(n);
  std::vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n - 1);
    y[i] = std::sin(x[i]);
  }

  spline::CubicSpline s(x, y);

  for (int i = 10; i < n - 10; ++i) {
    double xval = x[i];
    EXPECT_NEAR(s.derivative(xval), std::cos(xval), 1e-3);
  }
}

TEST(CubicSpline, SecondDerivativeOfCubic) {
  const int n = 50;
  std::vector<double> x(n);
  std::vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = -2.0 + 4.0 * static_cast<double>(i) / static_cast<double>(n - 1);
    y[i] = x[i] * x[i] * x[i];
  }

  spline::CubicSpline s(x, y);

  for (int i = 5; i < n - 5; ++i) {
    EXPECT_NEAR(s.derivative2(x[i]), 6.0 * x[i], 0.1);
  }
}

TEST(CubicSpline, IntegrateSin) {
  const int n = 200;
  std::vector<double> x(n);
  std::vector<double> y(n);
  for (int i = 0; i < n; ++i) {
    x[i] = M_PI * static_cast<double>(i) / static_cast<double>(n - 1);
    y[i] = std::sin(x[i]);
  }

  spline::CubicSpline s(x, y);
  double result = s.integrate(0.0, M_PI);
  EXPECT_NEAR(result, 2.0, 1e-4);
}

TEST(CubicSpline, XMinMaxSize) {
  std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> y = {1.0, 4.0, 9.0, 16.0, 25.0};

  spline::CubicSpline s(x, y);
  EXPECT_DOUBLE_EQ(s.x_min(), 1.0);
  EXPECT_DOUBLE_EQ(s.x_max(), 5.0);
  EXPECT_EQ(s.size(), 5U);
}

TEST(CubicSpline, TooFewPointsThrows) {
  std::vector<double> x = {0.0, 1.0};
  std::vector<double> y = {0.0, 1.0};
  EXPECT_THROW(spline::CubicSpline(x, y), std::invalid_argument);
}

TEST(CubicSpline, MismatchedSizesThrows) {
  std::vector<double> x = {0.0, 1.0, 2.0};
  std::vector<double> y = {0.0, 1.0};
  EXPECT_THROW(spline::CubicSpline(x, y), std::invalid_argument);
}

TEST(CubicSpline, MoveConstruction) {
  std::vector<double> x = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> y = {0.0, 1.0, 4.0, 9.0};

  spline::CubicSpline s1(x, y);
  double val = s1(1.5);

  spline::CubicSpline s2(std::move(s1));
  EXPECT_NEAR(s2(1.5), val, 1e-15);
  EXPECT_EQ(s2.size(), 4U);
}

TEST(CubicSpline, MoveAssignment) {
  std::vector<double> x1 = {0.0, 1.0, 2.0, 3.0};
  std::vector<double> y1 = {0.0, 1.0, 4.0, 9.0};
  std::vector<double> x2 = {0.0, 1.0, 2.0, 3.0, 4.0};
  std::vector<double> y2 = {0.0, 1.0, 8.0, 27.0, 64.0};

  spline::CubicSpline s1(x1, y1);
  spline::CubicSpline s2(x2, y2);

  double val = s1(1.5);
  s2 = std::move(s1);
  EXPECT_NEAR(s2(1.5), val, 1e-15);
  EXPECT_EQ(s2.size(), 4U);
}

// ── BivariateSpline ─────────────────────────────────────────────────────────

TEST(BivariateSpline, InterpolateBilinear) {
  const int nx = 4;
  const int ny = 4;
  std::vector<double> x(nx);
  std::vector<double> y(ny);
  std::vector<double> z(nx * ny);

  for (int i = 0; i < nx; ++i) {
    x[i] = static_cast<double>(i);
  }
  for (int j = 0; j < ny; ++j) {
    y[j] = static_cast<double>(j);
  }
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      z[static_cast<std::size_t>(i * ny + j)] = x[i] * y[j];
    }
  }

  spline::BivariateSpline s(x, y, z);
  EXPECT_NEAR(s(1.5, 1.5), 2.25, 0.1);
  EXPECT_NEAR(s(2.0, 1.0), 2.0, 1e-10);
}

TEST(BivariateSpline, Derivatives) {
  const int nx = 10;
  const int ny = 10;
  std::vector<double> x(nx);
  std::vector<double> y(ny);
  std::vector<double> z(nx * ny);

  for (int i = 0; i < nx; ++i) {
    x[i] = static_cast<double>(i);
  }
  for (int j = 0; j < ny; ++j) {
    y[j] = static_cast<double>(j);
  }
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      z[static_cast<std::size_t>(i * ny + j)] = x[i] * x[i] + y[j] * y[j];
    }
  }

  spline::BivariateSpline s(x, y, z);
  EXPECT_NEAR(s.deriv_x(4.5, 4.5), 9.0, 0.3);
  EXPECT_NEAR(s.deriv_y(4.5, 4.5), 9.0, 0.3);
  EXPECT_NEAR(s.deriv_xx(4.5, 4.5), 2.0, 0.3);
  EXPECT_NEAR(s.deriv_yy(4.5, 4.5), 2.0, 0.3);
  EXPECT_NEAR(s.deriv_xy(4.5, 4.5), 0.0, 0.3);
}

TEST(BivariateSpline, MismatchedZThrows) {
  std::vector<double> x = {0.0, 1.0, 2.0};
  std::vector<double> y = {0.0, 1.0, 2.0};
  std::vector<double> z = {1.0, 2.0};
  EXPECT_THROW(spline::BivariateSpline(x, y, z), std::invalid_argument);
}

TEST(BivariateSpline, EmptyInputThrows) {
  std::vector<double> x = {};
  std::vector<double> y = {0.0, 1.0, 2.0};
  std::vector<double> z = {};
  EXPECT_THROW(spline::BivariateSpline(x, y, z), std::invalid_argument);
}

TEST(BivariateSpline, MoveConstruction) {
  const int nx = 4;
  const int ny = 4;
  std::vector<double> x(nx);
  std::vector<double> y(ny);
  std::vector<double> z(nx * ny);
  for (int i = 0; i < nx; ++i) {
    x[i] = static_cast<double>(i);
  }
  for (int j = 0; j < ny; ++j) {
    y[j] = static_cast<double>(j);
  }
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      z[static_cast<std::size_t>(i * ny + j)] = x[i] + y[j];
    }
  }

  spline::BivariateSpline s1(x, y, z);
  double val = s1(1.5, 1.5);

  spline::BivariateSpline s2(std::move(s1));
  EXPECT_NEAR(s2(1.5, 1.5), val, 1e-15);
}

TEST(BivariateSpline, MoveAssignment) {
  const int nx = 4;
  const int ny = 4;
  std::vector<double> x(nx);
  std::vector<double> y(ny);
  std::vector<double> z1(nx * ny);
  std::vector<double> z2(nx * ny);
  for (int i = 0; i < nx; ++i) {
    x[i] = static_cast<double>(i);
  }
  for (int j = 0; j < ny; ++j) {
    y[j] = static_cast<double>(j);
  }
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      auto idx = static_cast<std::size_t>(i * ny + j);
      z1[idx] = x[i] + y[j];
      z2[idx] = x[i] * y[j];
    }
  }

  spline::BivariateSpline s1(x, y, z1);
  spline::BivariateSpline s2(x, y, z2);
  double val = s1(1.5, 1.5);

  s2 = std::move(s1);
  EXPECT_NEAR(s2(1.5, 1.5), val, 1e-15);
}
