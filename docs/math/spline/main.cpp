#include "dft.hpp"

#include <armadillo>
#include <iomanip>
#include <iostream>
#include <numbers>

using namespace dft;

int main() {

  // CubicSpline: interpolating sin(x).

  console::info("CubicSpline: interpolating sin(x)");

  constexpr int n = 10;
  auto x = arma::linspace(0.0, 2.0 * std::numbers::pi, n);
  arma::vec y = arma::sin(x);

  auto sin_spline = math::CubicSpline({x.memptr(), x.n_elem}, {y.memptr(), y.n_elem});

  auto x_eval = arma::linspace(0.0, 2.0 * std::numbers::pi, 2 * n);
  arma::vec exact = arma::sin(x_eval);

  std::cout << "\n  " << std::setw(10) << "x" << std::setw(14) << "spline(x)"
            << std::setw(14) << "sin(x)" << std::setw(14) << "error" << "\n";
  std::cout << "  " << std::string(52, '-') << "\n";

  arma::vec errors(x_eval.n_elem);
  for (arma::uword i = 0; i < x_eval.n_elem; ++i) {
    double s = sin_spline(x_eval(i));
    errors(i) = std::abs(s - exact(i));
    std::cout << "  " << std::fixed << std::setprecision(4)
              << std::setw(10) << x_eval(i)
              << std::setw(14) << s
              << std::setw(14) << exact(i)
              << std::scientific << std::setw(14) << errors(i) << "\n";
  }

  std::cout << "\n  max interpolation error: " << errors.max() << "\n";

  // Derivatives at pi/4.

  console::info("Derivatives at x = pi/4");
  double x0 = std::numbers::pi / 4.0;
  std::cout << "  f(x)  = " << sin_spline(x0) << "  (exact: " << std::sin(x0) << ")\n";
  std::cout << "  f'(x) = " << sin_spline.derivative(x0) << "  (exact: " << std::cos(x0) << ")\n";
  std::cout << "  f\"(x) = " << sin_spline.derivative2(x0) << "  (exact: " << -std::sin(x0) << ")\n";

  // Integration.

  console::info("Integration");
  double integral = sin_spline.integrate(0.0, std::numbers::pi);
  std::cout << "  int[0, pi] sin(x) dx = " << integral << "\n";
  std::cout << "  exact:                  2.0\n";
  std::cout << "  error:                  " << std::abs(integral - 2.0) << "\n";

  // BivariateSpline: interpolating sin(x)*cos(y).

  console::info("BivariateSpline: sin(x)*cos(y)");

  constexpr int nx = 20;
  constexpr int ny = 20;
  auto bx = arma::linspace(0.0, 2.0 * std::numbers::pi, nx);
  auto by = arma::linspace(0.0, 2.0 * std::numbers::pi, ny);

  // Outer product sin(x) * cos(y), stored column-major (matching bz(j*nx+i)).
  arma::vec bz = arma::vectorise(arma::sin(bx) * arma::cos(by).t());

  auto surface = math::BivariateSpline(
      {bx.memptr(), bx.n_elem}, {by.memptr(), by.n_elem}, {bz.memptr(), bz.n_elem}
  );

  arma::mat test_points = {{std::numbers::pi / 4.0, std::numbers::pi / 4.0},
                            {std::numbers::pi / 3.0, std::numbers::pi / 6.0},
                            {std::numbers::pi / 2.0, std::numbers::pi / 4.0},
                            {2.0, 1.0}};

  std::cout << "\n  " << std::setw(8) << "x" << std::setw(8) << "y"
            << std::setw(16) << "spline" << std::setw(16) << "exact"
            << std::setw(16) << "error" << "\n";
  std::cout << "  " << std::string(64, '-') << "\n";

  for (arma::uword row = 0; row < test_points.n_rows; ++row) {
    double tx = test_points(row, 0);
    double ty = test_points(row, 1);
    double sv = surface(tx, ty);
    double ev = std::sin(tx) * std::cos(ty);
    std::cout << "  " << std::fixed << std::setprecision(4)
              << std::setw(8) << tx << std::setw(8) << ty
              << std::setprecision(8) << std::setw(16) << sv
              << std::setw(16) << ev
              << std::scientific << std::setw(16) << std::abs(sv - ev) << "\n";
  }

  // Partial derivatives.

  console::info("Partial derivatives at (pi/4, pi/4)");
  double px = std::numbers::pi / 4.0;
  double py = std::numbers::pi / 4.0;
  std::cout << "  df/dx   = " << surface.deriv_x(px, py)
            << "  (exact: " << std::cos(px) * std::cos(py) << ")\n";
  std::cout << "  df/dy   = " << surface.deriv_y(px, py)
            << "  (exact: " << -std::sin(px) * std::sin(py) << ")\n";
  std::cout << "  d2f/dxy = " << surface.deriv_xy(px, py)
            << "  (exact: " << -std::cos(px) * std::sin(py) << ")\n";
}
