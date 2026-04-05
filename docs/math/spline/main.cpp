#include "dft.hpp"
#include "plot.hpp"

#include <armadillo>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <print>

using namespace dft;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  // CubicSpline: interpolating sin(x).

  console::info("CubicSpline: interpolating sin(x)");

  constexpr int n = 10;
  auto x = arma::linspace(0.0, 2.0 * std::numbers::pi, n);
  arma::vec y = arma::sin(x);

  auto sin_spline = math::CubicSpline({ x.memptr(), x.n_elem }, { y.memptr(), y.n_elem });

  auto x_eval = arma::linspace(0.0, 2.0 * std::numbers::pi, 2 * n);
  arma::vec exact = arma::sin(x_eval);

  std::println(std::cout, "\n  {:>10s}{:>14s}{:>14s}{:>14s}", "x", "spline(x)", "sin(x)", "error");
  std::println(std::cout, "  {}", std::string(52, '-'));

  arma::vec errors(x_eval.n_elem);
  for (arma::uword i = 0; i < x_eval.n_elem; ++i) {
    double s = sin_spline(x_eval(i));
    errors(i) = std::abs(s - exact(i));
    std::println(std::cout, "  {:>10.4f}{:>14.4f}{:>14.4f}{:>14.4e}", x_eval(i), s, exact(i), errors(i));
  }

  std::println(std::cout, "\n  max interpolation error: {:.4e}", errors.max());

  // Derivatives at pi/4.

  console::info("Derivatives at x = pi/4");
  double x0 = std::numbers::pi / 4.0;
  std::println(std::cout, "  f(x)  = {}  (exact: {})", sin_spline(x0), std::sin(x0));
  std::println(std::cout, "  f'(x) = {}  (exact: {})", sin_spline.derivative(x0), std::cos(x0));
  std::println(std::cout, "  f\"(x) = {}  (exact: {})", sin_spline.derivative2(x0), -std::sin(x0));

  // Integration.

  console::info("Integration");
  double integral = sin_spline.integrate(0.0, std::numbers::pi);
  std::println(std::cout, "  int[0, pi] sin(x) dx = {}", integral);
  std::println(std::cout, "  exact:                  2.0");
  std::println(std::cout, "  error:                  {}", std::abs(integral - 2.0));

  // BivariateSpline: interpolating sin(x)*cos(y).

  console::info("BivariateSpline: sin(x)*cos(y)");

  constexpr int nx = 20;
  constexpr int ny = 20;
  auto bx = arma::linspace(0.0, 2.0 * std::numbers::pi, nx);
  auto by = arma::linspace(0.0, 2.0 * std::numbers::pi, ny);

  arma::vec bz = arma::vectorise(arma::sin(bx) * arma::cos(by).t());

  auto surface =
      math::BivariateSpline({ bx.memptr(), bx.n_elem }, { by.memptr(), by.n_elem }, { bz.memptr(), bz.n_elem });

  arma::mat test_points = { { std::numbers::pi / 4.0, std::numbers::pi / 4.0 },
                            { std::numbers::pi / 3.0, std::numbers::pi / 6.0 },
                            { std::numbers::pi / 2.0, std::numbers::pi / 4.0 },
                            { 2.0, 1.0 } };

  std::println(std::cout, "\n  {:>8s}{:>8s}{:>16s}{:>16s}{:>16s}", "x", "y", "spline", "exact", "error");
  std::println(std::cout, "  {}", std::string(64, '-'));

  for (arma::uword row = 0; row < test_points.n_rows; ++row) {
    double tx = test_points(row, 0);
    double ty = test_points(row, 1);
    double sv = surface(tx, ty);
    double ev = std::sin(tx) * std::cos(ty);
    std::println(std::cout, "  {:>8.4f}{:>8.4f}{:>16.8f}{:>16.8f}{:>16.4e}", tx, ty, sv, ev, std::abs(sv - ev));
  }

  // Partial derivatives.

  console::info("Partial derivatives at (pi/4, pi/4)");
  double px = std::numbers::pi / 4.0;
  double py = std::numbers::pi / 4.0;
  std::println(std::cout, "  df/dx   = {}  (exact: {})", surface.deriv_x(px, py), std::cos(px) * std::cos(py));
  std::println(std::cout, "  df/dy   = {}  (exact: {})", surface.deriv_y(px, py), -std::sin(px) * std::sin(py));
  std::println(std::cout, "  d2f/dxy = {}  (exact: {})", surface.deriv_xy(px, py), -std::cos(px) * std::sin(py));

  // Collect plot data.

  auto x_knots_v = arma::conv_to<std::vector<double>>::from(x);
  auto y_knots_v = arma::conv_to<std::vector<double>>::from(y);

  auto x_fine = arma::linspace(0.0, 2.0 * std::numbers::pi, 200);
  std::vector<double> x_fine_v(x_fine.n_elem), y_spline_v(x_fine.n_elem), y_exact_v(x_fine.n_elem);
  std::vector<double> d1v(x_fine.n_elem), d2v(x_fine.n_elem);
  std::vector<double> ed1(x_fine.n_elem), ed2(x_fine.n_elem);
  for (arma::uword i = 0; i < x_fine.n_elem; ++i) {
    double xi = x_fine(i);
    x_fine_v[i] = xi;
    y_spline_v[i] = sin_spline(xi);
    y_exact_v[i] = std::sin(xi);
    d1v[i] = sin_spline.derivative(xi);
    d2v[i] = sin_spline.derivative2(xi);
    ed1[i] = std::cos(xi);
    ed2[i] = -std::sin(xi);
  }

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(x_knots_v, y_knots_v, x_fine_v, y_spline_v, y_exact_v, d1v, d2v, ed1, ed2);
#endif
}
