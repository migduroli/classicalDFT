#include <classicaldft>

#include <armadillo>
#include <cmath>
#include <iomanip>

int main() {
  using namespace dft_core::numerics::spline;
  using namespace dft_core::io;

  // ── CubicSpline: 1D natural cubic spline interpolation ───────────────────

  console::info("CubicSpline — interpolating sin(x)");

  // Sample sin(x) at 10 evenly spaced points on [0, 2*pi] — NumPy style
  constexpr int n = 10;
  auto x = arma::linspace(0.0, 2.0 * M_PI, n);
  arma::vec y = arma::sin(x);

  auto sin_spline = CubicSpline({x.memptr(), x.n_elem}, {y.memptr(), y.n_elem});

  std::cout << "  data points:  " << sin_spline.size() << std::endl;
  std::cout << "  domain:       [" << sin_spline.x_min() << ", " << sin_spline.x_max() << "]" << std::endl;

  // Evaluate at intermediate points and compare with sin(x)
  auto x_eval = arma::linspace(0.0, 2.0 * M_PI, 2 * n);
  arma::vec exact = arma::sin(x_eval);

  std::cout << std::endl;
  std::cout << "  " << std::setw(10) << "x" << std::setw(14) << "spline(x)" << std::setw(14) << "sin(x)"
            << std::setw(14) << "error" << std::endl;
  std::cout << "  " << std::string(52, '-') << std::endl;

  auto errors = arma::vec(x_eval.n_elem);
  for (arma::uword i = 0; i < x_eval.n_elem; ++i) {
    auto spline_val = sin_spline(x_eval(i));
    errors(i) = std::abs(spline_val - exact(i));
    std::cout << "  " << std::fixed << std::setprecision(4) << std::setw(10) << x_eval(i) << std::setw(14)
              << spline_val << std::setw(14) << exact(i) << std::scientific << std::setw(14) << errors(i) << std::endl;
  }

  std::cout << std::endl;
  std::cout << "  max interpolation error: " << errors.max() << std::endl;

  std::cout << std::endl;

  // ── Derivatives ──────────────────────────────────────────────────────────

  console::info("CubicSpline — derivatives at x = pi/4");

  auto x0 = M_PI / 4.0;
  std::cout << "    f(x)  = " << sin_spline(x0) << "   (exact: " << std::sin(x0) << ")" << std::endl;
  std::cout << "    f'(x) = " << sin_spline.derivative(x0) << "  (exact: " << std::cos(x0) << ")" << std::endl;
  std::cout << "    f\"(x) = " << sin_spline.derivative2(x0) << " (exact: " << -std::sin(x0) << ")" << std::endl;

  std::cout << std::endl;

  // ── Integration ──────────────────────────────────────────────────────────

  console::info("CubicSpline — integration");

  auto integral = sin_spline.integrate(0.0, M_PI);
  std::cout << "  int[0, pi] sin(x) dx = " << integral << std::endl;
  std::cout << "  exact:                  2.00000000" << std::endl;
  std::cout << "  error:                  " << std::abs(integral - 2.0) << std::endl;

  std::cout << std::endl;

  // ── BivariateSpline: 2D bicubic interpolation ───────────────────────────

  console::info("BivariateSpline — interpolating sin(x)*cos(y)");

  constexpr int nx = 20;
  constexpr int ny = 20;
  auto bx = arma::linspace(0.0, 2.0 * M_PI, nx);
  auto by = arma::linspace(0.0, 2.0 * M_PI, ny);

  // Build z-values: z[j*nx + i] = sin(bx[i]) * cos(by[j])  (GSL layout)
  auto bz = arma::vec(nx * ny);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      bz(j * nx + i) = std::sin(bx(i)) * std::cos(by(j));
    }
  }

  auto surface = BivariateSpline({bx.memptr(), bx.n_elem}, {by.memptr(), by.n_elem}, {bz.memptr(), bz.n_elem});

  // Evaluate at test points
  console::new_line();
  std::cout << "  " << std::setw(8) << "x" << std::setw(8) << "y" << std::setw(16) << "spline" << std::setw(16)
            << "exact" << std::setw(16) << "error" << std::endl;
  std::cout << "  " << std::string(64, '-') << std::endl;

  auto test_points = arma::mat({{M_PI / 4, M_PI / 4}, {M_PI / 3, M_PI / 6}, {M_PI / 2, M_PI / 4}, {2.0, 1.0}});

  for (arma::uword row = 0; row < test_points.n_rows; ++row) {
    auto tx = test_points(row, 0);
    auto ty = test_points(row, 1);
    auto spline_val = surface(tx, ty);
    auto exact_val = std::sin(tx) * std::cos(ty);
    auto error = std::abs(spline_val - exact_val);
    std::cout << "  " << std::fixed << std::setprecision(4) << std::setw(8) << tx << std::setw(8) << ty
              << std::setprecision(8) << std::setw(16) << spline_val << std::setw(16) << exact_val << std::scientific
              << std::setw(16) << error << std::endl;
  }

  console::new_line();

  // ── Partial derivatives ──────────────────────────────────────────────────

  console::info("BivariateSpline — partial derivatives at (pi/4, pi/4)");

  auto px = M_PI / 4.0;
  auto py = M_PI / 4.0;
  console::write_line("  df/dx   = " + std::to_string(surface.deriv_x(px, py)) +
                       "  (exact: " + std::to_string(std::cos(px) * std::cos(py)) + ")");
  console::write_line("  df/dy   = " + std::to_string(surface.deriv_y(px, py)) +
                       "  (exact: " + std::to_string(-std::sin(px) * std::sin(py)) + ")");
  console::write_line("  d2f/dxy = " + std::to_string(surface.deriv_xy(px, py)) +
                       "  (exact: " + std::to_string(-std::cos(px) * std::sin(py)) + ")");

  console::new_line();
  console::info("Done.");
}
