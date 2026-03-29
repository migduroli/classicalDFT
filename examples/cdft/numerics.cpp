// ── Numerics example (modern cdft API) ──────────────────────────────────────
//
// Demonstrates:
//   - GSL integration wrappers (QAGS, QNG, semi-infinite)
//   - Fourier transforms (FFTW3)
//   - Cubic spline interpolation
//   - Autodiff convenience functions

#include <cdft.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>

int main() {
  using namespace cdft::numerics;

  std::cout << std::fixed << std::setprecision(10);

  // ── Integration: QAGS ───────────────────────────────────────────────────

  std::cout << "GSL integration\n";
  std::cout << std::string(60, '-') << "\n";

  auto result_qags = integrate_qags(
      [](double x) { return std::sin(x) * std::sin(x); },
      0.0, std::numbers::pi
  );
  std::cout << "  integral sin^2(x) dx [0, pi]  = " << result_qags.value
            << "  (exact: " << std::numbers::pi / 2.0 << ")\n";
  std::cout << "  error estimate                 = " << result_qags.error << "\n";

  // ── Integration: QNG (non-adaptive) ─────────────────────────────────────

  auto result_qng = integrate_qng(
      [](double x) { return x * x; },
      0.0, 1.0
  );
  std::cout << "\n  integral x^2 dx [0, 1]        = " << result_qng.value
            << "  (exact: " << 1.0 / 3.0 << ")\n";

  // ── Integration: semi-infinite ──────────────────────────────────────────

  auto result_inf = integrate_semi_infinite(
      [](double x) { return std::exp(-x); },
      0.0
  );
  std::cout << "  integral e^(-x) dx [0, inf)   = " << result_inf.value
            << "  (exact: 1.0)\n";

  // ── Fourier transform ──────────────────────────────────────────────────

  std::cout << "\n\nFourier transform round-trip\n";
  std::cout << std::string(60, '-') << "\n";

  auto ft = FourierTransform({64, 1, 1});
  auto real = ft.real();

  for (long i = 0; i < ft.total(); ++i) {
    real[i] = std::sin(2.0 * std::numbers::pi * i / 64.0);
  }

  double original_sum = 0.0;
  for (long i = 0; i < ft.total(); ++i) {
    original_sum += real[i] * real[i];
  }

  ft.forward();
  ft.backward();
  ft.scale(1.0 / static_cast<double>(ft.total()));

  double restored_sum = 0.0;
  auto restored = ft.real();
  for (long i = 0; i < ft.total(); ++i) {
    restored_sum += restored[i] * restored[i];
  }

  std::cout << "  Original sum(x^2)  = " << original_sum << "\n";
  std::cout << "  Restored sum(x^2)  = " << restored_sum << "\n";
  std::cout << "  Difference         = " << std::abs(original_sum - restored_sum) << "\n";

  // ── Cubic spline ───────────────────────────────────────────────────────

  std::cout << "\n\nCubic spline interpolation\n";
  std::cout << std::string(60, '-') << "\n";

  constexpr int N = 10;
  std::array<double, N> xs{};
  std::array<double, N> ys{};
  for (int i = 0; i < N; ++i) {
    xs[i] = static_cast<double>(i);
    ys[i] = std::sin(xs[i]);
  }

  auto spline = CubicSpline(std::span{xs}, std::span{ys});

  std::cout << std::setw(10) << "x" << std::setw(18) << "spline(x)" << std::setw(18) << "sin(x)"
            << std::setw(14) << "error\n";

  for (double x : {0.5, 1.5, 3.7, 5.2, 7.8}) {
    double interp = spline(x);
    double exact = std::sin(x);
    std::cout << std::setw(10) << x << std::setw(18) << interp << std::setw(18) << exact
              << std::setw(14) << std::abs(interp - exact) << "\n";
  }

  std::cout << "\n  Spline derivative at x=1.0   = " << spline.derivative(1.0)
            << "  (cos(1) = " << std::cos(1.0) << ")\n";
  std::cout << "  Spline integral [0, pi]      = " << spline.integrate(0.0, std::numbers::pi)
            << "  (exact: 2.0)\n";

  // ── Autodiff convenience ───────────────────────────────────────────────

  std::cout << "\n\nAutodiff: derivatives of exp(x) at x = 1\n";
  std::cout << std::string(40, '-') << "\n";

  auto [f, df, d2f, d3f] = cdft::derivatives_up_to_3(
      [](cdft::dual3rd x) -> cdft::dual3rd { return exp(x); },
      1.0
  );

  double e = std::exp(1.0);
  std::cout << "  f(1)    = " << f << "  (exact: " << e << ")\n";
  std::cout << "  f'(1)   = " << df << "  (exact: " << e << ")\n";
  std::cout << "  f''(1)  = " << d2f << "  (exact: " << e << ")\n";
  std::cout << "  f'''(1) = " << d3f << "  (exact: " << e << ")\n";

  return 0;
}
