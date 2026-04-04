#include "dft.hpp"
#include "plot.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

using namespace dft;
using namespace dft::math;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  std::cout << std::fixed << std::setprecision(12);

  // First derivatives via autodiff.

  console::info("First derivatives: derivatives_up_to_1");

  // Lambdas must accept the explicit dual type matching the derivatives_up_to_N level.
  // The autodiff library provides sin, cos, exp, log overloads via autodiff::detail.

  std::cout << "\n  " << std::setw(14) << "function"
            << std::setw(8) << "x"
            << std::setw(18) << "f(x)"
            << std::setw(18) << "f'(x)"
            << std::setw(18) << "f'(x) exact"
            << std::setw(14) << "error\n";
  std::cout << "  " << std::string(90, '-') << "\n";

  auto [sv, sd] = derivatives_up_to_1(
      [](dual x) -> dual { return autodiff::detail::sin(x); },
      std::numbers::pi / 4.0
  );
  double sd_exact = std::cos(std::numbers::pi / 4.0);
  std::cout << "  " << std::setw(14) << "sin(x)"
            << std::setw(8) << "pi/4"
            << std::setw(18) << sv
            << std::setw(18) << sd
            << std::setw(18) << sd_exact
            << std::scientific << std::setw(14) << std::abs(sd - sd_exact) << "\n";
  std::cout << std::fixed;

  auto [ev, ed] = derivatives_up_to_1(
      [](dual x) -> dual { return autodiff::detail::exp(x); }, 1.0
  );
  std::cout << "  " << std::setw(14) << "exp(x)"
            << std::setw(8) << "1.0"
            << std::setw(18) << ev
            << std::setw(18) << ed
            << std::setw(18) << std::exp(1.0)
            << std::scientific << std::setw(14) << std::abs(ed - std::exp(1.0)) << "\n";
  std::cout << std::fixed;

  auto [pv, pd] = derivatives_up_to_1(
      [](dual x) -> dual { return x * x * x - 2.0 * x * x + 3.0 * x - 1.0; }, 2.0
  );
  double pd_exact = 3.0 * 4.0 - 4.0 * 2.0 + 3.0;
  std::cout << "  " << std::setw(14) << "x^3-2x^2+3x-1"
            << std::setw(8) << "2.0"
            << std::setw(18) << pv
            << std::setw(18) << pd
            << std::setw(18) << pd_exact
            << std::scientific << std::setw(14) << std::abs(pd - pd_exact) << "\n";
  std::cout << std::fixed;

  // Second derivatives.

  console::info("Second derivatives: derivatives_up_to_2");

  double x0 = std::numbers::pi / 6.0;
  auto [s2v, s2d, s2d2] = derivatives_up_to_2(
      [](dual2nd x) -> dual2nd { return autodiff::detail::sin(x); }, x0
  );

  std::cout << "\n  sin(x) at x = pi/6:\n";
  std::cout << "    f(x)   = " << s2v   << "  (exact: " << std::sin(x0) << ")\n";
  std::cout << "    f'(x)  = " << s2d   << "  (exact: " << std::cos(x0) << ")\n";
  std::cout << "    f''(x) = " << s2d2  << "  (exact: " << -std::sin(x0) << ")\n";

  auto [e2v, e2d, e2d2] = derivatives_up_to_2(
      [](dual2nd x) -> dual2nd { return autodiff::detail::exp(x); }, 2.0
  );
  std::cout << "\n  exp(x) at x = 2:\n";
  std::cout << "    f(x)   = " << e2v  << "  (exact: " << std::exp(2.0) << ")\n";
  std::cout << "    f'(x)  = " << e2d  << "  (exact: " << std::exp(2.0) << ")\n";
  std::cout << "    f''(x) = " << e2d2 << "  (exact: " << std::exp(2.0) << ")\n";

  // Third derivatives.

  console::info("Third derivatives: derivatives_up_to_3");

  auto [s3v, s3d, s3d2, s3d3] = derivatives_up_to_3(
      [](dual3rd x) -> dual3rd { return autodiff::detail::sin(x); }, x0
  );

  std::cout << "\n  sin(x) at x = pi/6:\n";
  std::cout << "    f(x)    = " << s3v   << "  (exact: " << std::sin(x0) << ")\n";
  std::cout << "    f'(x)   = " << s3d   << "  (exact: " << std::cos(x0) << ")\n";
  std::cout << "    f''(x)  = " << s3d2  << "  (exact: " << -std::sin(x0) << ")\n";
  std::cout << "    f'''(x) = " << s3d3  << "  (exact: " << -std::cos(x0) << ")\n";

  auto [p3v, p3d, p3d2, p3d3] = derivatives_up_to_3(
      [](dual3rd x) -> dual3rd { return x * x * x - 2.0 * x * x + 3.0 * x - 1.0; }, 2.0
  );

  std::cout << "\n  x^3 - 2x^2 + 3x - 1 at x = 2:\n";
  std::cout << "    f(x)    = " << p3v   << "  (exact: " << (8.0 - 8.0 + 6.0 - 1.0) << ")\n";
  std::cout << "    f'(x)   = " << p3d   << "  (exact: " << (3.0 * 4.0 - 4.0 * 2.0 + 3.0) << ")\n";
  std::cout << "    f''(x)  = " << p3d2  << "  (exact: " << (6.0 * 2.0 - 4.0) << ")\n";
  std::cout << "    f'''(x) = " << p3d3  << "  (exact: " << 6.0 << ")\n";

  // Comparison with finite differences.

  console::info("Autodiff vs finite differences");

  double xc = 1.5;
  double h = 1e-5;

  auto [tv, td, td2] = derivatives_up_to_2(
      [](dual2nd x) -> dual2nd { return autodiff::detail::log(1.0 + x * x); }, xc
  );

  double fd1 = (std::log(1.0 + (xc + h) * (xc + h)) - std::log(1.0 + (xc - h) * (xc - h))) / (2.0 * h);
  double fd2 = (std::log(1.0 + (xc + h) * (xc + h))
                - 2.0 * std::log(1.0 + xc * xc)
                + std::log(1.0 + (xc - h) * (xc - h))) / (h * h);

  double exact_d1 = 2.0 * xc / (1.0 + xc * xc);
  double exact_d2 = 2.0 * (1.0 - xc * xc) / ((1.0 + xc * xc) * (1.0 + xc * xc));

  std::cout << "\n  log(1 + x^2) at x = " << xc << ":\n\n";
  std::cout << std::setw(16) << "" << std::setw(18) << "autodiff"
            << std::setw(18) << "finite diff"
            << std::setw(18) << "exact" << "\n";
  std::cout << "  " << std::string(68, '-') << "\n";
  std::cout << std::setw(16) << "f'(x)"
            << std::setw(18) << td
            << std::setw(18) << fd1
            << std::setw(18) << exact_d1 << "\n";
  std::cout << std::setw(16) << "f''(x)"
            << std::setw(18) << td2
            << std::setw(18) << fd2
            << std::setw(18) << exact_d2 << "\n";
  std::cout << "\n  autodiff error in f':  " << std::scientific << std::abs(td - exact_d1) << "\n";
  std::cout << "  fin. diff error in f': " << std::abs(fd1 - exact_d1) << "\n";
  std::cout << "  autodiff error in f'': " << std::abs(td2 - exact_d2) << "\n";
  std::cout << "  fin. diff error in f'': " << std::abs(fd2 - exact_d2) << "\n";

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  {
    // Plot sin(x) and its autodiff derivatives over [0, 2pi].
    int np = 200;
    std::vector<double> xp(np), fp(np), d1p(np), d2p(np);
    for (int i = 0; i < np; ++i) {
      double xi = 2.0 * std::numbers::pi * i / (np - 1);
      xp[i] = xi;
      auto [v, d, dd] = derivatives_up_to_2(
          [](dual2nd t) -> dual2nd { return autodiff::detail::sin(t); }, xi);
      fp[i] = v;
      d1p[i] = d;
      d2p[i] = dd;
    }
    plot::function_and_derivatives(
        xp, fp, d1p, d2p,
        R"(Autodiff derivatives of $\sin(x)$)",
        R"($\sin(x)$)", R"($\cos(x)$)", R"($-\sin(x)$)",
        "exports/autodiff_sin.png");
  }

  {
    // Accuracy comparison: autodiff vs finite differences for log(1+x^2).
    int np = 100;
    double hf = 1e-5;
    std::vector<double> xp(np), ae1(np), fe1(np), ae2(np), fe2(np);
    auto f_exact = [](double x) { return std::log(1.0 + x * x); };
    auto d1_exact = [](double x) { return 2.0 * x / (1.0 + x * x); };
    auto d2_exact = [](double x) { return 2.0 * (1.0 - x * x) / ((1.0 + x * x) * (1.0 + x * x)); };

    for (int i = 0; i < np; ++i) {
      double xi = 0.1 + 3.0 * i / (np - 1);
      xp[i] = xi;
      auto [v, d, dd] = derivatives_up_to_2(
          [](dual2nd t) -> dual2nd { return autodiff::detail::log(1.0 + t * t); }, xi);
      double fd1_val = (f_exact(xi + hf) - f_exact(xi - hf)) / (2.0 * hf);
      double fd2_val = (f_exact(xi + hf) - 2.0 * f_exact(xi) + f_exact(xi - hf)) / (hf * hf);
      ae1[i] = std::max(std::abs(d - d1_exact(xi)), 1e-18);
      fe1[i] = std::max(std::abs(fd1_val - d1_exact(xi)), 1e-18);
      ae2[i] = std::max(std::abs(dd - d2_exact(xi)), 1e-18);
      fe2[i] = std::max(std::abs(fd2_val - d2_exact(xi)), 1e-18);
    }
    plot::autodiff_vs_finite_diff(xp, ae1, fe1, ae2, fe2);
  }
#endif
}
