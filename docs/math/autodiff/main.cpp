#include "dft.hpp"
#include "plot.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <print>
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

  // First derivatives via autodiff.

  console::info("First derivatives: derivatives_up_to_1");

  std::println(std::cout, "\n  {:>14s}{:>8s}{:>18s}{:>18s}{:>18s}{:>14s}",
               "function", "x", "f(x)", "f'(x)", "f'(x) exact", "error");
  std::println(std::cout, "  {}", std::string(90, '-'));

  auto [sv, sd] = derivatives_up_to_1(
      [](dual x) -> dual { return autodiff::detail::sin(x); },
      std::numbers::pi / 4.0
  );
  double sd_exact = std::cos(std::numbers::pi / 4.0);
  std::println(std::cout, "  {:>14s}{:>8s}{:>18.12f}{:>18.12f}{:>18.12f}{:>14.4e}",
               "sin(x)", "pi/4", sv, sd, sd_exact, std::abs(sd - sd_exact));

  auto [ev, ed] = derivatives_up_to_1(
      [](dual x) -> dual { return autodiff::detail::exp(x); }, 1.0
  );
  std::println(std::cout, "  {:>14s}{:>8s}{:>18.12f}{:>18.12f}{:>18.12f}{:>14.4e}",
               "exp(x)", "1.0", ev, ed, std::exp(1.0), std::abs(ed - std::exp(1.0)));

  auto [pv, pd] = derivatives_up_to_1(
      [](dual x) -> dual { return x * x * x - 2.0 * x * x + 3.0 * x - 1.0; }, 2.0
  );
  double pd_exact = 3.0 * 4.0 - 4.0 * 2.0 + 3.0;
  std::println(std::cout, "  {:>14s}{:>8s}{:>18.12f}{:>18.12f}{:>18.12f}{:>14.4e}",
               "x^3-2x^2+3x-1", "2.0", pv, pd, pd_exact, std::abs(pd - pd_exact));

  // Second derivatives.

  console::info("Second derivatives: derivatives_up_to_2");

  double x0 = std::numbers::pi / 6.0;
  auto [s2v, s2d, s2d2] = derivatives_up_to_2(
      [](dual2nd x) -> dual2nd { return autodiff::detail::sin(x); }, x0
  );

  std::println(std::cout, "\n  sin(x) at x = pi/6:");
  std::println(std::cout, "    f(x)   = {:.12f}  (exact: {:.12f})", s2v, std::sin(x0));
  std::println(std::cout, "    f'(x)  = {:.12f}  (exact: {:.12f})", s2d, std::cos(x0));
  std::println(std::cout, "    f''(x) = {:.12f}  (exact: {:.12f})", s2d2, -std::sin(x0));

  auto [e2v, e2d, e2d2] = derivatives_up_to_2(
      [](dual2nd x) -> dual2nd { return autodiff::detail::exp(x); }, 2.0
  );
  std::println(std::cout, "\n  exp(x) at x = 2:");
  std::println(std::cout, "    f(x)   = {:.12f}  (exact: {:.12f})", e2v, std::exp(2.0));
  std::println(std::cout, "    f'(x)  = {:.12f}  (exact: {:.12f})", e2d, std::exp(2.0));
  std::println(std::cout, "    f''(x) = {:.12f}  (exact: {:.12f})", e2d2, std::exp(2.0));

  // Third derivatives.

  console::info("Third derivatives: derivatives_up_to_3");

  auto [s3v, s3d, s3d2, s3d3] = derivatives_up_to_3(
      [](dual3rd x) -> dual3rd { return autodiff::detail::sin(x); }, x0
  );

  std::println(std::cout, "\n  sin(x) at x = pi/6:");
  std::println(std::cout, "    f(x)    = {:.12f}  (exact: {:.12f})", s3v, std::sin(x0));
  std::println(std::cout, "    f'(x)   = {:.12f}  (exact: {:.12f})", s3d, std::cos(x0));
  std::println(std::cout, "    f''(x)  = {:.12f}  (exact: {:.12f})", s3d2, -std::sin(x0));
  std::println(std::cout, "    f'''(x) = {:.12f}  (exact: {:.12f})", s3d3, -std::cos(x0));

  auto [p3v, p3d, p3d2, p3d3] = derivatives_up_to_3(
      [](dual3rd x) -> dual3rd { return x * x * x - 2.0 * x * x + 3.0 * x - 1.0; }, 2.0
  );

  std::println(std::cout, "\n  x^3 - 2x^2 + 3x - 1 at x = 2:");
  std::println(std::cout, "    f(x)    = {:.12f}  (exact: {:.12f})", p3v, 8.0 - 8.0 + 6.0 - 1.0);
  std::println(std::cout, "    f'(x)   = {:.12f}  (exact: {:.12f})", p3d, 3.0 * 4.0 - 4.0 * 2.0 + 3.0);
  std::println(std::cout, "    f''(x)  = {:.12f}  (exact: {:.12f})", p3d2, 6.0 * 2.0 - 4.0);
  std::println(std::cout, "    f'''(x) = {:.12f}  (exact: {:.12f})", p3d3, 6.0);

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

  std::println(std::cout, "\n  log(1 + x^2) at x = {}:\n", xc);
  std::println(std::cout, "{:>16s}{:>18s}{:>18s}{:>18s}", "", "autodiff", "finite diff", "exact");
  std::println(std::cout, "  {}", std::string(68, '-'));
  std::println(std::cout, "{:>16s}{:>18.12f}{:>18.12f}{:>18.12f}", "f'(x)", td, fd1, exact_d1);
  std::println(std::cout, "{:>16s}{:>18.12f}{:>18.12f}{:>18.12f}", "f''(x)", td2, fd2, exact_d2);
  std::println(std::cout, "\n  autodiff error in f':  {:.4e}", std::abs(td - exact_d1));
  std::println(std::cout, "  fin. diff error in f': {:.4e}", std::abs(fd1 - exact_d1));
  std::println(std::cout, "  autodiff error in f'': {:.4e}", std::abs(td2 - exact_d2));
  std::println(std::cout, "  fin. diff error in f'': {:.4e}", std::abs(fd2 - exact_d2));

  // Collect plot data.

  int np_sin = 200;
  std::vector<double> x_sin(np_sin), f_sin(np_sin), d1_sin(np_sin), d2_sin(np_sin);
  for (int i = 0; i < np_sin; ++i) {
    double xi = 2.0 * std::numbers::pi * i / (np_sin - 1);
    x_sin[i] = xi;
    auto [v, d, dd] = derivatives_up_to_2(
        [](dual2nd t) -> dual2nd { return autodiff::detail::sin(t); }, xi);
    f_sin[i] = v;
    d1_sin[i] = d;
    d2_sin[i] = dd;
  }

  int np_err = 100;
  double hf = 1e-5;
  std::vector<double> x_err(np_err), ae1(np_err), fe1(np_err), ae2(np_err), fe2(np_err);
  auto f_exact_fn = [](double x) { return std::log(1.0 + x * x); };
  auto d1_exact_fn = [](double x) { return 2.0 * x / (1.0 + x * x); };
  auto d2_exact_fn = [](double x) { return 2.0 * (1.0 - x * x) / ((1.0 + x * x) * (1.0 + x * x)); };

  for (int i = 0; i < np_err; ++i) {
    double xi = 0.1 + 3.0 * i / (np_err - 1);
    x_err[i] = xi;
    auto [v, d, dd] = derivatives_up_to_2(
        [](dual2nd t) -> dual2nd { return autodiff::detail::log(1.0 + t * t); }, xi);
    double fd1_val = (f_exact_fn(xi + hf) - f_exact_fn(xi - hf)) / (2.0 * hf);
    double fd2_val = (f_exact_fn(xi + hf) - 2.0 * f_exact_fn(xi) + f_exact_fn(xi - hf)) / (hf * hf);
    ae1[i] = std::max(std::abs(d - d1_exact_fn(xi)), 1e-18);
    fe1[i] = std::max(std::abs(fd1_val - d1_exact_fn(xi)), 1e-18);
    ae2[i] = std::max(std::abs(dd - d2_exact_fn(xi)), 1e-18);
    fe2[i] = std::max(std::abs(fd2_val - d2_exact_fn(xi)), 1e-18);
  }

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(x_sin, f_sin, d1_sin, d2_sin, x_err, ae1, fe1, ae2, fe2);
#endif
}
