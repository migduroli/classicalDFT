#include "dft.hpp"
#include "plot.hpp"

#include <algorithm>
#include <armadillo>
#include <filesystem>
#include <iostream>
#include <numbers>
#include <numeric>
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

  // Convolution theorem: delta * f = f.

  console::info("Convolution: delta * constant = constant");

  auto shape = std::vector<long>{ 16, 16, 16 };
  long N = shape[0] * shape[1] * shape[2];

  auto plan_a = math::FourierTransform(shape);
  auto plan_b = math::FourierTransform(shape);

  auto real_a = plan_a.real();
  std::fill(real_a.begin(), real_a.end(), 0.0);
  real_a[0] = 1.0;

  auto real_b = plan_b.real();
  std::fill(real_b.begin(), real_b.end(), 3.0);

  plan_a.forward();
  plan_b.forward();

  auto result = math::convolve(plan_a.fourier(), plan_b.fourier(), shape);
  result /= static_cast<double>(N);

  std::println(std::cout, "  result range: [{}, {}]", result.min(), result.max());
  std::println(std::cout, "  expected:     3.0 everywhere");
  std::println(std::cout, "  max error:    {:.2e}", arma::max(arma::abs(result - 3.0)));

  // Self-convolution of a Gaussian.

  console::info("Self-convolution of a Gaussian");

  double sigma = 1.5;
  double dx = 0.5;
  auto shape_1d = std::vector<long>{ 64, 1, 1 };
  long N_1d = shape_1d[0];
  double L = N_1d * dx;

  auto plan_g = math::FourierTransform(shape_1d);
  auto real_g = plan_g.real();
  for (long i = 0; i < N_1d; ++i) {
    double x = i * dx;
    if (x > L / 2.0)
      x -= L;
    real_g[static_cast<std::size_t>(i)] = std::exp(-x * x / (2.0 * sigma * sigma));
  }

  plan_g.forward();
  auto g_k = plan_g.fourier();

  auto gg = math::convolve(g_k, g_k, shape_1d);
  gg *= dx / static_cast<double>(N_1d);

  double sigma_out = sigma * std::sqrt(2.0);
  double norm_factor = sigma * std::sqrt(std::numbers::pi);

  std::println(std::cout, "  sigma_in:   {:.6f}", sigma);
  std::println(std::cout, "  sigma_out:  {:.6f} (expected: sigma * sqrt(2))", sigma_out);

  std::println(std::cout, "\n  {:>8s}{:>14s}{:>14s}{:>14s}", "x", "numerical", "analytical", "error");
  std::println(std::cout, "  {}", std::string(48, '-'));

  double max_error = 0.0;
  for (long i = 0; i < N_1d; i += 4) {
    double x = i * dx;
    if (x > L / 2.0)
      x -= L;
    double numerical = gg(static_cast<arma::uword>(i));
    double analytical = norm_factor * std::exp(-x * x / (2.0 * sigma_out * sigma_out));
    double error = std::abs(numerical - analytical);
    max_error = std::max(max_error, error);

    if (i < 24 || i >= N_1d - 8) {
      std::println(std::cout, "  {:>8.2f}{:>14.6f}{:>14.6f}{:>14.2e}", x, numerical, analytical, error);
    }
  }
  std::println(std::cout, "\n  max error: {:.2e}", max_error);

  // Collect plot data (sorted by x for clean rendering).

  std::vector<double> x_plot, num_plot, ana_plot;
  for (long i = 0; i < N_1d; ++i) {
    double xp = i * dx;
    if (xp > L / 2.0)
      xp -= L;
    x_plot.push_back(xp);
    num_plot.push_back(gg(static_cast<arma::uword>(i)));
    ana_plot.push_back(norm_factor * std::exp(-xp * xp / (2.0 * sigma_out * sigma_out)));
  }
  std::vector<std::size_t> idx(x_plot.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return x_plot[a] < x_plot[b]; });
  std::vector<double> xs(x_plot.size()), ns(x_plot.size()), as(x_plot.size());
  for (std::size_t i = 0; i < idx.size(); ++i) {
    xs[i] = x_plot[idx[i]];
    ns[i] = num_plot[idx[i]];
    as[i] = ana_plot[idx[i]];
  }

  // Back-convolution symmetry.

  console::info("Back-convolution: adjoint property");

  auto plan_rho = math::FourierTransform(shape);
  auto plan_d = math::FourierTransform(shape);

  arma::vec rho_r = arma::randu(static_cast<arma::uword>(N));
  arma::vec d_r = arma::randu(static_cast<arma::uword>(N));
  arma::vec w_r = arma::randu(static_cast<arma::uword>(N));

  auto plan_w = math::FourierTransform(shape);
  plan_w.set_real(w_r);
  plan_w.forward();
  auto w_k = plan_w.fourier_vec();

  plan_rho.set_real(rho_r);
  plan_rho.forward();
  auto rho_k = plan_rho.fourier();

  auto n = math::convolve(
      std::span<const std::complex<double>>{ w_k.memptr(), static_cast<std::size_t>(w_k.n_elem) },
      rho_k,
      shape
  );
  double lhs = arma::dot(n, d_r);

  auto bc = math::back_convolve(
      std::span<const std::complex<double>>{ w_k.memptr(), static_cast<std::size_t>(w_k.n_elem) },
      d_r,
      shape,
      true
  );
  math::FourierTransform scratch(shape);
  scratch.set_fourier(bc);
  scratch.backward();
  auto bc_r = scratch.real_vec();
  double rhs = arma::dot(rho_r, bc_r);

  std::println(std::cout, "  <convolve(w, rho), d>   = {:.10f}", lhs);
  std::println(std::cout, "  <rho, IFFT[back(w, d)]> = {:.10f}", rhs);
  std::println(std::cout, "  relative error:           {:.2e}", std::abs(lhs - rhs) / std::abs(lhs));

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(xs, ns, as);
#endif
}
