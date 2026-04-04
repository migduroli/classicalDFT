#include "dft.hpp"

#include <armadillo>
#include <iomanip>
#include <iostream>
#include <numbers>

using namespace dft;

int main() {
  std::cout << std::fixed;

  // Convolution theorem: delta * f = f.

  console::info("Convolution: delta * constant = constant");

  auto shape = std::vector<long>{16, 16, 16};
  long N = shape[0] * shape[1] * shape[2];

  auto plan_a = math::FourierTransform(shape);
  auto plan_b = math::FourierTransform(shape);

  // Real-space delta at the origin.
  auto real_a = plan_a.real();
  std::fill(real_a.begin(), real_a.end(), 0.0);
  real_a[0] = 1.0;

  // Uniform function f(r) = 3.0.
  auto real_b = plan_b.real();
  std::fill(real_b.begin(), real_b.end(), 3.0);

  plan_a.forward();
  plan_b.forward();

  auto result = math::convolve(plan_a.fourier(), plan_b.fourier(), shape);
  result /= static_cast<double>(N);

  std::cout << "  result range: [" << result.min() << ", " << result.max() << "]\n";
  std::cout << "  expected:     3.0 everywhere\n";
  std::cout << "  max error:    " << std::setprecision(2) << std::scientific
            << arma::max(arma::abs(result - 3.0)) << "\n";

  // Self-convolution of a Gaussian.

  console::info("Self-convolution of a Gaussian");

  double sigma = 1.5;
  double dx = 0.5;
  auto shape_1d = std::vector<long>{64, 1, 1};
  long N_1d = shape_1d[0];
  double L = N_1d * dx;

  // Build Gaussian g(x) = exp(-x^2 / (2*sigma^2)) on a periodic grid.
  auto plan_g = math::FourierTransform(shape_1d);
  auto real_g = plan_g.real();
  for (long i = 0; i < N_1d; ++i) {
    double x = i * dx;
    if (x > L / 2.0) x -= L;
    real_g[static_cast<std::size_t>(i)] = std::exp(-x * x / (2.0 * sigma * sigma));
  }

  plan_g.forward();
  auto g_k = plan_g.fourier();

  // convolve(g, g) = IFFT[g_k .* g_k] * dx.
  auto gg = math::convolve(g_k, g_k, shape_1d);
  gg *= dx / static_cast<double>(N_1d);

  // Analytical: Gaussian * Gaussian = Gaussian with sigma_out = sqrt(2) * sigma
  // and amplitude sigma * sqrt(pi).
  double sigma_out = sigma * std::sqrt(2.0);
  double norm_factor = sigma * std::sqrt(std::numbers::pi);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "  sigma_in:   " << sigma << "\n";
  std::cout << "  sigma_out:  " << sigma_out << " (expected: sigma * sqrt(2))\n\n";

  std::cout << std::setw(8) << "x" << std::setw(14) << "numerical"
            << std::setw(14) << "analytical" << std::setw(14) << "error" << "\n";
  std::cout << "  " << std::string(48, '-') << "\n";

  double max_error = 0.0;
  for (long i = 0; i < N_1d; i += 4) {
    double x = i * dx;
    if (x > L / 2.0) x -= L;
    double numerical = gg(static_cast<arma::uword>(i));
    double analytical = norm_factor * std::exp(-x * x / (2.0 * sigma_out * sigma_out));
    double error = std::abs(numerical - analytical);
    max_error = std::max(max_error, error);

    if (i < 24 || i >= N_1d - 8) {
      std::cout << std::setw(8) << x
                << std::setw(14) << numerical
                << std::setw(14) << analytical
                << std::scientific << std::setw(14) << error << "\n";
      std::cout << std::fixed;
    }
  }
  std::cout << "\n  max error: " << std::scientific << max_error << "\n";

  // Back-convolution symmetry.

  console::info("Back-convolution: adjoint property");

  // For a general weight w, the adjoint of convolution uses the
  // time-reversed kernel w~(r) = w(-r), whose FFT is conj(w_k).
  // back_convolve(w_k, d, shape, conjugate=true) implements this:
  //   <convolve(w_k, rho_k), d> == <rho, IFFT[conj(w_k) .* FFT(d)]>

  auto plan_rho = math::FourierTransform(shape);
  auto plan_d = math::FourierTransform(shape);

  arma::vec rho_r = arma::randu(static_cast<arma::uword>(N));
  arma::vec d_r = arma::randu(static_cast<arma::uword>(N));
  arma::vec w_r = arma::randu(static_cast<arma::uword>(N));

  // Compute w_k.
  auto plan_w = math::FourierTransform(shape);
  plan_w.set_real(w_r);
  plan_w.forward();
  auto w_k = plan_w.fourier_vec();

  // rho_k.
  plan_rho.set_real(rho_r);
  plan_rho.forward();
  auto rho_k = plan_rho.fourier();

  // Forward: n = convolve(w_k, rho_k), then <n, d>.
  auto n = math::convolve(
      std::span<const std::complex<double>>{w_k.memptr(), static_cast<std::size_t>(w_k.n_elem)},
      rho_k, shape
  );
  double lhs = arma::dot(n, d_r);

  // Adjoint: bc = back_convolve(w_k, d, conjugate=true), then <rho, IFFT[bc]>.
  auto bc = math::back_convolve(
      std::span<const std::complex<double>>{w_k.memptr(), static_cast<std::size_t>(w_k.n_elem)},
      d_r, shape, true
  );
  // IFFT of bc.
  math::FourierTransform scratch(shape);
  scratch.set_fourier(bc);
  scratch.backward();
  auto bc_r = scratch.real_vec();
  double rhs = arma::dot(rho_r, bc_r);

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "  <convolve(w, rho), d> = " << lhs << "\n";
  std::cout << "  <rho, IFFT[back(w, d)]> = " << rhs << "\n";
  std::cout << "  relative error:         " << std::scientific
            << std::abs(lhs - rhs) / std::abs(lhs) << "\n";
}
