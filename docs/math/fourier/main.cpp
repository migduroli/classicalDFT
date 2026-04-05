#include <armadillo>
#include <dftlib>
#include <iostream>
#include <numbers>
#include <print>

using namespace dft;

int main() {
  // FourierTransform: RAII wrapper for FFTW3.

  console::info("FourierTransform: round-trip test");

  auto shape = std::vector<long>{8, 8, 8};
  auto plan = math::FourierTransform(shape);

  std::println(std::cout, "  grid:           {}x{}x{}", shape[0], shape[1], shape[2]);
  std::println(std::cout, "  total points:   {}", plan.total());
  std::println(std::cout, "  fourier points: {}", plan.fourier_total());

  auto grid_x = arma::linspace(0.0, 2.0 * std::numbers::pi * (1.0 - 1.0 / shape[0]), shape[0]);
  arma::vec sin_x = arma::sin(grid_x);

  // Fill real-space data: sin(x) replicated across y,z planes.
  // z-fastest layout: each x-slice is Ny*Nz contiguous values.
  arma::vec sin_3d = arma::repelem(sin_x, shape[1] * shape[2], 1);
  auto real = plan.real();
  std::copy(sin_3d.begin(), sin_3d.end(), real.begin());

  auto original = arma::vec(real.data(), real.size());

  plan.forward();

  auto fk = plan.fourier();
  std::println(std::cout, "  first 4 Fourier coefficients:");
  for (int n = 0; n < 4; ++n) {
    std::println(std::cout, "    k={} -> ({:.4}, {:.4})", n, fk[n].real(), fk[n].imag());
  }

  plan.backward();
  plan.scale(1.0 / static_cast<double>(plan.total()));

  auto roundtrip = arma::vec(plan.real().data(), plan.real().size());
  std::println(std::cout, "  round-trip max error: {}", arma::max(arma::abs(roundtrip - original)));

  // Parseval's theorem.

  console::info("Parseval's theorem");

  double real_energy = arma::dot(roundtrip, roundtrip);
  plan.forward();
  double fourier_energy = 0.0;
  for (auto c : plan.fourier()) {
    fourier_energy += std::norm(c);
  }
  fourier_energy /= static_cast<double>(plan.total());

  std::println(std::cout, "  real-space energy:    {:f}", real_energy);
  std::println(std::cout, "  fourier-space energy: {:f}", fourier_energy);
  std::println(std::cout, "  ratio:                {:f}", fourier_energy / real_energy);

  // FourierConvolution: delta * constant = constant.

  console::info("FourierConvolution: delta * constant");

  auto conv = math::FourierConvolution(shape);

  auto a = conv.input_a();
  std::fill(a.begin(), a.end(), 0.0);
  a[0] = 1.0;

  auto b = conv.input_b();
  std::fill(b.begin(), b.end(), 3.0);

  conv.execute();

  auto result = conv.result();
  auto result_vec = arma::vec(result.data(), result.size());
  std::println(std::cout, "  result range: [{}, {}]", result_vec.min(), result_vec.max());
  std::println(std::cout, "  expected: 3.0 everywhere");
  std::println(std::cout, "  match: {}", (std::abs(result_vec.min() - 3.0) < 1e-10 ? "true" : "false"));
}
