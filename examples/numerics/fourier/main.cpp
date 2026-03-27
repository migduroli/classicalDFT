#include <classicaldft>

#include <armadillo>
#include <cmath>
#include <iomanip>

int main() {
  using namespace dft_core::numerics;
  using namespace dft_core::io;

  // ── FourierTransform: RAII wrapper for FFTW3 ─────────────────────────

  console::info("FourierTransform — round-trip test");

  auto shape = std::vector<long>{8, 8, 8};
  auto plan = fourier::FourierTransform(shape);

  std::cout << "  grid:           " << shape[0] << " x " << shape[1] << " x " << shape[2] << std::endl;
  std::cout << "  total points:   " << plan.total() << std::endl;
  std::cout << "  fourier points: " << plan.fourier_total() << std::endl;

  // Fill with a single sine mode: sin(2*pi*i / N) on the x-axis
  auto grid_x = arma::linspace(0.0, 2.0 * M_PI * (1.0 - 1.0 / shape[0]), shape[0]);
  arma::vec sin_x = arma::sin(grid_x);

  auto real = plan.real();
  for (long i = 0; i < shape[0]; ++i) {
    for (long j = 0; j < shape[1]; ++j) {
      for (long k = 0; k < shape[2]; ++k) {
        real[k + shape[2] * (j + shape[1] * i)] = sin_x(i);
      }
    }
  }

  // Save original data
  auto original = arma::vec(real.data(), real.size());

  // Forward FFT
  plan.forward();

  // Show first few Fourier coefficients
  auto fk = plan.fourier();
  std::cout << "  first 4 Fourier coefficients:" << std::endl;
  for (int n = 0; n < 4; ++n) {
    std::cout << "    k=" << n << " -> (" << std::setprecision(4)
              << fk[n].real() << ", " << fk[n].imag() << ")" << std::endl;
  }

  // Backward FFT + normalisation
  plan.backward();
  plan.scale(1.0 / static_cast<double>(plan.total()));

  // Verify round-trip: compare with saved original
  auto roundtrip = arma::vec(real.data(), real.size());
  auto max_error = arma::max(arma::abs(roundtrip - original));
  std::cout << "  round-trip max error: " << max_error << std::endl;

  std::cout << std::endl;

  // ── Parseval's theorem: energy is conserved ──────────────────────────────

  console::info("Parseval's theorem");

  auto real_energy = arma::dot(roundtrip, roundtrip);

  // Forward again (Parseval compares post-FFT)
  plan.forward();
  auto fourier_energy = 0.0;
  for (auto c : plan.fourier()) {
    fourier_energy += std::norm(c);
  }
  fourier_energy /= static_cast<double>(plan.total());

  std::cout << std::fixed;
  std::cout << "  real-space energy:    " << real_energy << std::endl;
  std::cout << "  fourier-space energy: " << fourier_energy << std::endl;
  std::cout << "  ratio:                " << fourier_energy / real_energy << std::endl;

  std::cout << std::endl;

  // ── FourierConvolution: cyclic convolution via FFT ───────────────────────

  console::info("FourierConvolution — convolving a delta with a constant");

  auto conv = fourier::FourierConvolution(shape);

  // input_a = delta function (1 at origin, 0 elsewhere)
  auto a = conv.input_a();
  std::fill(a.begin(), a.end(), 0.0);
  a[0] = 1.0;

  // input_b = constant field (value = 3.0)
  auto b = conv.input_b();
  std::fill(b.begin(), b.end(), 3.0);

  conv.execute();

  // delta * constant = constant (scaled by 1/N from normalisation)
  auto result = conv.result();
  auto result_vec = arma::vec(result.data(), result.size());
  std::cout << "  result range: [" << result_vec.min() << ", " << result_vec.max() << "]" << std::endl;
  std::cout << "  expected:     constant = 3.0 everywhere" << std::endl;
  std::cout << "  match:        " << (std::abs(result_vec.min() - 3.0) < 1e-10 ? "true" : "false") << std::endl;

  std::cout << std::endl;
  console::info("Done.");
}
