#include "dft/math/convolution.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <vector>

using namespace dft::math;

TEST_CASE("convolve with unit weight returns IFFT(rho_k)", "[convolution]") {
  FourierTransform scratch({4, 4, 4});
  auto n = scratch.fourier_total();

  // Set up rho_k as FFT of constant 1
  FourierTransform rho_ft({4, 4, 4});
  auto r = rho_ft.real();
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = 1.0;
  }
  rho_ft.forward();

  // Unit weight: w_k[i] = 1/N so that convolve gives back 1
  std::vector<std::complex<double>> weight_k(n, {1.0 / 64.0, 0.0});
  auto rho_k = rho_ft.fourier();

  auto result = convolve(weight_k, rho_k, scratch);
  CHECK(result.n_elem == 64);
  for (arma::uword i = 0; i < result.n_elem; ++i) {
    CHECK(result(i) == Catch::Approx(1.0).margin(1e-10));
  }
}

TEST_CASE("convolve with zero weight returns zero", "[convolution]") {
  FourierTransform scratch({4, 4, 4});
  auto n = scratch.fourier_total();

  std::vector<std::complex<double>> weight_k(n, {0.0, 0.0});
  std::vector<std::complex<double>> rho_k(n, {1.0, 0.0});

  auto result = convolve(weight_k, rho_k, scratch);
  for (arma::uword i = 0; i < result.n_elem; ++i) {
    CHECK(result(i) == Catch::Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("accumulate adds to force buffer", "[convolution]") {
  FourierTransform scratch({4, 4, 4});
  auto n = scratch.fourier_total();

  arma::vec derivative(64, arma::fill::ones);
  std::vector<std::complex<double>> weight_k(n, {1.0, 0.0});
  std::vector<std::complex<double>> force_k(n, {0.0, 0.0});

  accumulate(weight_k, derivative, scratch, force_k);

  // The DC component should be N (FFT of all-ones = N at DC)
  CHECK(force_k[0].real() == Catch::Approx(64.0).epsilon(1e-10));
}

TEST_CASE("accumulate with conjugate flag uses conj(weight)", "[convolution]") {
  FourierTransform scratch({4, 4, 4});
  auto n = scratch.fourier_total();

  arma::vec derivative(64, arma::fill::ones);
  // weight with imaginary part
  std::vector<std::complex<double>> weight_k(n, {0.0, 1.0});
  std::vector<std::complex<double>> force_k(n, {0.0, 0.0});

  accumulate(weight_k, derivative, scratch, force_k, true);

  // conj({0, 1}) = {0, -1}, multiplied by FFT of ones (64 at DC, 0 elsewhere)
  CHECK(force_k[0].real() == Catch::Approx(0.0).margin(1e-10));
  CHECK(force_k[0].imag() == Catch::Approx(-64.0).epsilon(1e-10));
}

TEST_CASE("accumulate accumulates into existing force", "[convolution]") {
  FourierTransform scratch({4, 4, 4});
  auto n = scratch.fourier_total();

  arma::vec derivative(64, arma::fill::ones);
  std::vector<std::complex<double>> weight_k(n, {1.0, 0.0});
  std::vector<std::complex<double>> force_k(n, {10.0, 0.0});

  accumulate(weight_k, derivative, scratch, force_k);

  // DC gets +64 on top of the existing 10
  CHECK(force_k[0].real() == Catch::Approx(74.0).epsilon(1e-10));
}
