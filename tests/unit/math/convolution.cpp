#include "dft/math/convolution.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <vector>

using namespace dft::math;

static const std::vector<long> SHAPE = { 4, 4, 4 };
static constexpr long N = 64;

TEST_CASE("convolve with unit weight returns IFFT(rho_k)", "[convolution]") {
  FourierTransform rho_ft(SHAPE);
  auto n = rho_ft.fourier_total();

  auto r = rho_ft.real();
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = 1.0;
  }
  rho_ft.forward();

  std::vector<std::complex<double>> weight_k(n, { 1.0 / static_cast<double>(N), 0.0 });
  auto rho_k = rho_ft.fourier();

  auto result = convolve(weight_k, rho_k, SHAPE);
  CHECK(result.n_elem == static_cast<arma::uword>(N));
  for (arma::uword i = 0; i < result.n_elem; ++i) {
    CHECK(result(i) == Catch::Approx(1.0).margin(1e-10));
  }
}

TEST_CASE("convolve with zero weight returns zero", "[convolution]") {
  FourierTransform tmp(SHAPE);
  auto n = tmp.fourier_total();

  std::vector<std::complex<double>> weight_k(n, { 0.0, 0.0 });
  std::vector<std::complex<double>> rho_k(n, { 1.0, 0.0 });

  auto result = convolve(weight_k, rho_k, SHAPE);
  for (arma::uword i = 0; i < result.n_elem; ++i) {
    CHECK(result(i) == Catch::Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("back_convolve computes weight_k * FFT(derivative)", "[convolution]") {
  FourierTransform tmp(SHAPE);
  auto n = tmp.fourier_total();

  arma::vec derivative(N, arma::fill::ones);
  std::vector<std::complex<double>> weight_k(n, { 1.0, 0.0 });

  auto result = back_convolve(weight_k, derivative, SHAPE);

  // The DC component should be N (FFT of all-ones = N at DC)
  CHECK(result[0].real() == Catch::Approx(static_cast<double>(N)).epsilon(1e-10));
}

TEST_CASE("back_convolve with conjugate flag uses conj(weight)", "[convolution]") {
  FourierTransform tmp(SHAPE);
  auto n = tmp.fourier_total();

  arma::vec derivative(N, arma::fill::ones);
  // weight with imaginary part
  std::vector<std::complex<double>> weight_k(n, { 0.0, 1.0 });

  auto result = back_convolve(weight_k, derivative, SHAPE, true);

  // conj({0, 1}) = {0, -1}, multiplied by FFT of ones (N at DC, 0 elsewhere)
  CHECK(result[0].real() == Catch::Approx(0.0).margin(1e-10));
  CHECK(result[0].imag() == Catch::Approx(-static_cast<double>(N)).epsilon(1e-10));
}

TEST_CASE("back_convolve results can be summed for accumulation", "[convolution]") {
  FourierTransform tmp(SHAPE);
  auto n = tmp.fourier_total();

  arma::vec derivative(N, arma::fill::ones);
  std::vector<std::complex<double>> weight_k(n, { 1.0, 0.0 });

  auto r1 = back_convolve(weight_k, derivative, SHAPE);
  auto r2 = back_convolve(weight_k, derivative, SHAPE);

  // Sum two contributions via Armadillo +=
  r1 += r2;

  // DC gets 2 * N
  CHECK(r1(0).real() == Catch::Approx(2.0 * static_cast<double>(N)).epsilon(1e-10));
}
