#include "dft/math/fourier.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <numeric>

using namespace dft::math;

TEST_CASE("FourierTransform shape is stored correctly", "[fourier]") {
  FourierTransform ft({4, 4, 4});
  CHECK(ft.shape() == std::vector<long>{4, 4, 4});
  CHECK(ft.total() == 64);
  CHECK(ft.fourier_total() == 4 * 4 * 3);
}

TEST_CASE("FourierTransform zeros initializes to zero", "[fourier]") {
  FourierTransform ft({4, 4, 4});
  auto r = ft.real();
  for (std::size_t i = 0; i < r.size(); ++i) {
    CHECK(r[i] == 0.0);
  }
}

TEST_CASE("FourierTransform forward-backward roundtrip preserves data", "[fourier]") {
  FourierTransform ft({4, 4, 4});
  auto r = ft.real();
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = static_cast<double>(i);
  }
  ft.forward();
  ft.backward();
  ft.scale(1.0 / static_cast<double>(ft.total()));

  auto r2 = ft.real();
  for (std::size_t i = 0; i < r2.size(); ++i) {
    CHECK(r2[i] == Catch::Approx(static_cast<double>(i)).margin(1e-10));
  }
}

TEST_CASE("FourierTransform scale multiplies all real values", "[fourier]") {
  FourierTransform ft({2, 2, 2});
  auto r = ft.real();
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = 1.0;
  }
  ft.scale(3.0);
  for (std::size_t i = 0; i < r.size(); ++i) {
    CHECK(r[i] == Catch::Approx(3.0));
  }
}

TEST_CASE("FourierTransform DC component is sum", "[fourier]") {
  FourierTransform ft({4, 4, 4});
  auto r = ft.real();
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = 1.0;
  }
  ft.forward();
  auto f = ft.fourier();
  CHECK(f[0].real() == Catch::Approx(64.0).epsilon(1e-10));
  CHECK(f[0].imag() == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("FourierTransform throws for wrong shape size", "[fourier]") {
  REQUIRE_THROWS_AS(FourierTransform({4, 4}), std::invalid_argument);
  REQUIRE_THROWS_AS(FourierTransform({4, 4, 4, 4}), std::invalid_argument);
}

TEST_CASE("FourierTransform throws for zero dimension", "[fourier]") {
  REQUIRE_THROWS_AS(FourierTransform({4, 0, 4}), std::invalid_argument);
}

TEST_CASE("FourierTransform is move-constructible", "[fourier]") {
  FourierTransform ft1({4, 4, 4});
  auto r = ft1.real();
  r[0] = 42.0;
  FourierTransform ft2(std::move(ft1));
  CHECK(ft2.real()[0] == 42.0);
  CHECK(ft2.total() == 64);
}

TEST_CASE("FourierTransform is move-assignable", "[fourier]") {
  FourierTransform ft1({4, 4, 4});
  ft1.real()[0] = 99.0;
  FourierTransform ft2({2, 2, 2});
  ft2 = std::move(ft1);
  CHECK(ft2.real()[0] == 99.0);
  CHECK(ft2.total() == 64);
}

TEST_CASE("FourierConvolution computes cyclic convolution", "[fourier]") {
  FourierConvolution conv({4, 4, 4});
  auto a = conv.input_a();
  auto b = conv.input_b();

  // delta(0) conv constant(1) = sum of constant * delta / N = 1
  // FourierConvolution divides by N, so: IFFT[FFT(delta)*FFT(ones)]/N
  // FFT(delta) = 1 everywhere, FFT(ones) = N at DC and 0 elsewhere
  // product = N at DC and 0 elsewhere, IFFT = N everywhere (since c2r of DC=N gives N for all points)
  // after /N => 1 everywhere
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = 0.0;
    b[i] = 1.0;
  }
  a[0] = 1.0;
  conv.execute();

  auto r = conv.result();
  for (std::size_t i = 0; i < r.size(); ++i) {
    CHECK(r[i] == Catch::Approx(1.0).margin(1e-10));
  }
}

TEST_CASE("FourierConvolution shape and total are correct", "[fourier]") {
  FourierConvolution conv({8, 4, 2});
  CHECK(conv.shape() == std::vector<long>{8, 4, 2});
  CHECK(conv.total() == 64);
}

TEST_CASE("FourierConvolution delta * delta gives delta", "[fourier]") {
  FourierConvolution conv({4, 4, 4});
  auto a = conv.input_a();
  auto b = conv.input_b();
  for (std::size_t i = 0; i < a.size(); ++i) {
    a[i] = 0.0;
    b[i] = 0.0;
  }
  a[0] = 1.0;
  b[0] = 1.0;
  conv.execute();

  auto r = conv.result();
  CHECK(r[0] == Catch::Approx(1.0).margin(1e-12));
  for (std::size_t i = 1; i < r.size(); ++i) {
    CHECK(r[i] == Catch::Approx(0.0).margin(1e-12));
  }
}
