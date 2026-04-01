#include "dft/functionals/fmt/weights.hpp"

#include "dft/functionals/fmt/measures.hpp"
#include "dft/math/convolution.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

using namespace dft::functionals::fmt;
using dft::Grid;

static constexpr double DX = 0.1;
static constexpr double DIAMETER = 1.0;
static constexpr double R = DIAMETER / 2.0;
static const Grid GRID = Grid{.dx = DX, .box_size = {1.6, 1.6, 1.6}, .shape = {16, 16, 16}};
static constexpr long N = 16 * 16 * 16;

// DC components (k = 0)

TEST_CASE("w3 DC component is sphere volume / N", "[fmt][weights]") {
  auto ws = generate_weights(DIAMETER, GRID);

  double expected = (4.0 / 3.0) * std::numbers::pi * R * R * R / static_cast<double>(N);
  auto fk = ws.w3.fourier();
  CHECK(fk[0].real() == Catch::Approx(expected).margin(1e-14));
  CHECK(fk[0].imag() == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("w2 DC component is sphere surface area / N", "[fmt][weights]") {
  auto ws = generate_weights(DIAMETER, GRID);

  double expected = 4.0 * std::numbers::pi * R * R / static_cast<double>(N);
  auto fk = ws.w2.fourier();
  CHECK(fk[0].real() == Catch::Approx(expected).margin(1e-14));
  CHECK(fk[0].imag() == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("wv2 DC component is zero (odd parity)", "[fmt][weights]") {
  auto ws = generate_weights(DIAMETER, GRID);

  for (int a = 0; a < 3; ++a) {
    CHECK(std::abs(ws.wv2[a].fourier()[0]) == Catch::Approx(0.0).margin(1e-14));
  }
}

TEST_CASE("wT DC component is isotropic", "[fmt][weights]") {
  auto ws = generate_weights(DIAMETER, GRID);

  double expected_diag = (4.0 * std::numbers::pi / 3.0) * R * R / static_cast<double>(N);

  for (int a = 0; a < 3; ++a) {
    auto fk = ws.tensor(a, a).fourier();
    CHECK(fk[0].real() == Catch::Approx(expected_diag).margin(1e-14));
    CHECK(fk[0].imag() == Catch::Approx(0.0).margin(1e-14));
  }

  for (auto [i, j] : std::initializer_list<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}}) {
    auto fk = ws.tensor(i, j).fourier();
    CHECK(std::abs(fk[0]) == Catch::Approx(0.0).margin(1e-14));
  }
}

// Different diameter gives different weights

TEST_CASE("different diameter gives different weights", "[fmt][weights]") {
  auto ws1 = generate_weights(1.0, GRID);
  auto ws2 = generate_weights(1.5, GRID);

  CHECK(std::abs(ws1.w3.fourier()[0]) != Catch::Approx(std::abs(ws2.w3.fourier()[0])).margin(1e-14));
}

// Sum rules: convolution with uniform density

TEST_CASE("w3 convolution with uniform density gives packing fraction", "[fmt][weights]") {
  double rho0 = 0.8;
  auto ws = generate_weights(DIAMETER, GRID);

  std::vector<long> s(GRID.shape.begin(), GRID.shape.end());
  dft::math::FourierTransform rho_ft(s);
  for (auto& v : rho_ft.real()) {
    v = rho0;
  }
  rho_ft.forward();

  arma::vec eta = dft::math::convolve(ws.w3.fourier(), rho_ft.fourier(), s);

  double expected = (std::numbers::pi / 6.0) * DIAMETER * DIAMETER * DIAMETER * rho0;
  for (arma::uword i = 0; i < eta.n_elem; ++i) {
    CHECK(eta(i) == Catch::Approx(expected).margin(1e-10));
  }
}

TEST_CASE("w2 convolution with uniform density gives n2", "[fmt][weights]") {
  double rho0 = 0.8;
  auto ws = generate_weights(DIAMETER, GRID);

  std::vector<long> s(GRID.shape.begin(), GRID.shape.end());
  dft::math::FourierTransform rho_ft(s);
  for (auto& v : rho_ft.real()) {
    v = rho0;
  }
  rho_ft.forward();

  arma::vec n2 = dft::math::convolve(ws.w2.fourier(), rho_ft.fourier(), s);

  double expected = std::numbers::pi * DIAMETER * DIAMETER * rho0;
  for (arma::uword i = 0; i < n2.n_elem; ++i) {
    CHECK(n2(i) == Catch::Approx(expected).margin(1e-10));
  }
}

TEST_CASE("wv2 convolution with uniform density vanishes", "[fmt][weights]") {
  double rho0 = 0.8;
  auto ws = generate_weights(DIAMETER, GRID);

  std::vector<long> s(GRID.shape.begin(), GRID.shape.end());
  dft::math::FourierTransform rho_ft(s);
  for (auto& v : rho_ft.real()) {
    v = rho0;
  }
  rho_ft.forward();

  for (int a = 0; a < 3; ++a) {
    arma::vec v = dft::math::convolve(ws.wv2[a].fourier(), rho_ft.fourier(), s);
    CHECK(arma::max(arma::abs(v)) == Catch::Approx(0.0).margin(1e-10));
  }
}

TEST_CASE("wT convolution with uniform density gives n2/3 on diagonal", "[fmt][weights]") {
  double rho0 = 0.8;
  auto ws = generate_weights(DIAMETER, GRID);

  std::vector<long> s(GRID.shape.begin(), GRID.shape.end());
  dft::math::FourierTransform rho_ft(s);
  for (auto& v : rho_ft.real()) {
    v = rho0;
  }
  rho_ft.forward();

  double expected_diag = std::numbers::pi * DIAMETER * DIAMETER * rho0 / 3.0;

  for (int a = 0; a < 3; ++a) {
    arma::vec t = dft::math::convolve(ws.tensor(a, a).fourier(), rho_ft.fourier(), s);
    CHECK(t(0) == Catch::Approx(expected_diag).margin(1e-10));
  }
}

TEST_CASE("wT convolution with uniform density vanishes off-diagonal", "[fmt][weights]") {
  double rho0 = 0.8;
  auto ws = generate_weights(DIAMETER, GRID);

  std::vector<long> s(GRID.shape.begin(), GRID.shape.end());
  dft::math::FourierTransform rho_ft(s);
  for (auto& v : rho_ft.real()) {
    v = rho0;
  }
  rho_ft.forward();

  for (auto [i, j] : std::initializer_list<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}}) {
    arma::vec t = dft::math::convolve(ws.tensor(i, j).fourier(), rho_ft.fourier(), s);
    CHECK(arma::max(arma::abs(t)) == Catch::Approx(0.0).margin(1e-10));
  }
}

// Uniform weighted densities match make_uniform_measures

TEST_CASE("weighted densities from convolution match make_uniform_measures", "[fmt][weights]") {
  double rho0 = 0.6;
  auto ws = generate_weights(DIAMETER, GRID);

  std::vector<long> s(GRID.shape.begin(), GRID.shape.end());
  dft::math::FourierTransform rho_ft(s);
  for (auto& v : rho_ft.real()) {
    v = rho0;
  }
  rho_ft.forward();

  arma::vec eta = dft::math::convolve(ws.w3.fourier(), rho_ft.fourier(), s);
  arma::vec n2 = dft::math::convolve(ws.w2.fourier(), rho_ft.fourier(), s);

  auto m = make_uniform_measures(rho0, DIAMETER);

  CHECK(eta(0) == Catch::Approx(m.eta).margin(1e-10));
  CHECK(n2(0) == Catch::Approx(m.n2).margin(1e-10));

  // n1 and n0 from Rosenfeld scaling
  double n1_from_n2 = n2(0) / (4.0 * std::numbers::pi * R);
  double n0_from_n2 = n2(0) / (4.0 * std::numbers::pi * R * R);
  CHECK(n1_from_n2 == Catch::Approx(m.n1).margin(1e-10));
  CHECK(n0_from_n2 == Catch::Approx(m.n0).margin(1e-10));
}
