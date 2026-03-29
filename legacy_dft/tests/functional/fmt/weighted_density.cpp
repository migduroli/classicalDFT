#include "classicaldft_bits/functional/fmt/weighted_density.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft::functional::fmt;
using namespace dft::numerics::fourier;

static const std::vector<long> test_shape = {8, 8, 8};
static constexpr long test_N = 8 * 8 * 8;

// ── Construction ────────────────────────────────────────────────────────────

TEST(WeightedDensity, ConstructionAllocatesCorrectSizes) {
  WeightedDensity cf(test_shape);
  EXPECT_EQ(cf.total(), test_N);
  EXPECT_EQ(static_cast<long>(cf.field().n_elem), test_N);
  EXPECT_EQ(static_cast<long>(cf.derivative().n_elem), test_N);
}

TEST(WeightedDensity, ConstructionInitializesToZero) {
  WeightedDensity cf(test_shape);
  EXPECT_DOUBLE_EQ(arma::accu(arma::abs(cf.field())), 0.0);
  EXPECT_DOUBLE_EQ(arma::accu(arma::abs(cf.derivative())), 0.0);
}

TEST(WeightedDensity, MoveConstruction) {
  WeightedDensity cf(test_shape);
  auto n = cf.total();
  WeightedDensity moved(std::move(cf));
  EXPECT_EQ(moved.total(), n);
}

// ── set_weight_from_real ────────────────────────────────────────────────────

TEST(WeightedDensity, SetWeightFromRealSizeMismatchThrows) {
  WeightedDensity cf(test_shape);
  arma::vec bad(10, arma::fill::zeros);
  EXPECT_THROW(cf.set_weight_from_real(bad), std::invalid_argument);
}

// ── Convolution: uniform * uniform = uniform ────────────────────────────────

TEST(WeightedDensity, UniformConvolutionGivesUniform) {
  double c = 2.5;
  double rho0 = 0.8;

  WeightedDensity cf(test_shape);
  arma::vec w(test_N, arma::fill::value(c));
  cf.set_weight_from_real(w);

  FourierTransform rho_fft(test_shape);
  for (auto& v : rho_fft.real())
    v = rho0;
  rho_fft.forward();

  cf.convolve(rho_fft.fourier());

  double expected = c * rho0 * test_N;
  for (arma::uword i = 0; i < cf.field().n_elem; ++i) {
    EXPECT_NEAR(cf.field()(i), expected, 1e-8) << "at index " << i;
  }
}

// ── Delta-like weight recovers density ──────────────────────────────────────

TEST(WeightedDensity, DeltaWeightRecoversDensity) {
  WeightedDensity cf(test_shape);
  arma::vec w(test_N, arma::fill::zeros);
  w(0) = 1.0;
  cf.set_weight_from_real(w);

  arma::vec rho(test_N);
  for (arma::uword i = 0; i < rho.n_elem; ++i) {
    rho(i) = std::sin(2.0 * std::numbers::pi * static_cast<double>(i) / test_N);
  }

  FourierTransform rho_fft(test_shape);
  std::copy_n(rho.memptr(), rho.n_elem, rho_fft.real().data());
  rho_fft.forward();

  cf.convolve(rho_fft.fourier());

  for (arma::uword i = 0; i < rho.n_elem; ++i) {
    EXPECT_NEAR(cf.field()(i), rho(i), 1e-10) << "at index " << i;
  }
}

// ── Accumulate round-trip ───────────────────────────────────────────────────

TEST(WeightedDensity, AccumulateBasic) {
  WeightedDensity cf(test_shape);
  arma::vec w(test_N, arma::fill::zeros);
  w(0) = 1.0;
  cf.set_weight_from_real(w);

  for (arma::uword i = 0; i < cf.derivative().n_elem; ++i) {
    cf.derivative()(i) = static_cast<double>(i) * 0.01;
  }

  FourierTransform output(test_shape);
  output.zeros();
  cf.accumulate(output.fourier());

  double sum = 0.0;
  for (auto& c : output.fourier())
    sum += std::abs(c);
  EXPECT_GT(sum, 0.0);
}

TEST(WeightedDensity, AccumulateAccumulates) {
  WeightedDensity cf(test_shape);
  arma::vec w(test_N, arma::fill::value(1.0));
  cf.set_weight_from_real(w);
  cf.derivative().fill(1.0);

  FourierTransform out1(test_shape);
  out1.zeros();
  cf.accumulate(out1.fourier());

  FourierTransform out2(test_shape);
  out2.zeros();
  cf.accumulate(out2.fourier());
  cf.accumulate(out2.fourier());

  for (long i = 0; i < out1.fourier_total(); ++i) {
    EXPECT_NEAR(std::abs(out2.fourier()[i]), 2.0 * std::abs(out1.fourier()[i]), 1e-10);
  }
}
