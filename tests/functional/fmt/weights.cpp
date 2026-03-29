#include "dft/functional/fmt/weights.h"

#include "dft/functional/fmt/measures.h"
#include "dft/math/fourier.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft::functional::fmt;
using namespace dft::math::fourier;

static const std::vector<long> test_shape = {16, 16, 16};
static constexpr double test_dx = 0.1;
static constexpr double test_diameter = 1.0;
static constexpr long test_N = 16 * 16 * 16;

static WeightSet make_weight_set(const std::vector<long>& shape) {
  WeightSet ws;
  ws.for_each([&](ConvolutionField& ch) { ch = ConvolutionField(shape); });
  return ws;
}

// ── Sum rules for uniform density ───────────────────────────────────────────

class WeightsUniformTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ws_ = make_weight_set(test_shape);
    Weights::generate(test_diameter, test_dx, test_shape, ws_);

    rho_fft_ = FourierTransform(test_shape);
    for (auto& v : rho_fft_.real())
      v = rho0_;
    rho_fft_.forward();

    ws_.for_each([&](ConvolutionField& ch) { ch.convolve(rho_fft_.fourier()); });
  }

  WeightSet ws_;
  FourierTransform rho_fft_;
  double rho0_ = 0.8;
  double R_ = test_diameter / 2.0;
};

TEST_F(WeightsUniformTest, EtaGivesPackingFraction) {
  double expected = (std::numbers::pi / 6.0) * test_diameter * test_diameter * test_diameter * rho0_;
  for (arma::uword i = 0; i < ws_.eta.field().n_elem; ++i) {
    EXPECT_NEAR(ws_.eta.field()(i), expected, 1e-10) << "at index " << i;
  }
}

TEST_F(WeightsUniformTest, ScalarGivesN2) {
  double expected = std::numbers::pi * test_diameter * test_diameter * rho0_;
  for (arma::uword i = 0; i < ws_.scalar.field().n_elem; ++i) {
    EXPECT_NEAR(ws_.scalar.field()(i), expected, 1e-10) << "at index " << i;
  }
}

TEST_F(WeightsUniformTest, VectorFieldsVanish) {
  for (int a = 0; a < 3; ++a) {
    for (arma::uword i = 0; i < ws_.vector[a].field().n_elem; ++i) {
      EXPECT_NEAR(ws_.vector[a].field()(i), 0.0, 1e-10) << "axis " << a << " index " << i;
    }
  }
}

TEST_F(WeightsUniformTest, TensorDiagonalGivesN2Over3) {
  double expected = std::numbers::pi * test_diameter * test_diameter * rho0_ / 3.0;
  for (int a = 0; a < 3; ++a) {
    for (arma::uword i = 0; i < ws_.tensor(a, a).field().n_elem; ++i) {
      EXPECT_NEAR(ws_.tensor(a, a).field()(i), expected, 1e-10) << "axis " << a << " index " << i;
    }
  }
}

TEST_F(WeightsUniformTest, TensorOffDiagonalVanishes) {
  for (auto [i, j] : std::initializer_list<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}}) {
    for (arma::uword k = 0; k < ws_.tensor(i, j).field().n_elem; ++k) {
      EXPECT_NEAR(ws_.tensor(i, j).field()(k), 0.0, 1e-10) << "(" << i << "," << j << ") index " << k;
    }
  }
}

TEST_F(WeightsUniformTest, EtaMatchesMeasures) {
  auto m = Measures::uniform(rho0_, test_diameter);
  EXPECT_NEAR(ws_.eta.field()(0), m.eta, 1e-10);
}

TEST_F(WeightsUniformTest, ScalarMatchesMeasures) {
  auto m = Measures::uniform(rho0_, test_diameter);
  EXPECT_NEAR(ws_.scalar.field()(0), m.n2, 1e-10);
}

// ── DC component sanity checks ──────────────────────────────────────────────

TEST(Weights, DifferentDiameterGivesDifferentWeights) {
  auto ws1 = make_weight_set(test_shape);
  Weights::generate(1.0, test_dx, test_shape, ws1);

  auto ws2 = make_weight_set(test_shape);
  Weights::generate(1.5, test_dx, test_shape, ws2);

  EXPECT_NE(std::abs(ws1.eta.weight().fourier()[0]), std::abs(ws2.eta.weight().fourier()[0]));
}

TEST(Weights, ScalarDCComponent) {
  auto ws = make_weight_set(test_shape);
  Weights::generate(test_diameter, test_dx, test_shape, ws);

  double R = test_diameter / 2.0;
  double expected = 4.0 * std::numbers::pi * R * R / static_cast<double>(test_N);

  auto fk = ws.scalar.weight().fourier();
  EXPECT_NEAR(fk[0].real(), expected, 1e-14);
  EXPECT_NEAR(fk[0].imag(), 0.0, 1e-14);
}

TEST(Weights, VolumeDCComponent) {
  auto ws = make_weight_set(test_shape);
  Weights::generate(test_diameter, test_dx, test_shape, ws);

  double R = test_diameter / 2.0;
  double expected = (4.0 / 3.0) * std::numbers::pi * R * R * R / static_cast<double>(test_N);

  auto fk = ws.eta.weight().fourier();
  EXPECT_NEAR(fk[0].real(), expected, 1e-14);
  EXPECT_NEAR(fk[0].imag(), 0.0, 1e-14);
}

TEST(Weights, VectorDCComponentIsZero) {
  auto ws = make_weight_set(test_shape);
  Weights::generate(test_diameter, test_dx, test_shape, ws);

  for (int a = 0; a < 3; ++a) {
    EXPECT_NEAR(std::abs(ws.vector[a].weight().fourier()[0]), 0.0, 1e-14) << "axis " << a;
  }
}

TEST(Weights, TensorDCComponentIsotropic) {
  auto ws = make_weight_set(test_shape);
  Weights::generate(test_diameter, test_dx, test_shape, ws);

  double R = test_diameter / 2.0;
  double expected_diag = (4.0 * std::numbers::pi / 3.0) * R * R / static_cast<double>(test_N);

  for (int a = 0; a < 3; ++a) {
    auto fk = ws.tensor(a, a).weight().fourier();
    EXPECT_NEAR(fk[0].real(), expected_diag, 1e-14) << "axis " << a;
    EXPECT_NEAR(fk[0].imag(), 0.0, 1e-14) << "axis " << a;
  }
  for (auto [i, j] : std::initializer_list<std::pair<int, int>>{{0, 1}, {0, 2}, {1, 2}}) {
    EXPECT_NEAR(std::abs(ws.tensor(i, j).weight().fourier()[0]), 0.0, 1e-14);
  }
}

// ── Trace identity: Tr(T) = n2 for arbitrary density ────────────────────────

TEST(Weights, TensorTraceIdentity) {
  auto ws = make_weight_set(test_shape);
  Weights::generate(test_diameter, test_dx, test_shape, ws);

  FourierTransform rho_fft(test_shape);
  {
    auto real = rho_fft.real();
    long idx = 0;
    for (long ix = 0; ix < test_shape[0]; ++ix) {
      for (long iy = 0; iy < test_shape[1]; ++iy) {
        for (long iz = 0; iz < test_shape[2]; ++iz) {
          real[idx++] = 0.5 + 0.1 * std::sin(2.0 * std::numbers::pi * ix / test_shape[0]);
        }
      }
    }
    rho_fft.forward();
  }

  ws.for_each([&](ConvolutionField& ch) { ch.convolve(rho_fft.fourier()); });

  for (arma::uword i = 0; i < static_cast<arma::uword>(test_N); ++i) {
    double trace = ws.tensor(0, 0).field()(i) + ws.tensor(1, 1).field()(i) + ws.tensor(2, 2).field()(i);
    double n2 = ws.scalar.field()(i);
    EXPECT_NEAR(trace, n2, 1e-8) << "at index " << i;
  }
}
