#include "classicaldft_bits/functional/fmt/data_structures.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft::functional::fmt;

// ── Default construction ────────────────────────────────────────────────────

TEST(FundamentalMeasures, DefaultConstructionAllZero) {
  FundamentalMeasures m;
  EXPECT_DOUBLE_EQ(m.eta, 0.0);
  EXPECT_DOUBLE_EQ(m.n0, 0.0);
  EXPECT_DOUBLE_EQ(m.n1, 0.0);
  EXPECT_DOUBLE_EQ(m.n2, 0.0);
  EXPECT_DOUBLE_EQ(arma::norm(m.v1), 0.0);
  EXPECT_DOUBLE_EQ(arma::norm(m.v2), 0.0);
  EXPECT_DOUBLE_EQ(arma::accu(arma::abs(m.T)), 0.0);
}

TEST(FundamentalMeasures, DefaultDerivedAllZero) {
  FundamentalMeasures m;
  m.compute_contractions();
  EXPECT_DOUBLE_EQ(m.contractions.dot_v1_v2, 0.0);
  EXPECT_DOUBLE_EQ(m.contractions.norm_v2_squared, 0.0);
  EXPECT_DOUBLE_EQ(m.contractions.quadratic_v2_T, 0.0);
  EXPECT_DOUBLE_EQ(m.contractions.trace_T_squared, 0.0);
  EXPECT_DOUBLE_EQ(m.contractions.trace_T_cubed, 0.0);
}

// ── Uniform fluid factory ───────────────────────────────────────────────────

TEST(FundamentalMeasures, UniformEta) {
  double rho = 0.8;
  double d = 1.0;
  auto m = FundamentalMeasures::uniform(rho, d);
  EXPECT_NEAR(m.eta, std::numbers::pi / 6.0 * rho, 1e-14);
}

TEST(FundamentalMeasures, UniformScalars) {
  double rho = 0.6;
  double d = 1.2;
  double R = d / 2.0;
  auto m = FundamentalMeasures::uniform(rho, d);

  EXPECT_NEAR(m.n2, std::numbers::pi * rho * d * d, 1e-14);
  EXPECT_NEAR(m.n1, R * rho, 1e-14);
  EXPECT_NEAR(m.n0, rho, 1e-14);
}

TEST(FundamentalMeasures, UniformRelations) {
  double rho = 0.5;
  double d = 1.4;
  auto m = FundamentalMeasures::uniform(rho, d);

  EXPECT_NEAR(m.n0, m.n2 / (std::numbers::pi * d * d), 1e-14);
  EXPECT_NEAR(m.n1, m.n2 / (2.0 * std::numbers::pi * d), 1e-14);
}

TEST(FundamentalMeasures, UniformVectorsVanish) {
  auto m = FundamentalMeasures::uniform(0.7, 1.0);
  EXPECT_DOUBLE_EQ(arma::norm(m.v1), 0.0);
  EXPECT_DOUBLE_EQ(arma::norm(m.v2), 0.0);
}

TEST(FundamentalMeasures, UniformTensorIsDiagonal) {
  double rho = 0.5;
  double d = 1.3;
  auto m = FundamentalMeasures::uniform(rho, d);

  double T_diag = m.n2 / 3.0;
  EXPECT_NEAR(m.T(0, 0), T_diag, 1e-14);
  EXPECT_NEAR(m.T(1, 1), T_diag, 1e-14);
  EXPECT_NEAR(m.T(2, 2), T_diag, 1e-14);

  EXPECT_DOUBLE_EQ(m.T(0, 1), 0.0);
  EXPECT_DOUBLE_EQ(m.T(0, 2), 0.0);
  EXPECT_DOUBLE_EQ(m.T(1, 2), 0.0);
}

TEST(FundamentalMeasures, UniformTensorIsSymmetric) {
  auto m = FundamentalMeasures::uniform(0.8, 1.5);
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j)
      EXPECT_DOUBLE_EQ(m.T(i, j), m.T(j, i));
}

// ── Derived quantities ──────────────────────────────────────────────────────

TEST(FundamentalMeasures, DotProducts) {
  FundamentalMeasures m;
  m.v1 = {1.0, 2.0, 3.0};
  m.v2 = {4.0, 5.0, 6.0};
  m.compute_contractions();

  EXPECT_DOUBLE_EQ(m.contractions.dot_v1_v2, 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);
  EXPECT_DOUBLE_EQ(m.contractions.norm_v2_squared, 4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0);
}

TEST(FundamentalMeasures, TraceT2ForDiagonalTensor) {
  FundamentalMeasures m;
  m.T = arma::diagmat(arma::vec({2.0, 3.0, 5.0}));
  m.compute_contractions();

  EXPECT_NEAR(m.contractions.trace_T_squared, 4.0 + 9.0 + 25.0, 1e-14);
}

TEST(FundamentalMeasures, TraceT3ForDiagonalTensor) {
  FundamentalMeasures m;
  m.T = arma::diagmat(arma::vec({2.0, 3.0, 5.0}));
  m.compute_contractions();

  EXPECT_NEAR(m.contractions.trace_T_cubed, 8.0 + 27.0 + 125.0, 1e-14);
}

TEST(FundamentalMeasures, TraceT2ForFullSymmetricTensor) {
  FundamentalMeasures m;
  m.T = {{1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, 6.0}};
  m.compute_contractions();

  arma::mat33 T2 = m.T * m.T;
  EXPECT_NEAR(m.contractions.trace_T_squared, arma::trace(T2), 1e-12);
}

TEST(FundamentalMeasures, TraceT3ForFullSymmetricTensor) {
  FundamentalMeasures m;
  m.T = {{1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, 6.0}};
  m.compute_contractions();

  arma::mat33 T3 = m.T * m.T * m.T;
  EXPECT_NEAR(m.contractions.trace_T_cubed, arma::trace(T3), 1e-10);
}

TEST(FundamentalMeasures, VTvKnownCase) {
  FundamentalMeasures m;
  m.v2 = {1.0, 0.0, 0.0};
  m.T = arma::eye<arma::mat>(3, 3) * 3.0;
  m.compute_contractions();

  EXPECT_NEAR(m.contractions.quadratic_v2_T, 3.0, 1e-14);
}

TEST(FundamentalMeasures, UniformDerivedConsistency) {
  double rho = 0.7;
  double d = 1.3;
  auto m = FundamentalMeasures::uniform(rho, d);

  EXPECT_NEAR(m.contractions.dot_v1_v2, 0.0, 1e-14);
  EXPECT_NEAR(m.contractions.norm_v2_squared, 0.0, 1e-14);
  EXPECT_NEAR(m.contractions.quadratic_v2_T, 0.0, 1e-14);

  double T_diag = m.n2 / 3.0;
  EXPECT_NEAR(m.contractions.trace_T_squared, 3.0 * T_diag * T_diag, 1e-14);
  EXPECT_NEAR(m.contractions.trace_T_cubed, 3.0 * T_diag * T_diag * T_diag, 1e-14);
}
