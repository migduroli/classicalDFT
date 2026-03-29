#include "cdft/functional/fmt.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

#include "cdft/numerics/autodiff.hpp"

namespace cdft::functional {

  TEST(MeasuresTest, UniformConstruction) {
    auto m = Measures<>::uniform(0.5, 1.0);
    double expected_eta = (std::numbers::pi / 6.0) * 0.5;
    EXPECT_NEAR(m.eta, expected_eta, 1e-14);
    EXPECT_GT(m.n2, 0.0);
    EXPECT_GT(m.n1, 0.0);
    EXPECT_GT(m.n0, 0.0);
  }

  TEST(MeasuresTest, ContractionsComputed) {
    auto m = Measures<>::uniform(0.5, 1.0);
    EXPECT_NEAR(m.contractions.dot_v1_v2, 0.0, 1e-14);
    EXPECT_NEAR(m.contractions.norm_v2_squared, 0.0, 1e-14);
  }

  TEST(RosenfeldTest, IdealFactor) {
    EXPECT_NEAR(Rosenfeld::ideal_factor(0.0), 0.0, 1e-14);
    double eta = 0.3;
    double f = Rosenfeld::ideal_factor(eta);
    EXPECT_NEAR(f, std::log(1.0 - eta), 1e-14);
  }

  TEST(RosenfeldTest, PairFactor) {
    double eta = 0.3;
    double f = Rosenfeld::pair_factor(eta);
    EXPECT_NEAR(f, 1.0 / (1.0 - eta), 1e-14);
  }

  TEST(FMTModelVariantTest, Dispatch) {
    FMTModel model = Rosenfeld{};
    EXPECT_EQ(fmt_name(model), "Rosenfeld");
    EXPECT_FALSE(fmt_needs_tensor(model));
  }

  TEST(FMTModelVariantTest, WhiteBearNeedsTensor) {
    FMTModel wbi = WhiteBearI{};
    EXPECT_TRUE(fmt_needs_tensor(wbi));
    EXPECT_EQ(fmt_name(wbi), "WhiteBearI");
  }

  TEST(FMTPhiTest, RosenfeldUniform) {
    FMTModel model = Rosenfeld{};
    auto m = Measures<>::uniform(0.5, 1.0);
    double phi = fmt_phi(model, m);
    EXPECT_NE(phi, 0.0);
  }

  TEST(FMTDPhiTest, RosenfeldUniform) {
    FMTModel model = Rosenfeld{};
    auto m = Measures<>::uniform(0.5, 1.0);
    auto dm = fmt_d_phi(model, m);
    EXPECT_NE(dm.eta, 0.0);
    EXPECT_NE(dm.n0, 0.0);
    EXPECT_NE(dm.n2, 0.0);
  }

  TEST(FMTBulkTest, AllModels) {
    double rho = 0.5;
    double d = 1.0;

    for (auto model : {FMTModel{Rosenfeld{}}, FMTModel{RSLT{}}, FMTModel{WhiteBearI{}}, FMTModel{WhiteBearII{}}}) {
      double f = fmt_bulk_free_energy_density(model, rho, d);
      double mu = fmt_bulk_excess_chemical_potential(model, rho, d);
      EXPECT_FALSE(std::isnan(f)) << "NaN for " << fmt_name(model);
      EXPECT_FALSE(std::isnan(mu)) << "NaN for " << fmt_name(model);
    }
  }

  TEST(FMTBulkTest, ZeroDensityZeroEnergy) {
    FMTModel model = Rosenfeld{};
    double f = fmt_bulk_free_energy_density(model, 0.0, 1.0);
    EXPECT_NEAR(f, 0.0, 1e-14);
  }

  // ── AutoDiff tests for FMT factors ────────────────────────────────────

  TEST(FMTAutoTest, RosenfeldFactorsViaAutoDiff) {
    using namespace cdft;
    double eta = 0.3;

    // Get df1/deta via autodiff
    auto [f1, df1] = derivatives_up_to_1(
        [](dual x) { return Rosenfeld::ideal_factor(x); }, eta);
    double e = 1.0 - eta;
    EXPECT_NEAR(df1, -1.0 / e, 1e-12);

    auto [f2, df2] = derivatives_up_to_1(
        [](dual x) { return Rosenfeld::pair_factor(x); }, eta);
    EXPECT_NEAR(df2, 1.0 / (e * e), 1e-12);

    auto [f3, df3] = derivatives_up_to_1(
        [](dual x) { return Rosenfeld::triplet_factor(x); }, eta);
    EXPECT_NEAR(df3, 2.0 / (e * e * e), 1e-12);
  }

  TEST(FMTAutoTest, WhiteBearIITripletDerivative) {
    using namespace cdft;
    double eta = 0.3;

    auto [f3, df3] = derivatives_up_to_1(
        [](dual x) { return WhiteBearII::triplet_factor(x); }, eta);

    // Verify against finite difference
    double h = 1e-7;
    double fd = (WhiteBearII::triplet_factor(eta + h)
                 - WhiteBearII::triplet_factor(eta - h)) / (2.0 * h);
    EXPECT_NEAR(df3, fd, 1e-5);
  }

}  // namespace cdft::functional
