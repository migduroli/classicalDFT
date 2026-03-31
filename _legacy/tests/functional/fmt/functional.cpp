#include "dft/functional/fmt/functional.h"

#include "dft/thermodynamics/enskog.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft::functional::fmt;
namespace thermo = dft::thermodynamics;

// ── Helpers ─────────────────────────────────────────────────────────────────

static double numerical_derivative(std::function<double(double)> f, double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

// ── Dilute limits (eta -> 0) ────────────────────────────────────────────────

TEST(Functional, RosenfeldDiluteLimits) {
  FMT model(Rosenfeld{});
  double eta = 1e-10;
  EXPECT_NEAR(model.ideal_factor(eta), std::log(1.0 - eta), 1e-14);
  EXPECT_NEAR(model.pair_factor(eta), 1.0, 1e-9);
  EXPECT_NEAR(model.triplet_factor(eta), 1.0, 1e-9);
}

TEST(Functional, RSLTDiluteLimits) {
  FMT model(RSLT{});
  double eta = 1e-10;
  EXPECT_NEAR(model.ideal_factor(eta), std::log(1.0 - eta), 1e-14);
  EXPECT_NEAR(model.pair_factor(eta), 1.0, 1e-8);
  EXPECT_NEAR(model.triplet_factor(eta), 1.0, 1e-6);
}

TEST(Functional, WhiteBearIDiluteLimits) {
  FMT model(WhiteBearI{});
  EXPECT_NEAR(model.triplet_factor(1e-10), 1.0, 1e-6);
}

TEST(Functional, WhiteBearIIDiluteLimits) {
  FMT model(WhiteBearII{});
  EXPECT_NEAR(model.triplet_factor(1e-10), 1.0, 1e-6);
}

// ── f-function derivative consistency ───────────────────────────────────────

TEST(Functional, RosenfeldF1Derivative) {
  FMT model(Rosenfeld{});
  EXPECT_NEAR(
      model.ideal_factor(0.3, 1), numerical_derivative([&](double e) { return model.ideal_factor(e); }, 0.3), 1e-8
  );
}

TEST(Functional, RosenfeldF2Derivative) {
  FMT model(Rosenfeld{});
  EXPECT_NEAR(
      model.pair_factor(0.3, 1), numerical_derivative([&](double e) { return model.pair_factor(e); }, 0.3), 1e-8
  );
}

TEST(Functional, RosenfeldF3Derivative) {
  FMT model(Rosenfeld{});
  EXPECT_NEAR(
      model.triplet_factor(0.3, 1), numerical_derivative([&](double e) { return model.triplet_factor(e); }, 0.3), 1e-8
  );
}

TEST(Functional, RosenfeldF1SecondDerivative) {
  FMT model(Rosenfeld{});
  EXPECT_NEAR(
      model.ideal_factor(0.3, 2), numerical_derivative([&](double e) { return model.ideal_factor(e, 1); }, 0.3), 1e-6
  );
}

TEST(Functional, RSLTDerivativesConsistent) {
  FMT model(RSLT{});
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    EXPECT_NEAR(
        model.triplet_factor(eta, 1), numerical_derivative([&](double e) { return model.triplet_factor(e); }, eta), 1e-5
    ) << "eta="
      << eta;
  }
}

TEST(Functional, WhiteBearIDerivativesConsistent) {
  FMT model(WhiteBearI{});
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    EXPECT_NEAR(
        model.triplet_factor(eta, 1), numerical_derivative([&](double e) { return model.triplet_factor(e); }, eta), 1e-5
    ) << "eta="
      << eta;
  }
}

TEST(Functional, WhiteBearIIDerivativesConsistent) {
  FMT model(WhiteBearII{});
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    EXPECT_NEAR(
        model.pair_factor(eta, 1), numerical_derivative([&](double e) { return model.pair_factor(e); }, eta), 1e-5
    ) << "eta="
      << eta;
    EXPECT_NEAR(
        model.triplet_factor(eta, 1), numerical_derivative([&](double e) { return model.triplet_factor(e); }, eta), 1e-5
    ) << "eta="
      << eta;
  }
}

// ── needs_tensor / name ─────────────────────────────────────────────────────

TEST(Functional, NeedsTensor) {
  EXPECT_FALSE(FMT(Rosenfeld{}).needs_tensor());
  EXPECT_FALSE(FMT(RSLT{}).needs_tensor());
  EXPECT_TRUE(FMT(WhiteBearI{}).needs_tensor());
  EXPECT_TRUE(FMT(WhiteBearII{}).needs_tensor());
}

TEST(Functional, Names) {
  EXPECT_EQ(FMT(Rosenfeld{}).name(), "Rosenfeld");
  EXPECT_EQ(FMT(RSLT{}).name(), "RSLT");
  EXPECT_EQ(FMT(WhiteBearI{}).name(), "WhiteBearI");
  EXPECT_EQ(FMT(WhiteBearII{}).name(), "WhiteBearII");
}

// ── Bulk free energy cross-check ────────────────────────────────────────────

TEST(Functional, RosenfeldMatchesPYCompressibility) {
  FMT model(Rosenfeld{});

  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    double eta = thermo::packing_fraction(rho);
    double f_fmt = model.bulk_free_energy_density(rho, 1.0) / rho;
    double f_py = thermo::PercusYevickCompressibility::excess_free_energy(eta);
    EXPECT_NEAR(f_fmt, f_py, 1e-10) << "rho=" << rho;
  }
}

TEST(Functional, WhiteBearIMatchesCarnahanStarling) {
  FMT model(WhiteBearI{});

  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    double eta = thermo::packing_fraction(rho);
    double f_fmt = model.bulk_free_energy_density(rho, 1.0) / rho;
    double f_cs = thermo::CarnahanStarling::excess_free_energy(eta);
    EXPECT_NEAR(f_fmt, f_cs, 1e-10) << "rho=" << rho;
  }
}

// ── Bulk chemical potential cross-check ─────────────────────────────────────

TEST(Functional, RosenfeldChemicalPotentialMatchesPY) {
  FMT model(Rosenfeld{});
  thermo::HardSphereModel py = thermo::PercusYevickCompressibility{};

  for (double rho : {0.1, 0.3, 0.5, 0.7}) {
    double eta = thermo::packing_fraction(rho);
    double mu_fmt = model.bulk_excess_chemical_potential(rho, 1.0);
    double f = thermo::hs_excess_free_energy(py, eta);
    double df = thermo::hs_excess_free_energy(py, eta, 1);
    double mu_py = f + eta * df;
    EXPECT_NEAR(mu_fmt, mu_py, 1e-8) << "rho=" << rho;
  }
}

TEST(Functional, WhiteBearIChemicalPotentialMatchesCS) {
  FMT model(WhiteBearI{});
  thermo::HardSphereModel cs = thermo::CarnahanStarling{};

  for (double rho : {0.1, 0.3, 0.5, 0.7}) {
    double eta = thermo::packing_fraction(rho);
    double mu_fmt = model.bulk_excess_chemical_potential(rho, 1.0);
    double f = thermo::hs_excess_free_energy(cs, eta);
    double df = thermo::hs_excess_free_energy(cs, eta, 1);
    double mu_cs = f + eta * df;
    EXPECT_NEAR(mu_fmt, mu_cs, 1e-8) << "rho=" << rho;
  }
}

// ── d_phi numerical gradient check ──────────────────────────────────────────

TEST(Functional, RosenfeldDPhiNumericalGradient) {
  FMT model(Rosenfeld{});
  auto m = Measures::uniform(0.5, 1.0);
  auto dm = model.d_phi(m);
  double h = 1e-7;

  {
    auto mp = m;
    mp.eta += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.eta -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.eta, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-6);
  }
  {
    auto mp = m;
    mp.n0 += h;
    auto mm = m;
    mm.n0 -= h;
    EXPECT_NEAR(dm.n0, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-6);
  }
  {
    auto mp = m;
    mp.n1 += h;
    auto mm = m;
    mm.n1 -= h;
    EXPECT_NEAR(dm.n1, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-6);
  }
  {
    auto mp = m;
    mp.n2 += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.n2 -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.n2, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-6);
  }
}

TEST(Functional, WhiteBearIDPhiNumericalGradient) {
  FMT model(WhiteBearI{});
  auto m = Measures::uniform(0.5, 1.0);
  auto dm = model.d_phi(m);
  double h = 1e-7;

  {
    auto mp = m;
    mp.eta += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.eta -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.eta, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
  {
    auto mp = m;
    mp.n2 += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.n2 -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.n2, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
  {
    auto mp = m;
    mp.T(0, 0) += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.T(0, 0) -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.T(0, 0), (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
}

// ── Phi3 known case ─────────────────────────────────────────────────────────

TEST(Functional, RosenfeldTripletPhiUniformFluid) {
  auto m = Measures::uniform(0.5, 1.0);
  FMT model(Rosenfeld{});
  double expected = m.n2 * m.n2 * m.n2 / (24.0 * std::numbers::pi);
  EXPECT_NEAR(model.triplet_phi(m), expected, 1e-14);
}

// ── RSLT d_phi and near-zero coverage ───────────────────────────────────────

TEST(Functional, RSLTDPhiNumericalGradient) {
  FMT model(RSLT{});
  auto m = Measures::uniform(0.5, 1.0);
  auto dm = model.d_phi(m);
  double h = 1e-7;

  {
    auto mp = m;
    mp.eta += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.eta -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.eta, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
  {
    auto mp = m;
    mp.n2 += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.n2 -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.n2, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
}

TEST(Functional, RSLTNearZeroDensity) {
  FMT model(RSLT{});
  auto m = Measures::uniform(1e-40, 1.0);
  EXPECT_NEAR(model.phi(m), 0.0, 1e-30);
  auto dm = model.d_phi(m);
  EXPECT_TRUE(std::isfinite(dm.eta));
}

// ── WhiteBearII d_phi coverage (tensor + d_T path) ──────────────────────────

TEST(Functional, WhiteBearIIDPhiNumericalGradient) {
  FMT model(WhiteBearII{});
  auto m = Measures::uniform(0.5, 1.0);
  auto dm = model.d_phi(m);
  double h = 1e-7;

  {
    auto mp = m;
    mp.eta += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.eta -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.eta, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
  {
    auto mp = m;
    mp.T(0, 0) += h;
    mp.compute_inner_products();
    auto mm = m;
    mm.T(0, 0) -= h;
    mm.compute_inner_products();
    EXPECT_NEAR(dm.T(0, 0), (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
}

// ── Zero density gives zero ─────────────────────────────────────────────────

TEST(Functional, ZeroDensityGivesZeroFreeEnergy) {
  for (auto model : {FMT(Rosenfeld{}), FMT(RSLT{}), FMT(WhiteBearI{}), FMT(WhiteBearII{})}) {
    EXPECT_NEAR(model.bulk_free_energy_density(0.0, 1.0), 0.0, 1e-14) << model.name();
  }
}
