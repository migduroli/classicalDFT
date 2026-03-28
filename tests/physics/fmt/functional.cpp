#include "classicaldft_bits/physics/fmt/functional.h"

#include "classicaldft_bits/physics/thermodynamics/enskog.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft_core::physics::fmt;
namespace thermo = dft_core::physics::thermodynamics;

// ── Helpers ─────────────────────────────────────────────────────────────────

static double numerical_derivative(std::function<double(double)> f, double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

// ── Dilute limits (eta -> 0) ────────────────────────────────────────────────

TEST(Functional, RosenfeldDiluteLimits) {
  Rosenfeld model;
  double eta = 1e-10;
  EXPECT_NEAR(model.f1(eta), std::log(1.0 - eta), 1e-14);
  EXPECT_NEAR(model.f2(eta), 1.0, 1e-9);
  EXPECT_NEAR(model.f3(eta), 1.0, 1e-9);
}

TEST(Functional, RSLTDiluteLimits) {
  RSLT model;
  double eta = 1e-10;
  EXPECT_NEAR(model.f1(eta), std::log(1.0 - eta), 1e-14);
  EXPECT_NEAR(model.f2(eta), 1.0, 1e-8);
  EXPECT_NEAR(model.f3(eta), 1.0, 1e-6);
}

TEST(Functional, WhiteBearIDiluteLimits) {
  WhiteBearI model;
  EXPECT_NEAR(model.f3(1e-10), 1.0, 1e-6);
}

TEST(Functional, WhiteBearIIDiluteLimits) {
  WhiteBearII model;
  EXPECT_NEAR(model.f3(1e-10), 1.0, 1e-6);
}

// ── f-function derivative consistency ───────────────────────────────────────

TEST(Functional, RosenfeldF1Derivative) {
  Rosenfeld model;
  EXPECT_NEAR(model.d_f1(0.3), numerical_derivative([&](double e) { return model.f1(e); }, 0.3), 1e-8);
}

TEST(Functional, RosenfeldF2Derivative) {
  Rosenfeld model;
  EXPECT_NEAR(model.d_f2(0.3), numerical_derivative([&](double e) { return model.f2(e); }, 0.3), 1e-8);
}

TEST(Functional, RosenfeldF3Derivative) {
  Rosenfeld model;
  EXPECT_NEAR(model.d_f3(0.3), numerical_derivative([&](double e) { return model.f3(e); }, 0.3), 1e-8);
}

TEST(Functional, RosenfeldF1SecondDerivative) {
  Rosenfeld model;
  EXPECT_NEAR(model.d2_f1(0.3), numerical_derivative([&](double e) { return model.d_f1(e); }, 0.3), 1e-6);
}

TEST(Functional, RSLTDerivativesConsistent) {
  RSLT model;
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    EXPECT_NEAR(model.d_f3(eta), numerical_derivative([&](double e) { return model.f3(e); }, eta), 1e-5)
        << "eta=" << eta;
  }
}

TEST(Functional, WhiteBearIDerivativesConsistent) {
  WhiteBearI model;
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    EXPECT_NEAR(model.d_f3(eta), numerical_derivative([&](double e) { return model.f3(e); }, eta), 1e-5)
        << "eta=" << eta;
  }
}

TEST(Functional, WhiteBearIIDerivativesConsistent) {
  WhiteBearII model;
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    EXPECT_NEAR(model.d_f2(eta), numerical_derivative([&](double e) { return model.f2(e); }, eta), 1e-5)
        << "eta=" << eta;
    EXPECT_NEAR(model.d_f3(eta), numerical_derivative([&](double e) { return model.f3(e); }, eta), 1e-5)
        << "eta=" << eta;
  }
}

// ── needs_tensor / name ─────────────────────────────────────────────────────

TEST(Functional, NeedsTensor) {
  EXPECT_FALSE(Rosenfeld().needs_tensor());
  EXPECT_FALSE(RSLT().needs_tensor());
  EXPECT_TRUE(WhiteBearI().needs_tensor());
  EXPECT_TRUE(WhiteBearII().needs_tensor());
}

TEST(Functional, Names) {
  EXPECT_EQ(Rosenfeld().name(), "Rosenfeld");
  EXPECT_EQ(RSLT().name(), "RSLT");
  EXPECT_EQ(WhiteBearI().name(), "WhiteBearI");
  EXPECT_EQ(WhiteBearII().name(), "WhiteBearII");
}

// ── Bulk free energy cross-check ────────────────────────────────────────────

TEST(Functional, RosenfeldMatchesPYCompressibility) {
  Rosenfeld model;
  thermo::PercusYevick py(thermo::PercusYevick::Route::Compressibility);

  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    double eta = thermo::packing_fraction(rho);
    double f_fmt = model.bulk_free_energy_density(rho, 1.0) / rho;
    double f_py = py.excess_free_energy(eta);
    EXPECT_NEAR(f_fmt, f_py, 1e-10) << "rho=" << rho;
  }
}

TEST(Functional, WhiteBearIMatchesCarnahanStarling) {
  WhiteBearI model;
  thermo::CarnahanStarling cs;

  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    double eta = thermo::packing_fraction(rho);
    double f_fmt = model.bulk_free_energy_density(rho, 1.0) / rho;
    double f_cs = cs.excess_free_energy(eta);
    EXPECT_NEAR(f_fmt, f_cs, 1e-10) << "rho=" << rho;
  }
}

// ── Bulk chemical potential cross-check ─────────────────────────────────────

TEST(Functional, RosenfeldChemicalPotentialMatchesPY) {
  Rosenfeld model;
  thermo::PercusYevick py(thermo::PercusYevick::Route::Compressibility);

  for (double rho : {0.1, 0.3, 0.5, 0.7}) {
    double eta = thermo::packing_fraction(rho);
    double mu_fmt = model.bulk_excess_chemical_potential(rho, 1.0);
    double mu_py = py.excess_free_energy(eta) + eta * py.d_excess_free_energy(eta);
    EXPECT_NEAR(mu_fmt, mu_py, 1e-8) << "rho=" << rho;
  }
}

TEST(Functional, WhiteBearIChemicalPotentialMatchesCS) {
  WhiteBearI model;
  thermo::CarnahanStarling cs;

  for (double rho : {0.1, 0.3, 0.5, 0.7}) {
    double eta = thermo::packing_fraction(rho);
    double mu_fmt = model.bulk_excess_chemical_potential(rho, 1.0);
    double mu_cs = cs.excess_free_energy(eta) + eta * cs.d_excess_free_energy(eta);
    EXPECT_NEAR(mu_fmt, mu_cs, 1e-8) << "rho=" << rho;
  }
}

// ── d_phi numerical gradient check ──────────────────────────────────────────

TEST(Functional, RosenfeldDPhiNumericalGradient) {
  Rosenfeld model;
  auto m = Measures::uniform(0.5, 1.0);
  auto dm = model.d_phi(m);
  double h = 1e-7;

  {
    auto mp = m;
    mp.eta += h;
    mp.compute_derived();
    auto mm = m;
    mm.eta -= h;
    mm.compute_derived();
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
    mp.compute_derived();
    auto mm = m;
    mm.n2 -= h;
    mm.compute_derived();
    EXPECT_NEAR(dm.n2, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-6);
  }
}

TEST(Functional, WhiteBearIDPhiNumericalGradient) {
  WhiteBearI model;
  auto m = Measures::uniform(0.5, 1.0);
  auto dm = model.d_phi(m);
  double h = 1e-7;

  {
    auto mp = m;
    mp.eta += h;
    mp.compute_derived();
    auto mm = m;
    mm.eta -= h;
    mm.compute_derived();
    EXPECT_NEAR(dm.eta, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
  {
    auto mp = m;
    mp.n2 += h;
    mp.compute_derived();
    auto mm = m;
    mm.n2 -= h;
    mm.compute_derived();
    EXPECT_NEAR(dm.n2, (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
  {
    auto mp = m;
    mp.T(0, 0) += h;
    mp.compute_derived();
    auto mm = m;
    mm.T(0, 0) -= h;
    mm.compute_derived();
    EXPECT_NEAR(dm.T(0, 0), (model.phi(mp) - model.phi(mm)) / (2.0 * h), 1e-5);
  }
}

// ── Phi3 known case ─────────────────────────────────────────────────────────

TEST(Functional, RosenfeldPhi3UniformFluid) {
  auto m = Measures::uniform(0.5, 1.0);
  Rosenfeld model;
  double expected = m.n2 * m.n2 * m.n2 / (24.0 * std::numbers::pi);
  EXPECT_NEAR(model.phi3(m), expected, 1e-14);
}

// ── Zero density gives zero ─────────────────────────────────────────────────

TEST(Functional, ZeroDensityGivesZeroFreeEnergy) {
  for (auto* model :
       {static_cast<Functional*>(new Rosenfeld()),
        static_cast<Functional*>(new RSLT()),
        static_cast<Functional*>(new WhiteBearI()),
        static_cast<Functional*>(new WhiteBearII())}) {
    EXPECT_NEAR(model->bulk_free_energy_density(0.0, 1.0), 0.0, 1e-14) << model->name();
    delete model;
  }
}
