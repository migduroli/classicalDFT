#include "dft/functional/fmt/species.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft::functional::fmt;
using namespace dft::density;

static constexpr double test_dx = 0.1;
static constexpr double test_diameter = 1.0;
static const arma::rowvec3 test_box = {1.6, 1.6, 1.6};

static Density make_density(double rho0 = 0.5) {
  Density d(test_dx, test_box);
  d.values().fill(rho0);
  return d;
}

// ── Construction ────────────────────────────────────────────────────────────

TEST(FMTSpecies, ConstructionBasic) {
  Species sp(make_density(), test_diameter);
  EXPECT_DOUBLE_EQ(sp.diameter(), test_diameter);
  EXPECT_DOUBLE_EQ(sp.chemical_potential(), 0.0);
}

TEST(FMTSpecies, ConstructionWithChemicalPotential) {
  Species sp(make_density(), test_diameter, 2.5);
  EXPECT_DOUBLE_EQ(sp.chemical_potential(), 2.5);
}

// ── Weighted densities for uniform fluid ────────────────────────────────────

TEST(FMTSpecies, UniformMeasuresMatchAnalytic) {
  double rho0 = 0.5;
  Species sp(make_density(rho0), test_diameter);

  sp.convolve_density(true);
  auto m = sp.measures_at(0);
  auto ref = Measures::uniform(rho0, test_diameter);

  EXPECT_NEAR(m.eta, ref.eta, 1e-10);
  EXPECT_NEAR(m.n2, ref.n2, 1e-10);
  EXPECT_NEAR(m.n1, ref.n1, 1e-10);
  EXPECT_NEAR(m.n0, ref.n0, 1e-10);

  EXPECT_NEAR(arma::norm(m.v0), 0.0, 1e-10);
  EXPECT_NEAR(arma::norm(m.v1), 0.0, 1e-10);

  double T_diag = ref.n2 / 3.0;
  EXPECT_NEAR(m.T(0, 0), T_diag, 1e-10);
  EXPECT_NEAR(m.T(1, 1), T_diag, 1e-10);
  EXPECT_NEAR(m.T(2, 2), T_diag, 1e-10);
}

// ── Free energy (without forces) ────────────────────────────────────────────

TEST(FMTSpecies, UniformFreeEnergyMatchesRosenfeld) {
  double rho0 = 0.5;
  Species sp(make_density(rho0), test_diameter);
  FMT model(Rosenfeld{});

  double F_ex = sp.compute_free_energy(model);

  double V = test_box(0) * test_box(1) * test_box(2);
  double F_expected = model.bulk_free_energy_density(rho0, test_diameter) * V;
  EXPECT_NEAR(F_ex, F_expected, std::abs(F_expected) * 1e-8);
}

TEST(FMTSpecies, UniformFreeEnergyMatchesWhiteBearI) {
  double rho0 = 0.5;
  Species sp(make_density(rho0), test_diameter);
  FMT model(WhiteBearI{});

  double F_ex = sp.compute_free_energy(model);

  double V = test_box(0) * test_box(1) * test_box(2);
  double F_expected = model.bulk_free_energy_density(rho0, test_diameter) * V;
  EXPECT_NEAR(F_ex, F_expected, std::abs(F_expected) * 1e-8);
}

// ── Forces are spatially constant for uniform density ───────────────────────

TEST(FMTSpecies, UniformForcesSpatiallyConstant) {
  double rho0 = 0.5;
  Species sp(make_density(rho0), test_diameter);
  FMT model(Rosenfeld{});

  sp.zero_force();
  sp.compute_forces(model);

  double mean_force = arma::mean(sp.force());
  double max_deviation = arma::max(arma::abs(sp.force() - mean_force));
  EXPECT_NEAR(max_deviation, 0.0, 1e-6);

  double dV = sp.density().cell_volume();
  double mu_ex = model.bulk_excess_chemical_potential(rho0, test_diameter);
  EXPECT_NEAR(mean_force, mu_ex * dV, std::abs(mu_ex * dV) * 1e-6);
}

// ── Bounded alias ───────────────────────────────────────────────────────────

TEST(FMTSpecies, AliasRoundTrip) {
  double rho0 = 0.5;
  Species sp(make_density(rho0), test_diameter);

  arma::vec x = sp.density_alias();
  sp.set_density_from_alias(x);

  for (arma::uword i = 0; i < sp.density().values().n_elem; ++i) {
    EXPECT_NEAR(sp.density().values()(i), rho0, 1e-12) << "index " << i;
  }
}

TEST(FMTSpecies, AliasBounded) {
  Species sp(make_density(), test_diameter);

  arma::vec x(sp.density().size(), arma::fill::value(1e6));
  sp.set_density_from_alias(x);

  double max_rho = arma::max(sp.density().values());
  double eta_max = max_rho * (std::numbers::pi / 6.0) * test_diameter * test_diameter * test_diameter;
  EXPECT_LT(eta_max, 1.0);
}

TEST(FMTSpecies, AliasChainRuleConsistency) {
  double rho0 = 0.3;
  Species sp(make_density(rho0), test_diameter);
  FMT model(Rosenfeld{});

  sp.zero_force();
  sp.compute_forces(model);

  arma::vec x = sp.density_alias();
  arma::vec af = sp.alias_force(x);

  double mean_af = arma::mean(af);
  double max_deviation = arma::max(arma::abs(af - mean_af));
  EXPECT_NEAR(max_deviation, 0.0, 1e-6);
}

TEST(FMTSpecies, AliasChainRuleNumerical) {
  double rho0 = 0.5;
  Species sp(make_density(rho0), test_diameter);
  FMT model(Rosenfeld{});

  arma::vec x = sp.density_alias();

  sp.set_density_from_alias(x);
  sp.zero_force();
  double F0 = sp.compute_forces(model);
  arma::vec af = sp.alias_force(x);

  double h = 1e-5;
  arma::vec x_pert = x;
  x_pert(0) += h;
  sp.set_density_from_alias(x_pert);
  sp.zero_force();
  double F1 = sp.compute_forces(model);

  double numerical = (F1 - F0) / h;
  EXPECT_NEAR(af(0), numerical, std::abs(numerical) * 0.01 + 1e-6);
}
