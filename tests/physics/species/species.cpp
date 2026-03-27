#include "classicaldft_bits/physics/species/species.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace dft_core::physics::species;
using namespace dft_core::physics::density;

// ── Helper ──────────────────────────────────────────────────────────────────

static Density make_density(double dx = 1.0, arma::rowvec3 box = {3.0, 3.0, 3.0}) {
  return Density(dx, box);
}

// ── Construction ────────────────────────────────────────────────────────────

TEST(Species, ConstructionDefaultChemicalPotential) {
  Species s(make_density());
  EXPECT_DOUBLE_EQ(s.chemical_potential(), 0.0);
}

TEST(Species, ConstructionWithChemicalPotential) {
  Species s(make_density(), 1.5);
  EXPECT_DOUBLE_EQ(s.chemical_potential(), 1.5);
}

TEST(Species, DensityAccessible) {
  Species s(make_density());
  EXPECT_EQ(s.density().size(), 27U);
}

TEST(Species, ForceInitializedToZero) {
  Species s(make_density());
  EXPECT_DOUBLE_EQ(arma::accu(s.force()), 0.0);
  EXPECT_EQ(s.force().n_elem, s.density().size());
}

// ── Force management ────────────────────────────────────────────────────────

TEST(Species, ZeroForce) {
  Species s(make_density());
  arma::vec f(s.density().size(), arma::fill::ones);
  s.add_to_force(f);
  s.zero_force();
  EXPECT_DOUBLE_EQ(arma::accu(s.force()), 0.0);
}

TEST(Species, AddToForceVector) {
  Species s(make_density());
  arma::vec f(s.density().size(), arma::fill::value(2.0));
  s.add_to_force(f);
  EXPECT_DOUBLE_EQ(s.force()(0), 2.0);
  s.add_to_force(f);
  EXPECT_DOUBLE_EQ(s.force()(0), 4.0);
}

TEST(Species, AddToForceSingleElement) {
  Species s(make_density());
  s.add_to_force(5, 3.0);
  EXPECT_DOUBLE_EQ(s.force()(5), 3.0);
  EXPECT_DOUBLE_EQ(s.force()(0), 0.0);
}

// ── Chemical potential ──────────────────────────────────────────────────────

TEST(Species, SetChemicalPotential) {
  Species s(make_density());
  s.set_chemical_potential(2.5);
  EXPECT_DOUBLE_EQ(s.chemical_potential(), 2.5);
}

// ── Fixed-mass constraint ───────────────────────────────────────────────────

TEST(Species, FixedMassDefaultDisabled) {
  Species s(make_density());
  EXPECT_FALSE(s.has_fixed_mass());
  EXPECT_EQ(s.fixed_mass(), std::nullopt);
}

TEST(Species, SetFixedMassResetsChemPotential) {
  Species s(make_density(), 1.0);
  s.set_fixed_mass(10.0);
  EXPECT_TRUE(s.has_fixed_mass());
  EXPECT_DOUBLE_EQ(*s.fixed_mass(), 10.0);
  EXPECT_DOUBLE_EQ(s.chemical_potential(), 0.0);
}

TEST(Species, SetFixedMassThrowsOnNonPositive) {
  Species s(make_density());
  EXPECT_THROW(s.set_fixed_mass(0.0), std::invalid_argument);
  EXPECT_THROW(s.set_fixed_mass(-1.0), std::invalid_argument);
}

TEST(Species, ClearFixedMass) {
  Species s(make_density());
  s.set_fixed_mass(10.0);
  EXPECT_TRUE(s.has_fixed_mass());
  s.clear_fixed_mass();
  EXPECT_FALSE(s.has_fixed_mass());
}

TEST(Species, BeginForceCalculationRescalesDensity) {
  Species s(make_density());
  s.density().values().fill(1.0);
  double target = 10.0;
  s.set_fixed_mass(target);

  s.begin_force_calculation();
  EXPECT_NEAR(s.density().number_of_atoms(), target, 1e-12);
}

TEST(Species, EndForceCalculationProjectsForce) {
  Species s(make_density());
  double rho0 = 1.0;
  s.density().values().fill(rho0);
  double n_atoms = rho0 * s.density().size() * s.density().cell_volume();
  s.set_fixed_mass(n_atoms);
  // Simulate a force: dF = 1.0 everywhere
  arma::vec f(s.density().size(), arma::fill::ones);
  s.add_to_force(f);

  s.end_force_calculation();

  // mu = dot(dF, rho) / m_fixed
  double expected_mu = arma::dot(f, s.density().values()) / n_atoms;
  EXPECT_NEAR(s.chemical_potential(), expected_mu, 1e-12);

  // force should be projected: dF_i - mu * dV
  double dV = s.density().cell_volume();
  for (arma::uword i = 0; i < s.force().n_elem; ++i) {
    EXPECT_NEAR(s.force()(i), 1.0 - expected_mu * dV, 1e-12);
  }
}

// ── Convergence monitor ─────────────────────────────────────────────────────

TEST(Species, ConvergenceMonitor) {
  Species s(make_density());
  arma::vec f(s.density().size(), arma::fill::zeros);
  f(0) = 5.0;
  s.add_to_force(f);

  double dV = s.density().cell_volume();
  EXPECT_NEAR(s.convergence_monitor(), 5.0 / dV, 1e-12);
}

// ── Alias coordinates ───────────────────────────────────────────────────────

TEST(Species, AliasRoundTrip) {
  Species s(make_density());
  arma::vec x = arma::randu(s.density().size()) * 2.0;

  s.set_density_from_alias(x);
  arma::vec recovered = s.density_alias();

  for (arma::uword i = 0; i < x.n_elem; ++i) {
    EXPECT_NEAR(recovered(i), x(i), 1e-12);
  }
}

TEST(Species, AliasGuaranteesPositiveDensity) {
  Species s(make_density());
  arma::vec x(s.density().size(), arma::fill::zeros);
  s.set_density_from_alias(x);

  for (arma::uword i = 0; i < s.density().size(); ++i) {
    EXPECT_GE(s.density().values()(i), Species::rho_min);
  }
}

TEST(Species, AliasForceChainRule) {
  Species s(make_density());
  arma::vec x = arma::randu(s.density().size()) + 0.1;

  // Set a known force
  arma::vec dF(s.density().size(), arma::fill::value(3.0));
  s.add_to_force(dF);

  arma::vec alias_f = s.alias_force(x);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    EXPECT_NEAR(alias_f(i), 2.0 * x(i) * dF(i), 1e-12);
  }
}

TEST(Species, AliasForceNumericalConsistency) {
  Species s(make_density());
  arma::vec x(s.density().size(), arma::fill::value(1.5));

  // dF/drho = 1.0 everywhere
  arma::vec dF(s.density().size(), arma::fill::ones);
  s.add_to_force(dF);

  // Numerical: perturb x_0 by epsilon, measure density change
  double eps = 1e-7;
  arma::vec x_plus = x;
  x_plus(0) += eps;

  Species s1(make_density());
  s1.set_density_from_alias(x);
  Species s2(make_density());
  s2.set_density_from_alias(x_plus);

  double drho_dx = (s2.density().values()(0) - s1.density().values()(0)) / eps;
  // Analytic: drho/dx = 2x
  EXPECT_NEAR(drho_dx, 2.0 * x(0), 1e-5);
}

// ── External field energy ───────────────────────────────────────────────────

TEST(Species, ExternalFieldEnergyNoForce) {
  Species s(make_density());
  s.density().values().fill(1.0);
  s.density().external_field().fill(2.0);
  s.set_chemical_potential(0.5);

  double dV = s.density().cell_volume();
  double expected =
      arma::dot(s.density().values(), s.density().external_field()) * dV - 0.5 * s.density().number_of_atoms();

  double energy = s.external_field_energy(false);
  EXPECT_NEAR(energy, expected, 1e-12);
  // Force should still be zero (we didn't accumulate)
  EXPECT_DOUBLE_EQ(arma::accu(s.force()), 0.0);
}

TEST(Species, ExternalFieldEnergyWithForce) {
  Species s(make_density());
  s.density().values().fill(1.0);
  s.density().external_field().fill(2.0);
  s.set_chemical_potential(0.5);

  s.external_field_energy(true);

  double dV = s.density().cell_volume();
  // force(i) = (vext - mu) * dV = (2.0 - 0.5) * dV
  for (arma::uword i = 0; i < s.force().n_elem; ++i) {
    EXPECT_NEAR(s.force()(i), 1.5 * dV, 1e-12);
  }
}
