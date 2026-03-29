#include "dft/solver.h"

#include "dft/density.h"
#include "dft/functional/fmt/functional.h"
#include "dft/functional/fmt/species.h"
#include "dft/functional/interaction.h"
#include "dft/species.h"
#include "dft/thermodynamics/enskog.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft;

// ── Construction ──────────────────────────────────────────────────────────

TEST(Solver, DefaultConstructionHasNoSpecies) {
  Solver solver;
  EXPECT_EQ(solver.num_species(), 0);
}

TEST(Solver, AddSpeciesIncreasesCount) {
  Solver solver;
  auto d = density::Density(0.1, {1.0, 1.0, 1.0});
  solver.add_species(std::make_unique<species::Species>(std::move(d)));
  EXPECT_EQ(solver.num_species(), 1);
}

TEST(Solver, SpeciesAccessorReturnsCorrectObject) {
  Solver solver;
  auto d = density::Density(0.1, {1.0, 1.0, 1.0});
  d.values().fill(0.5);
  solver.add_species(std::make_unique<species::Species>(std::move(d)));
  EXPECT_NEAR(solver.density(0).values()(0), 0.5, 1e-15);
}

TEST(Solver, FmtNameReturnsNoneWhenNotSet) {
  Solver solver;
  EXPECT_EQ(solver.fmt_name(), "none");
}

TEST(Solver, ThrowsOnComputeWithNoSpecies) {
  Solver solver;
  EXPECT_THROW(solver.compute_free_energy_and_forces(), std::runtime_error);
}

// ── Ideal gas free energy ────────────────────────────────────────────────

TEST(Solver, IdealGasUniformDensityFreeEnergy) {
  double dx = 0.5;
  double l = 5.0;
  double rho0 = 0.3;

  auto d = density::Density(dx, {l, l, l});
  d.values().fill(rho0);

  auto s = std::make_unique<species::Species>(std::move(d));

  Solver solver;
  solver.add_species(std::move(s));

  double f = solver.compute_free_energy_and_forces();

  // For uniform rho: F_id = (rho * ln(rho) - rho) * V
  // F_ext = -mu * N = 0 since mu = 0
  double v = l * l * l;
  double expected_id = (rho0 * std::log(rho0) - rho0) * v;
  EXPECT_NEAR(solver.ideal_free_energy(), expected_id, std::abs(expected_id) * 1e-10);
  EXPECT_NEAR(solver.hard_sphere_free_energy(), 0.0, 1e-15);
  EXPECT_NEAR(solver.mean_field_free_energy(), 0.0, 1e-15);
}

TEST(Solver, IdealGasForceIsLogRhoDv) {
  double dx = 0.5;
  double l = 2.0;
  double rho0 = 0.4;

  auto d = density::Density(dx, {l, l, l});
  d.values().fill(rho0);

  auto s = std::make_unique<species::Species>(std::move(d), 0.0);
  Solver solver;
  solver.add_species(std::move(s));

  solver.compute_free_energy_and_forces();

  // Force at each point = (ln(rho) + V_ext - mu) * dV = ln(rho) * dV (mu=0, no ext)
  // But after end_force_calculation, if no fixed mass, force stays
  double d_v = dx * dx * dx;
  double expected = std::log(rho0) * d_v;
  const auto& force = solver.species(0).force();
  EXPECT_NEAR(force(0), expected, std::abs(expected) * 1e-10);
}

TEST(Solver, ExcessOnlySkipsIdealAndExternal) {
  double dx = 0.5;
  double l = 2.0;
  auto d = density::Density(dx, {l, l, l});
  d.values().fill(0.3);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double f = solver.compute_free_energy_and_forces(true);

  EXPECT_DOUBLE_EQ(solver.ideal_free_energy(), 0.0);
  EXPECT_DOUBLE_EQ(solver.external_free_energy(), 0.0);
  EXPECT_DOUBLE_EQ(f, 0.0);  // no FMT, no interactions
}

// ── Free energy decomposition ────────────────────────────────────────────

TEST(Solver, FreeEnergyDecompositionSumsToTotal) {
  double dx = 0.5;
  double l = 2.0;
  auto d = density::Density(dx, {l, l, l});
  d.values().fill(0.3);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double f_total = solver.compute_free_energy_and_forces();
  double f_sum = solver.ideal_free_energy() + solver.hard_sphere_free_energy() + solver.mean_field_free_energy() +
      solver.external_free_energy();
  EXPECT_NEAR(f_total, f_sum, std::abs(f_total) * 1e-14);
}

// ── Bulk thermodynamics ──────────────────────────────────────────────────

TEST(Solver, IdealGasChemicalPotentialIsLogRho) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  d.values().fill(0.3);
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double rho = 0.5;
  double mu = solver.chemical_potential(rho);
  EXPECT_NEAR(mu, std::log(rho), 1e-14);
}

TEST(Solver, IdealGasHelmholtzIsSelfConsistent) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  d.values().fill(0.3);
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double rho = 0.4;
  double f = solver.helmholtz_free_energy_density(rho);
  double expected = rho * std::log(rho) - rho;
  EXPECT_NEAR(f, expected, std::abs(expected) * 1e-14);
}

TEST(Solver, IdealGasGrandPotentialDensity) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  d.values().fill(0.3);
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double rho = 0.4;
  double omega = solver.grand_potential_density(rho);
  // omega = f - rho * mu = (rho*ln(rho) - rho) - rho*ln(rho) = -rho
  EXPECT_NEAR(omega, -rho, 1e-14);
}

TEST(Solver, ThermodynamicIdentityPressureFromOmega) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  d.values().fill(0.3);
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double rho = 0.3;
  // P = -omega = rho  (ideal gas: P = rho kT, and here kT=1)
  double p = -solver.grand_potential_density(rho);
  EXPECT_NEAR(p, rho, 1e-14);
}

// ── Convergence ──────────────────────────────────────────────────────────

TEST(Solver, ConvergenceMonitorIsNonNegative) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  d.values().fill(0.3);
  solver.add_species(std::make_unique<species::Species>(std::move(d)));
  solver.compute_free_energy_and_forces();
  EXPECT_GE(solver.convergence_monitor(), 0.0);
}

// ── Dimension ────────────────────────────────────────────────────────────

TEST(Solver, DimensionMatchesDensitySize) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  solver.add_species(std::make_unique<species::Species>(std::move(d)));
  EXPECT_EQ(solver.dimension(), solver.density(0).size());
}

TEST(Solver, DimensionIsZeroWithNoSpecies) {
  Solver solver;
  EXPECT_EQ(solver.dimension(), 0u);
}

// ── Hessian ──────────────────────────────────────────────────────────────

TEST(Solver, IdealGasHessianIsDiagonal) {
  double dx = 0.5;
  double l = 2.0;
  double rho0 = 0.4;

  auto d = density::Density(dx, {l, l, l});
  d.values().fill(rho0);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  arma::uword n = solver.dimension();
  arma::vec v(n, arma::fill::ones);
  arma::vec result(n);

  solver.hessian_dot_v(v, result);

  // H_ii = dV / rho_i, so H*1 = dV/rho at each point
  double d_v = dx * dx * dx;
  double expected = d_v / rho0;
  EXPECT_NEAR(result(0), expected, 1e-14);
  EXPECT_NEAR(result(n / 2), expected, 1e-14);
  EXPECT_NEAR(result(n - 1), expected, 1e-14);
}
