#include "dft/solver.h"

#include "dft/density.h"
#include "dft/functional/fmt/functional.h"
#include "dft/functional/fmt/species.h"
#include "dft/functional/interaction.h"
#include "dft/potentials/potential.h"
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
  EXPECT_THROW((void)solver.compute_free_energy_and_forces(), std::runtime_error);
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

  (void)solver.compute_free_energy_and_forces();

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
  (void)solver.compute_free_energy_and_forces();
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

TEST(Solver, HessianDotVThrowsWithNoSpecies) {
  Solver solver;
  arma::vec v(1, arma::fill::ones);
  arma::vec result(1);
  EXPECT_THROW(solver.hessian_dot_v(v, result), std::runtime_error);
}

// ── Component management ─────────────────────────────────────────────────

TEST(Solver, SetFmtAllowsFmtName) {
  Solver solver;
  solver.set_fmt(std::make_unique<functional::fmt::FMT>(functional::fmt::Rosenfeld{}));
  EXPECT_EQ(solver.fmt_name(), "Rosenfeld");
}

TEST(Solver, AddInteractionStoresInteraction) {
  double dx = 0.1;
  arma::rowvec3 box = {1.6, 1.6, 1.6};
  auto d = density::Density(dx, box);
  d.values().fill(0.3);

  auto sp = std::make_unique<species::Species>(std::move(d));
  auto& sp_ref = *sp;

  potentials::LennardJones lj(1.0, 1.0, 0.7);
  auto inter = std::make_unique<functional::interaction::Interaction>(sp_ref, sp_ref, lj, 1.0);

  Solver solver;
  solver.add_species(std::move(sp));
  solver.add_interaction(std::move(inter));

  // The interaction is used during compute
  double f = solver.compute_free_energy_and_forces();
  EXPECT_NE(solver.mean_field_free_energy(), 0.0);
  EXPECT_NE(f, solver.ideal_free_energy() + solver.external_free_energy());
}

// ── FMT integration ─────────────────────────────────────────────────────

TEST(Solver, FmtHardSphereContributionIsNonZero) {
  double dx = 0.1;
  arma::rowvec3 box = {1.6, 1.6, 1.6};
  double rho0 = 0.5;

  auto d = density::Density(dx, box);
  d.values().fill(rho0);

  auto sp = std::make_unique<functional::fmt::Species>(std::move(d), 1.0);
  Solver solver;
  solver.add_species(std::move(sp));
  solver.set_fmt(std::make_unique<functional::fmt::FMT>(functional::fmt::WhiteBearI{}));

  double f = solver.compute_free_energy_and_forces();
  EXPECT_NE(solver.hard_sphere_free_energy(), 0.0);
  (void)f;
}

TEST(Solver, FmtChemicalPotentialExceedsIdeal) {
  double dx = 0.1;
  arma::rowvec3 box = {1.6, 1.6, 1.6};
  double rho0 = 0.5;

  auto d = density::Density(dx, box);
  d.values().fill(rho0);

  auto sp = std::make_unique<functional::fmt::Species>(std::move(d), 1.0);
  Solver solver;
  solver.add_species(std::move(sp));
  solver.set_fmt(std::make_unique<functional::fmt::FMT>(functional::fmt::WhiteBearI{}));

  double mu_ideal = std::log(rho0);
  double mu_total = solver.chemical_potential(rho0);
  // Hard-sphere excess chemical potential is positive for rho>0
  EXPECT_GT(mu_total, mu_ideal);
}

TEST(Solver, FmtHelmholtzExceedsIdeal) {
  double dx = 0.1;
  arma::rowvec3 box = {1.6, 1.6, 1.6};
  double rho0 = 0.5;

  auto d = density::Density(dx, box);
  d.values().fill(rho0);

  auto sp = std::make_unique<functional::fmt::Species>(std::move(d), 1.0);
  Solver solver;
  solver.add_species(std::move(sp));
  solver.set_fmt(std::make_unique<functional::fmt::FMT>(functional::fmt::WhiteBearI{}));

  double f_ideal = rho0 * std::log(rho0) - rho0;
  double f_total = solver.helmholtz_free_energy_density(rho0);
  EXPECT_GT(f_total, f_ideal);
}

// ── Interaction bulk thermodynamics ─────────────────────────────────────

TEST(Solver, InteractionChemicalPotentialShifts) {
  double dx = 0.1;
  arma::rowvec3 box = {1.6, 1.6, 1.6};
  double rho0 = 0.3;

  auto d = density::Density(dx, box);
  d.values().fill(rho0);

  auto sp = std::make_unique<species::Species>(std::move(d));
  auto& sp_ref = *sp;

  potentials::LennardJones lj(1.0, 1.0, 0.7);
  auto inter = std::make_unique<functional::interaction::Interaction>(sp_ref, sp_ref, lj, 1.0);

  Solver solver;
  solver.add_species(std::move(sp));

  double mu_no_inter = solver.chemical_potential(rho0);

  solver.add_interaction(std::move(inter));
  double mu_with_inter = solver.chemical_potential(rho0);

  // LJ attractive tail gives a_vdw < 0, so mu shifts down
  EXPECT_LT(mu_with_inter, mu_no_inter);
}

TEST(Solver, InteractionHelmholtzShifts) {
  double dx = 0.1;
  arma::rowvec3 box = {1.6, 1.6, 1.6};
  double rho0 = 0.3;

  auto d = density::Density(dx, box);
  d.values().fill(rho0);

  auto sp = std::make_unique<species::Species>(std::move(d));
  auto& sp_ref = *sp;

  potentials::LennardJones lj(1.0, 1.0, 0.7);
  auto inter = std::make_unique<functional::interaction::Interaction>(sp_ref, sp_ref, lj, 1.0);

  Solver solver;
  solver.add_species(std::move(sp));
  solver.add_interaction(std::move(inter));

  double f_ideal = rho0 * std::log(rho0) - rho0;
  double f_total = solver.helmholtz_free_energy_density(rho0);
  EXPECT_NE(f_total, f_ideal);
}

// ── Coexistence utilities ────────────────────────────────────────────────

static Solver make_vdw_solver() {
  double dx = 0.5;
  arma::rowvec3 box = {6.0, 6.0, 6.0};

  auto d = density::Density(dx, box);
  d.values().fill(0.1);

  auto sp = std::make_unique<functional::fmt::Species>(std::move(d), 1.0);
  auto& sp_ref = *sp;

  potentials::LennardJones lj(1.0, 1.0, 2.5);
  double kT = 0.7;
  auto inter = std::make_unique<functional::interaction::Interaction>(sp_ref, sp_ref, lj, kT);

  Solver solver;
  solver.add_species(std::move(sp));
  solver.set_fmt(std::make_unique<functional::fmt::FMT>(functional::fmt::WhiteBearI{}));
  solver.add_interaction(std::move(inter));
  return solver;
}

TEST(Solver, FindSpinodalReturnsTwoDensities) {
  auto solver = make_vdw_solver();

  double rho_s1 = 0.0;
  double rho_s2 = 0.0;
  solver.find_spinodal(1.1, 0.005, rho_s1, rho_s2, 1e-6);

  EXPECT_GT(rho_s1, 0.0);
  EXPECT_GT(rho_s2, rho_s1);
  EXPECT_LT(rho_s2, 1.1);
}

TEST(Solver, FindDensityFromChemicalPotential) {
  // Use ideal gas where mu(rho) = ln(rho) is strictly monotonic
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  d.values().fill(0.3);
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double rho_target = 0.4;
  double mu = solver.chemical_potential(rho_target);
  double rho_found = solver.find_density_from_chemical_potential(mu, 0.1, 0.9, 1e-8);

  EXPECT_NEAR(rho_found, rho_target, 1e-4);
}

TEST(Solver, FindDensityFromChemicalPotentialThrowsWhenNotBracketed) {
  auto solver = make_vdw_solver();

  // mu at rho=0.01 is very negative; searching in [0.1, 0.2] won't bracket it
  double mu_low = solver.chemical_potential(0.01);
  EXPECT_THROW((void)solver.find_density_from_chemical_potential(mu_low, 0.1, 0.2, 1e-8), std::runtime_error);
}

TEST(Solver, FindCoexistenceReturnsTwoDensities) {
  auto solver = make_vdw_solver();

  double rho_v = 0.0;
  double rho_l = 0.0;
  solver.find_coexistence(1.1, 0.005, rho_v, rho_l, 1e-6);

  // Vapor should be low density, liquid high
  EXPECT_GT(rho_v, 0.0);
  EXPECT_GT(rho_l, rho_v);
  EXPECT_LT(rho_l, 1.1);

  // At coexistence: pressures and chemical potentials should match
  double p_v = -solver.grand_potential_density(rho_v);
  double p_l = -solver.grand_potential_density(rho_l);
  EXPECT_NEAR(p_v, p_l, 0.01);

  double mu_v = solver.chemical_potential(rho_v);
  double mu_l = solver.chemical_potential(rho_l);
  EXPECT_NEAR(mu_v, mu_l, 0.01);
}

// ── Structural properties (stubs) ────────────────────────────────────────

TEST(Solver, RealSpaceDcfReturnsZeroForSingleSpecies) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  EXPECT_DOUBLE_EQ(solver.real_space_dcf(1.0, 0.5), 0.0);
}

TEST(Solver, FourierSpaceDcfReturnsZeroForSingleSpecies) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  EXPECT_DOUBLE_EQ(solver.fourier_space_dcf(1.0, 0.5), 0.0);
}

// ── Alias support ────────────────────────────────────────────────────────

TEST(Solver, ConvertForcesToAliasDerivatives) {
  double dx = 0.5;
  double l = 2.0;
  auto d = density::Density(dx, {l, l, l});
  d.values().fill(0.3);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  (void)solver.compute_free_energy_and_forces();

  // Get alias from species
  std::vector<arma::vec> aliases(1);
  aliases[0] = solver.species(0).density_alias();

  solver.convert_forces_to_alias_derivatives(aliases);

  // After conversion, aliases[0] holds the alias-space force
  EXPECT_EQ(aliases[0].n_elem, solver.density(0).size());
}

TEST(Solver, MutableSpeciesAccessor) {
  Solver solver;
  auto d = density::Density(0.5, {2.0, 2.0, 2.0});
  d.values().fill(0.3);
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  // Non-const accessor: mutation should work
  solver.species(0).set_chemical_potential(1.5);
  EXPECT_DOUBLE_EQ(solver.species(0).chemical_potential(), 1.5);
}
