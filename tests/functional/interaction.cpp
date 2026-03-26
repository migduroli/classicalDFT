#include "dft/functional/interaction.h"

#include "dft/density.h"
#include "dft/potentials/potential.h"
#include "dft/species.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft::functional::interaction;
using namespace dft::species;
using namespace dft::density;
using namespace dft::potentials;

// ── Helpers ─────────────────────────────────────────────────────────────────

static Density make_density(double dx = 0.5, arma::rowvec3 box = {4.0, 4.0, 4.0}) {
  return Density(dx, box);
}

static Species make_species(double dx = 0.5, arma::rowvec3 box = {4.0, 4.0, 4.0}, double mu = 0.0) {
  return Species(make_density(dx, box), mu);
}

static LennardJones make_lj(double sigma = 1.0, double epsilon = 1.0, double r_cutoff = 2.5) {
  return LennardJones(sigma, epsilon, r_cutoff);
}

// ── Construction ────────────────────────────────────────────────────────────

TEST(Interaction, ConstructionSelfInteraction) {
  auto s = make_species();
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);
  EXPECT_DOUBLE_EQ(inter.temperature(), 1.0);
  EXPECT_EQ(inter.scheme(), WeightScheme::InterpolationLinearF);
}

TEST(Interaction, ConstructionWithScheme) {
  auto s = make_species();
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0, WeightScheme::GaussE, 3);
  EXPECT_EQ(inter.scheme(), WeightScheme::GaussE);
}

TEST(Interaction, ConstructionNegativeKtThrows) {
  auto s = make_species();
  auto lj = make_lj();
  EXPECT_THROW(Interaction(s, s, lj, -1.0), std::invalid_argument);
}

TEST(Interaction, ConstructionZeroKtThrows) {
  auto s = make_species();
  auto lj = make_lj();
  EXPECT_THROW(Interaction(s, s, lj, 0.0), std::invalid_argument);
}

TEST(Interaction, ConstructionIncompatibleGridsThrows) {
  auto s1 = make_species(0.5);
  auto s2 = make_species(0.25);
  auto lj = make_lj();
  EXPECT_THROW(Interaction(s1, s2, lj, 1.0), std::invalid_argument);
}

TEST(Interaction, SpeciesAccessors) {
  auto s1 = make_species();
  auto s2 = make_species();
  auto lj = make_lj();
  Interaction inter(s1, s2, lj, 1.5);
  EXPECT_EQ(&inter.species_1(), &s1);
  EXPECT_EQ(&inter.species_2(), &s2);
  EXPECT_DOUBLE_EQ(inter.temperature(), 1.5);
}

// ── VdW parameter ───────────────────────────────────────────────────────────

TEST(Interaction, VdwParameterIsNegativeForLJ) {
  // The attractive tail of LJ is negative, so a_vdw should be negative
  auto s = make_species();
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);
  EXPECT_LT(inter.vdw_parameter(), 0.0);
}

TEST(Interaction, VdwParameterMatchesContinuousIntegral) {
  // Compare discrete a_vdw with the analytic 4*pi*integral.
  // Use BH perturbation to avoid the constant continuation near the origin.
  auto s = make_species(0.1, {6.0, 6.0, 6.0});
  auto lj = make_lj(1.0, 1.0, 2.5);
  lj.set_bh_perturbation();
  double kT = 1.0;
  Interaction inter(s, s, lj, kT, WeightScheme::GaussE, 5);

  // compute_van_der_waals_integral returns (2*pi/kT) * integral r^2 w_att(r) dr.
  // The full 3D integral is (4*pi/kT) * integral r^2 w_att(r) dr = 2 * that.
  auto lj_copy = make_lj(1.0, 1.0, 2.5);
  lj_copy.set_bh_perturbation();
  double a_continuous = 2.0 * lj_copy.compute_van_der_waals_integral(kT);

  EXPECT_NEAR(inter.vdw_parameter(), a_continuous, std::abs(a_continuous) * 0.05);
}

TEST(Interaction, TemperatureScalingHalvesVdw) {
  auto s1 = make_species();
  auto s2 = make_species();
  auto lj = make_lj();

  Interaction inter1(s1, s1, lj, 1.0);
  Interaction inter2(s2, s2, lj, 2.0);

  // Doubling kT should halve a_vdw
  EXPECT_NEAR(inter2.vdw_parameter(), inter1.vdw_parameter() / 2.0, std::abs(inter1.vdw_parameter()) * 1e-10);
}

// ── Zero density ────────────────────────────────────────────────────────────

TEST(Interaction, ZeroDensityZeroEnergy) {
  auto s = make_species();
  // Density is zero by default
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);
  double energy = inter.compute_free_energy();
  EXPECT_NEAR(energy, 0.0, 1e-15);
}

TEST(Interaction, ZeroDensityZeroForce) {
  auto s = make_species();
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);
  s.zero_force();
  inter.compute_forces();
  EXPECT_NEAR(arma::accu(arma::abs(s.force())), 0.0, 1e-15);
}

// ── Uniform density ─────────────────────────────────────────────────────────

TEST(Interaction, UniformDensityEnergy) {
  double rho0 = 0.5;
  auto s = make_species();
  s.density().values().fill(rho0);
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);

  double energy = inter.compute_free_energy();
  double dv = s.density().cell_volume();
  double volume = static_cast<double>(s.density().size()) * dv;

  // F = (1/2) * a_vdw * rho^2 * V
  double expected = 0.5 * inter.vdw_parameter() * rho0 * rho0 * volume;
  EXPECT_NEAR(energy, expected, std::abs(expected) * 1e-10);
}

TEST(Interaction, UniformDensityForce) {
  double rho0 = 0.5;
  auto s = make_species();
  s.density().values().fill(rho0);
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);

  s.zero_force();
  inter.compute_forces();

  double dv = s.density().cell_volume();
  // For self-interaction: dF/drho_i = (w * rho)(i) * dV = a_vdw * rho * dV
  // (uniform density: convolution of uniform with w gives sum(w_real)*rho = a_vdw*rho)
  double expected_force_per_element = inter.vdw_parameter() * rho0 * dv;
  // All force elements should be equal for uniform density
  double f0 = s.force()(0);
  EXPECT_NEAR(f0, expected_force_per_element, std::abs(expected_force_per_element) * 1e-10);

  // Check all elements are the same
  for (arma::uword i = 1; i < s.force().n_elem; ++i) {
    EXPECT_NEAR(s.force()(i), f0, std::abs(f0) * 1e-10);
  }
}

// ── Two-species interaction ─────────────────────────────────────────────────

TEST(Interaction, TwoSpeciesUniformEnergy) {
  double rho1 = 0.3;
  double rho2 = 0.7;
  auto s1 = make_species();
  auto s2 = make_species();
  s1.density().values().fill(rho1);
  s2.density().values().fill(rho2);
  auto lj = make_lj();
  Interaction inter(s1, s2, lj, 1.0);

  double energy = inter.compute_free_energy();
  double dv = s1.density().cell_volume();
  double volume = static_cast<double>(s1.density().size()) * dv;

  double expected = 0.5 * inter.vdw_parameter() * rho1 * rho2 * volume;
  EXPECT_NEAR(energy, expected, std::abs(expected) * 1e-10);
}

TEST(Interaction, TwoSpeciesForcesBothAccumulated) {
  double rho1 = 0.3;
  double rho2 = 0.7;
  auto s1 = make_species();
  auto s2 = make_species();
  s1.density().values().fill(rho1);
  s2.density().values().fill(rho2);
  auto lj = make_lj();
  Interaction inter(s1, s2, lj, 1.0);

  s1.zero_force();
  s2.zero_force();
  inter.compute_forces();

  double dv = s1.density().cell_volume();
  // Cross-interaction: dF/drho1_i = (1/2) * (w*rho2)(i) * dV
  double expected_f1 = 0.5 * inter.vdw_parameter() * rho2 * dv;
  // dF/drho2_j = (1/2) * (w*rho1)(j) * dV
  double expected_f2 = 0.5 * inter.vdw_parameter() * rho1 * dv;

  EXPECT_NEAR(s1.force()(0), expected_f1, std::abs(expected_f1) * 1e-10);
  EXPECT_NEAR(s2.force()(0), expected_f2, std::abs(expected_f2) * 1e-10);
}

// ── Bulk thermodynamics ─────────────────────────────────────────────────────

TEST(Interaction, BulkFreeEnergyDensity) {
  auto s = make_species();
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);

  double rho = 0.5;
  double f = inter.bulk_free_energy_density(rho, rho);
  EXPECT_DOUBLE_EQ(f, 0.5 * inter.vdw_parameter() * rho * rho);
}

TEST(Interaction, BulkChemicalPotential) {
  auto s = make_species();
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);

  double rho = 0.5;
  double mu = inter.bulk_chemical_potential(rho);
  EXPECT_DOUBLE_EQ(mu, inter.vdw_parameter() * rho);
}

// ── Weight symmetry ─────────────────────────────────────────────────────────

TEST(Interaction, WeightSymmetryViaEnergyCommutes) {
  // E(rho1, rho2) should equal E(rho2, rho1) with swapped species
  double rho1 = 0.3;
  double rho2 = 0.7;
  auto s1a = make_species();
  auto s2a = make_species();
  s1a.density().values().fill(rho1);
  s2a.density().values().fill(rho2);

  auto s1b = make_species();
  auto s2b = make_species();
  s1b.density().values().fill(rho2);
  s2b.density().values().fill(rho1);

  auto lj = make_lj();
  Interaction inter_ab(s1a, s2a, lj, 1.0);
  Interaction inter_ba(s1b, s2b, lj, 1.0);

  double e_ab = inter_ab.compute_free_energy();
  double e_ba = inter_ba.compute_free_energy();

  EXPECT_NEAR(e_ab, e_ba, std::abs(e_ab) * 1e-12);
}

// ── Energy-force consistency ────────────────────────────────────────────────

TEST(Interaction, EnergyForceConsistency) {
  // Numerical dF/drho_i should match the force
  double rho0 = 0.5;
  auto s = make_species();
  s.density().values().fill(rho0);
  auto lj = make_lj();

  // Compute forces
  auto s_force = make_species();
  s_force.density().values().fill(rho0);
  Interaction inter_force(s_force, s_force, lj, 1.0);
  s_force.zero_force();
  inter_force.compute_forces();
  double analytic_force_0 = s_force.force()(0);

  // Numerical derivative at element 0
  double h = 1e-5;
  auto s_plus = make_species();
  s_plus.density().values().fill(rho0);
  s_plus.density().set(0, rho0 + h);
  Interaction inter_plus(s_plus, s_plus, lj, 1.0);
  double e_plus = inter_plus.compute_free_energy();

  auto s_minus = make_species();
  s_minus.density().values().fill(rho0);
  s_minus.density().set(0, rho0 - h);
  Interaction inter_minus(s_minus, s_minus, lj, 1.0);
  double e_minus = inter_minus.compute_free_energy();

  // For self-interaction: dF/drho_i = sum_j w(i-j) * rho_j * dV
  // The force from compute_forces adds w*rho*dV to the force vector.
  // But F = (1/2) sum rho_i w_ij rho_j dV, so dF/drho_i = sum_j w_ij rho_j dV
  // (the factor of 2 from the 1/2 and the symmetric sum cancels).
  double numerical_deriv = (e_plus - e_minus) / (2.0 * h);

  EXPECT_NEAR(analytic_force_0, numerical_deriv, std::abs(numerical_deriv) * 1e-4);
}

// ── Scheme comparison ───────────────────────────────────────────────────────

TEST(Interaction, InterpolationZeroVsLinearAgree) {
  auto s1 = make_species(0.25, {4.0, 4.0, 4.0});
  auto s2 = make_species(0.25, {4.0, 4.0, 4.0});
  auto lj = make_lj(1.0, 1.0, 2.5);

  Interaction inter_zero(s1, s1, lj, 1.0, WeightScheme::InterpolationZero);
  Interaction inter_linear(s2, s2, lj, 1.0, WeightScheme::InterpolationLinearF);

  // The two schemes should give similar a_vdw (not identical, but close)
  EXPECT_NEAR(inter_zero.vdw_parameter(), inter_linear.vdw_parameter(), std::abs(inter_zero.vdw_parameter()) * 0.1);
}

TEST(Interaction, GaussConvergence) {
  // Higher gauss order should give a_vdw closer to the continuous value
  // Use a finer grid so the gauss order matters more than the grid resolution
  auto s3 = make_species(0.1, {6.0, 6.0, 6.0});
  auto s5 = make_species(0.1, {6.0, 6.0, 6.0});
  auto lj = make_lj(1.0, 1.0, 2.5);

  auto lj_ref = make_lj(1.0, 1.0, 2.5);
  double a_continuous = lj_ref.compute_van_der_waals_integral(1.0);

  Interaction inter3(s3, s3, lj, 1.0, WeightScheme::GaussE, 2);
  Interaction inter5(s5, s5, lj, 1.0, WeightScheme::GaussE, 5);

  double err3 = std::abs(inter3.vdw_parameter() - a_continuous);
  double err5 = std::abs(inter5.vdw_parameter() - a_continuous);

  // Higher order should be more accurate
  EXPECT_LT(err5, err3);
}

TEST(Interaction, GaussEAndGaussFSameVdw) {
  auto s1 = make_species(0.25, {4.0, 4.0, 4.0});
  auto s2 = make_species(0.25, {4.0, 4.0, 4.0});
  auto lj = make_lj();

  Interaction inter_e(s1, s1, lj, 1.0, WeightScheme::GaussE);
  Interaction inter_f(s2, s2, lj, 1.0, WeightScheme::GaussF);

  // Both routes use the same weight generation, so a_vdw should be identical
  EXPECT_DOUBLE_EQ(inter_e.vdw_parameter(), inter_f.vdw_parameter());
}

// ── Cutoff respected ────────────────────────────────────────────────────────

TEST(Interaction, CutoffRespected) {
  // With a short cutoff, a_vdw should differ from a long cutoff
  auto s_short = make_species(0.5, {6.0, 6.0, 6.0});
  auto s_long = make_species(0.5, {6.0, 6.0, 6.0});

  auto lj_short = make_lj(1.0, 1.0, 1.5);
  auto lj_long = make_lj(1.0, 1.0, 3.0);

  Interaction inter_short(s_short, s_short, lj_short, 1.0);
  Interaction inter_long(s_long, s_long, lj_long, 1.0);

  // Longer cutoff captures more of the attractive tail, so |a_vdw| should be larger
  EXPECT_LT(inter_long.vdw_parameter(), inter_short.vdw_parameter());
}

// ── Sinusoidal density ──────────────────────────────────────────────────────

TEST(Interaction, SinusoidalDensityConvolution) {
  // For a sinusoidal density rho(r) = rho0 + A*sin(2*pi*x/L),
  // the convolution with any even kernel gives
  // (w*rho)(r) = a_vdw*rho0 + A*w_hat(k1)*sin(2*pi*x/L)
  // The energy should be: (1/2) * (a_vdw^2 * rho0^2 * V + A^2 * |w_hat(k1)|^2 * V/2) * dV
  // This is hard to check exactly, but we can verify that the energy differs from
  // what you'd get with a simple uniform density.
  double rho0 = 0.5;
  double amp = 0.1;
  double dx = 0.5;
  arma::rowvec3 box = {4.0, 4.0, 4.0};
  auto s = make_species(dx, box);
  auto lj = make_lj();

  auto& rho = s.density().values();
  long nx = s.density().shape()[0];
  long ny = s.density().shape()[1];
  long nz = s.density().shape()[2];
  for (long ix = 0; ix < nx; ++ix) {
    double x = static_cast<double>(ix) * dx;
    double val = rho0 + amp * std::sin(2.0 * std::numbers::pi * x / box(0));
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        rho(s.density().flat_index(ix, iy, iz)) = val;
      }
    }
  }

  Interaction inter(s, s, lj, 1.0);
  double energy_sin = inter.compute_free_energy();

  // Uniform at same rho0
  auto s_unif = make_species(dx, box);
  s_unif.density().values().fill(rho0);
  Interaction inter_unif(s_unif, s_unif, lj, 1.0);
  double energy_unif = inter_unif.compute_free_energy();

  // Energies should differ (sinusoidal has extra contribution from modulation)
  EXPECT_NE(energy_sin, energy_unif);
}

// ── Scheme and temperature accessors ────────────────────────────────────────

TEST(Interaction, SchemeAccessor) {
  auto s = make_species();
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0, WeightScheme::InterpolationZero);
  EXPECT_EQ(inter.scheme(), WeightScheme::InterpolationZero);
}

TEST(Interaction, TemperatureAccessor) {
  auto s = make_species();
  auto lj = make_lj();
  double kT = 2.5;
  Interaction inter(s, s, lj, kT);
  EXPECT_DOUBLE_EQ(inter.temperature(), kT);
}

// ── All schemes produce negative a_vdw for LJ ──────────────────────────────

TEST(Interaction, AllSchemesNegativeVdwForLJ) {
  auto lj = make_lj();

  auto test_scheme = [&](WeightScheme scheme) {
    auto s = make_species();
    Interaction inter(s, s, lj, 1.0, scheme);
    EXPECT_LT(inter.vdw_parameter(), 0.0) << "Scheme failed: " << static_cast<int>(scheme);
  };

  test_scheme(WeightScheme::InterpolationZero);
  test_scheme(WeightScheme::InterpolationLinearE);
  test_scheme(WeightScheme::InterpolationLinearF);
  test_scheme(WeightScheme::GaussE);
  test_scheme(WeightScheme::GaussF);
}

// ── Weight sum matches a_vdw ────────────────────────────────────────────────

TEST(Interaction, WeightSumMatchesVdw) {
  // The vdw_parameter = sum(w_real) * dV, verified from uniform density energy
  double rho0 = 1.0;
  auto s = make_species();
  s.density().values().fill(rho0);
  auto lj = make_lj();
  Interaction inter(s, s, lj, 1.0);

  double energy = inter.compute_free_energy();
  double dv = s.density().cell_volume();
  double volume = static_cast<double>(s.density().size()) * dv;

  // F = (1/2) * a_vdw * rho^2 * V
  // With rho = 1: F = (1/2) * a_vdw * V
  double a_from_energy = 2.0 * energy / volume;
  EXPECT_NEAR(a_from_energy, inter.vdw_parameter(), std::abs(inter.vdw_parameter()) * 1e-10);
}
