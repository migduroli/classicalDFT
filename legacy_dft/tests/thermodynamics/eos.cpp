#include "classicaldft_bits/thermodynamics/eos.h"

#include <cmath>
#include <gtest/gtest.h>
#include <memory>

using namespace dft::thermodynamics::eos;

// ── Construction ────────────────────────────────────────────────────────────

TEST(EquationOfState, NegativeTemperatureThrows) {
  EXPECT_THROW(IdealGas(-1.0), std::invalid_argument);
  EXPECT_THROW(IdealGas(0.0), std::invalid_argument);
}

TEST(EquationOfState, TemperatureAccessor) {
  auto eos = IdealGas(1.5);
  EXPECT_DOUBLE_EQ(eos.temperature(), 1.5);
}

// ── IdealGas ────────────────────────────────────────────────────────────────

TEST(IdealGas, ExcessFreeEnergyIsZero) {
  auto eos = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos.excess_free_energy_per_particle(0.5), 0.0);
  EXPECT_DOUBLE_EQ(eos.d_excess_free_energy_per_particle(0.5), 0.0);
  EXPECT_DOUBLE_EQ(eos.d2_excess_free_energy_per_particle(0.5), 0.0);
  EXPECT_DOUBLE_EQ(eos.d3_excess_free_energy_per_particle(0.5), 0.0);
}

TEST(IdealGas, FreeEnergyPerParticle) {
  auto eos = IdealGas(1.0);
  double rho = 0.5;
  EXPECT_DOUBLE_EQ(eos.free_energy_per_particle(rho), std::log(rho) - 1.0);
}

TEST(IdealGas, PressureIsUnity) {
  auto eos = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos.pressure(0.3), 1.0);
  EXPECT_DOUBLE_EQ(eos.pressure(1.0), 1.0);
}

TEST(IdealGas, Name) {
  auto eos = IdealGas(1.0);
  EXPECT_EQ(eos.name(), "IdealGas");
}

// ── PercusYevick ────────────────────────────────────────────────────────────

TEST(PercusYevick, Name) {
  auto eos = PercusYevick(1.0);
  EXPECT_EQ(eos.name(), "PercusYevick");
}

TEST(PercusYevick, ExcessFreeEnergyAtZeroDensity) {
  auto eos = PercusYevick(1.0);
  EXPECT_NEAR(eos.excess_free_energy_per_particle(1e-10), 0.0, 1e-8);
}

TEST(PercusYevick, PressureIdealLimit) {
  auto eos = PercusYevick(1.0);
  EXPECT_NEAR(eos.pressure(1e-10), 1.0, 1e-8);
}

TEST(PercusYevick, ThermodynamicConsistency) {
  // P/(rho kT) = 1 + rho * d(phi_ex)/d(rho), verified via base class
  auto eos = PercusYevick(1.0);
  double rho = 0.5;
  double p = eos.pressure(rho);
  double p_check = 1.0 + rho * eos.d_excess_free_energy_per_particle(rho);
  EXPECT_NEAR(p, p_check, 1e-12);
}

TEST(PercusYevick, DerivativeConsistencyNumerical) {
  auto eos = PercusYevick(1.0);
  double rho = 0.4;
  double h = 1e-7;
  double numerical =
      (eos.excess_free_energy_per_particle(rho + h) - eos.excess_free_energy_per_particle(rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos.d_excess_free_energy_per_particle(rho), numerical, 1e-5);
}

// ── LennardJonesJZG ────────────────────────────────────────────────────────

TEST(LennardJonesJZG, Name) {
  auto eos = LennardJonesJZG(1.0);
  EXPECT_EQ(eos.name(), "LennardJonesJZG");
}

TEST(LennardJonesJZG, ExcessVanishesAtZeroDensity) {
  auto eos = LennardJonesJZG(1.5);
  EXPECT_NEAR(eos.excess_free_energy_per_particle(1e-12), 0.0, 1e-8);
}

TEST(LennardJonesJZG, PressureIdealLimit) {
  auto eos = LennardJonesJZG(1.5);
  EXPECT_NEAR(eos.pressure(1e-10), 1.0, 1e-4);
}

TEST(LennardJonesJZG, FirstDerivativeNumerical) {
  auto eos = LennardJonesJZG(1.0);
  double rho = 0.3;
  double h = 1e-7;
  double numerical =
      (eos.excess_free_energy_per_particle(rho + h) - eos.excess_free_energy_per_particle(rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos.d_excess_free_energy_per_particle(rho), numerical, 1e-4);
}

TEST(LennardJonesJZG, SecondDerivativeNumerical) {
  auto eos = LennardJonesJZG(1.0);
  double rho = 0.3;
  double h = 1e-5;
  double numerical =
      (eos.d_excess_free_energy_per_particle(rho + h) - eos.d_excess_free_energy_per_particle(rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos.d2_excess_free_energy_per_particle(rho), numerical, 1e-3);
}

TEST(LennardJonesJZG, ThirdDerivativeNumerical) {
  auto eos = LennardJonesJZG(1.0);
  double rho = 0.3;
  double h = 1e-4;
  double numerical =
      (eos.d2_excess_free_energy_per_particle(rho + h) - eos.d2_excess_free_energy_per_particle(rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos.d3_excess_free_energy_per_particle(rho), numerical, 1e-2);
}

TEST(LennardJonesJZG, CutoffCorrectionChangesResult) {
  auto eos_no_cut = LennardJonesJZG(1.0);
  auto eos_cut = LennardJonesJZG(1.0, 2.5);
  double rho = 0.5;
  EXPECT_NE(eos_no_cut.excess_free_energy_per_particle(rho), eos_cut.excess_free_energy_per_particle(rho));
}

TEST(LennardJonesJZG, ShiftedVsTruncated) {
  auto eos_truncated = LennardJonesJZG(1.0, 2.5, false);
  auto eos_shifted = LennardJonesJZG(1.0, 2.5, true);
  double rho = 0.5;
  EXPECT_NE(eos_truncated.excess_free_energy_per_particle(rho), eos_shifted.excess_free_energy_per_particle(rho));
}

// ── LennardJonesMecke ──────────────────────────────────────────────────────

TEST(LennardJonesMecke, Name) {
  auto eos = LennardJonesMecke(1.0);
  EXPECT_EQ(eos.name(), "LennardJonesMecke");
}

TEST(LennardJonesMecke, ExcessVanishesAtZeroDensity) {
  auto eos = LennardJonesMecke(1.5);
  EXPECT_NEAR(eos.excess_free_energy_per_particle(1e-10), 0.0, 1e-4);
}

TEST(LennardJonesMecke, FirstDerivativeNumerical) {
  auto eos = LennardJonesMecke(1.0);
  double rho = 0.3;
  double h = 1e-7;
  double numerical =
      (eos.excess_free_energy_per_particle(rho + h) - eos.excess_free_energy_per_particle(rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos.d_excess_free_energy_per_particle(rho), numerical, 1e-4);
}

TEST(LennardJonesMecke, SecondDerivativeNumerical) {
  auto eos = LennardJonesMecke(1.0);
  double rho = 0.3;
  double h = 1e-5;
  double numerical =
      (eos.d_excess_free_energy_per_particle(rho + h) - eos.d_excess_free_energy_per_particle(rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos.d2_excess_free_energy_per_particle(rho), numerical, 1e-2);
}

TEST(LennardJonesMecke, CutoffCorrection) {
  auto eos_no_cut = LennardJonesMecke(1.0);
  auto eos_cut = LennardJonesMecke(1.0, 2.5);
  double rho = 0.5;
  EXPECT_NE(eos_no_cut.excess_free_energy_per_particle(rho), eos_cut.excess_free_energy_per_particle(rho));
}

// ── Cross-model sanity ──────────────────────────────────────────────────────

TEST(EquationOfState, ExcessFreeEnergyDensityDefinition) {
  // f_ex_density = rho * phi_ex(rho)
  auto eos = PercusYevick(1.0);
  double rho = 0.4;
  EXPECT_DOUBLE_EQ(eos.excess_free_energy_density(rho), rho * eos.excess_free_energy_per_particle(rho));
}

TEST(EquationOfState, DExcessFreeEnergyDensityFormula) {
  // d/drho [rho * phi_ex] = phi_ex + rho * d(phi_ex)/drho
  auto eos = PercusYevick(1.0);
  double rho = 0.4;
  double expected = eos.excess_free_energy_per_particle(rho) + rho * eos.d_excess_free_energy_per_particle(rho);
  EXPECT_DOUBLE_EQ(eos.d_excess_free_energy_density(rho), expected);
}

TEST(EquationOfState, D2ExcessFreeEnergyDensityFormula) {
  auto eos = PercusYevick(1.0);
  double rho = 0.4;
  double expected =
      2.0 * eos.d_excess_free_energy_per_particle(rho) + rho * eos.d2_excess_free_energy_per_particle(rho);
  EXPECT_DOUBLE_EQ(eos.d2_excess_free_energy_density(rho), expected);
}

TEST(EquationOfState, PolymorphicUseViaBasePointer) {
  std::unique_ptr<EquationOfState> eos = std::make_unique<LennardJonesJZG>(1.0);
  EXPECT_EQ(eos->name(), "LennardJonesJZG");
  EXPECT_NO_THROW((void)eos->free_energy_per_particle(0.3));
}

// ── IdealGas: derived quantity coverage ─────────────────────────────────────

TEST(IdealGas, ExcessFreeEnergyDensityIsZero) {
  auto eos = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos.excess_free_energy_density(0.5), 0.0);
}

TEST(IdealGas, DExcessFreeEnergyDensityIsZero) {
  auto eos = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos.d_excess_free_energy_density(0.5), 0.0);
}

TEST(IdealGas, D2ExcessFreeEnergyDensityIsZero) {
  auto eos = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos.d2_excess_free_energy_density(0.5), 0.0);
}

// ── LennardJonesMecke: missing coverage ─────────────────────────────────────

TEST(LennardJonesMecke, ThirdDerivativeNumerical) {
  // Mecke d3 is itself a finite-difference stencil, so we verify its sign and
  // order of magnitude against an independent numerical estimate.
  auto eos = LennardJonesMecke(1.0);
  double rho = 0.3;
  double h = 1e-3;
  double numerical =
      (eos.d2_excess_free_energy_per_particle(rho + h) - eos.d2_excess_free_energy_per_particle(rho - h)) / (2.0 * h);
  double analytic = eos.d3_excess_free_energy_per_particle(rho);
  // Both should be negative and within 30% of each other
  EXPECT_LT(analytic, 0.0);
  EXPECT_LT(numerical, 0.0);
  EXPECT_NEAR(analytic / numerical, 1.0, 0.3);
}

TEST(LennardJonesMecke, PressureIdealLimit) {
  auto eos = LennardJonesMecke(1.5);
  EXPECT_NEAR(eos.pressure(1e-10), 1.0, 1e-4);
}

TEST(LennardJonesMecke, ShiftedVsTruncated) {
  auto eos_truncated = LennardJonesMecke(1.0, 2.5, false);
  auto eos_shifted = LennardJonesMecke(1.0, 2.5, true);
  double rho = 0.5;
  EXPECT_NE(eos_truncated.excess_free_energy_per_particle(rho), eos_shifted.excess_free_energy_per_particle(rho));
}

// ── EquationOfState: copy/move semantics ────────────────────────────────────

TEST(EquationOfState, CopyConstructor) {
  auto original = IdealGas(2.0);
  auto copy = original;
  EXPECT_DOUBLE_EQ(copy.temperature(), 2.0);
  EXPECT_DOUBLE_EQ(copy.pressure(0.5), original.pressure(0.5));
}

TEST(EquationOfState, MoveConstructor) {
  auto original = IdealGas(2.0);
  auto moved = std::move(original);
  EXPECT_DOUBLE_EQ(moved.temperature(), 2.0);
  EXPECT_DOUBLE_EQ(moved.pressure(0.5), 1.0);
}
