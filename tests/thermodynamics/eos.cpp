#include "dft/thermodynamics/eos.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace dft::thermodynamics::eos;

// ── Construction ────────────────────────────────────────────────────────────

TEST(EquationOfState, NegativeTemperatureThrows) {
  EXPECT_THROW(IdealGas(-1.0), std::invalid_argument);
  EXPECT_THROW(IdealGas(0.0), std::invalid_argument);
  EXPECT_THROW(LennardJonesJZG(-1.0), std::invalid_argument);
  EXPECT_THROW(LennardJonesMecke(0.0), std::invalid_argument);
}

TEST(EquationOfState, TemperatureAccessor) {
  EosModel model = IdealGas(1.5);
  EXPECT_DOUBLE_EQ(eos_temperature(model), 1.5);
}

// ── IdealGas ────────────────────────────────────────────────────────────────

TEST(IdealGas, ExcessFreeEnergyIsZero) {
  EosModel model = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos_excess_free_energy(model, 0.5), 0.0);
  EXPECT_DOUBLE_EQ(eos_excess_free_energy(model, 0.5, 1), 0.0);
  EXPECT_DOUBLE_EQ(eos_excess_free_energy(model, 0.5, 2), 0.0);
  EXPECT_DOUBLE_EQ(eos_excess_free_energy(model, 0.5, 3), 0.0);
}

TEST(IdealGas, FreeEnergyPerParticle) {
  EosModel model = IdealGas(1.0);
  double rho = 0.5;
  EXPECT_DOUBLE_EQ(eos_free_energy_per_particle(model, rho), std::log(rho) - 1.0);
}

TEST(IdealGas, PressureIsUnity) {
  EosModel model = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos_pressure(model, 0.3), 1.0);
  EXPECT_DOUBLE_EQ(eos_pressure(model, 1.0), 1.0);
}

TEST(IdealGas, Name) {
  EosModel model = IdealGas(1.0);
  EXPECT_EQ(eos_name(model), "IdealGas");
}

// ── PercusYevick ────────────────────────────────────────────────────────────

TEST(PercusYevick, Name) {
  EosModel model = PercusYevick(1.0);
  EXPECT_EQ(eos_name(model), "PercusYevick");
}

TEST(PercusYevick, ExcessFreeEnergyAtZeroDensity) {
  EosModel model = PercusYevick(1.0);
  EXPECT_NEAR(eos_excess_free_energy(model, 1e-10), 0.0, 1e-8);
}

TEST(PercusYevick, PressureIdealLimit) {
  EosModel model = PercusYevick(1.0);
  EXPECT_NEAR(eos_pressure(model, 1e-10), 1.0, 1e-8);
}

TEST(PercusYevick, ThermodynamicConsistency) {
  EosModel model = PercusYevick(1.0);
  double rho = 0.5;
  double p = eos_pressure(model, rho);
  double p_check = 1.0 + rho * eos_excess_free_energy(model, rho, 1);
  EXPECT_NEAR(p, p_check, 1e-12);
}

TEST(PercusYevick, DerivativeConsistencyNumerical) {
  EosModel model = PercusYevick(1.0);
  double rho = 0.4;
  double h = 1e-7;
  double numerical = (eos_excess_free_energy(model, rho + h) - eos_excess_free_energy(model, rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos_excess_free_energy(model, rho, 1), numerical, 1e-5);
}

// ── LennardJonesJZG ────────────────────────────────────────────────────────

TEST(LennardJonesJZG, Name) {
  EosModel model = LennardJonesJZG(1.0);
  EXPECT_EQ(eos_name(model), "LennardJonesJZG");
}

TEST(LennardJonesJZG, ExcessVanishesAtZeroDensity) {
  EosModel model = LennardJonesJZG(1.5);
  EXPECT_NEAR(eos_excess_free_energy(model, 1e-12), 0.0, 1e-8);
}

TEST(LennardJonesJZG, PressureIdealLimit) {
  EosModel model = LennardJonesJZG(1.5);
  EXPECT_NEAR(eos_pressure(model, 1e-10), 1.0, 1e-4);
}

TEST(LennardJonesJZG, FirstDerivativeNumerical) {
  EosModel model = LennardJonesJZG(1.0);
  double rho = 0.3;
  double h = 1e-7;
  double numerical = (eos_excess_free_energy(model, rho + h) - eos_excess_free_energy(model, rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos_excess_free_energy(model, rho, 1), numerical, 1e-4);
}

TEST(LennardJonesJZG, SecondDerivativeNumerical) {
  EosModel model = LennardJonesJZG(1.0);
  double rho = 0.3;
  double h = 1e-5;
  double numerical =
      (eos_excess_free_energy(model, rho + h, 1) - eos_excess_free_energy(model, rho - h, 1)) / (2.0 * h);
  EXPECT_NEAR(eos_excess_free_energy(model, rho, 2), numerical, 1e-3);
}

TEST(LennardJonesJZG, ThirdDerivativeNumerical) {
  EosModel model = LennardJonesJZG(1.0);
  double rho = 0.3;
  double h = 1e-4;
  double numerical =
      (eos_excess_free_energy(model, rho + h, 2) - eos_excess_free_energy(model, rho - h, 2)) / (2.0 * h);
  EXPECT_NEAR(eos_excess_free_energy(model, rho, 3), numerical, 1e-2);
}

TEST(LennardJonesJZG, CutoffCorrectionChangesResult) {
  EosModel eos_no_cut = LennardJonesJZG(1.0);
  EosModel eos_cut = LennardJonesJZG(1.0, 2.5);
  double rho = 0.5;
  EXPECT_NE(eos_excess_free_energy(eos_no_cut, rho), eos_excess_free_energy(eos_cut, rho));
}

TEST(LennardJonesJZG, ShiftedVsTruncated) {
  EosModel eos_truncated = LennardJonesJZG(1.0, 2.5, false);
  EosModel eos_shifted = LennardJonesJZG(1.0, 2.5, true);
  double rho = 0.5;
  EXPECT_NE(eos_excess_free_energy(eos_truncated, rho), eos_excess_free_energy(eos_shifted, rho));
}

// ── LennardJonesMecke ──────────────────────────────────────────────────────

TEST(LennardJonesMecke, Name) {
  EosModel model = LennardJonesMecke(1.0);
  EXPECT_EQ(eos_name(model), "LennardJonesMecke");
}

TEST(LennardJonesMecke, ExcessVanishesAtZeroDensity) {
  EosModel model = LennardJonesMecke(1.5);
  EXPECT_NEAR(eos_excess_free_energy(model, 1e-10), 0.0, 1e-4);
}

TEST(LennardJonesMecke, FirstDerivativeNumerical) {
  EosModel model = LennardJonesMecke(1.0);
  double rho = 0.3;
  double h = 1e-7;
  double numerical = (eos_excess_free_energy(model, rho + h) - eos_excess_free_energy(model, rho - h)) / (2.0 * h);
  EXPECT_NEAR(eos_excess_free_energy(model, rho, 1), numerical, 1e-4);
}

TEST(LennardJonesMecke, SecondDerivativeNumerical) {
  EosModel model = LennardJonesMecke(1.0);
  double rho = 0.3;
  double h = 1e-5;
  double numerical =
      (eos_excess_free_energy(model, rho + h, 1) - eos_excess_free_energy(model, rho - h, 1)) / (2.0 * h);
  EXPECT_NEAR(eos_excess_free_energy(model, rho, 2), numerical, 1e-2);
}

TEST(LennardJonesMecke, CutoffCorrection) {
  EosModel eos_no_cut = LennardJonesMecke(1.0);
  EosModel eos_cut = LennardJonesMecke(1.0, 2.5);
  double rho = 0.5;
  EXPECT_NE(eos_excess_free_energy(eos_no_cut, rho), eos_excess_free_energy(eos_cut, rho));
}

// ── Cross-model sanity ──────────────────────────────────────────────────────

TEST(EquationOfState, ExcessFreeEnergyDensityDefinition) {
  EosModel model = PercusYevick(1.0);
  double rho = 0.4;
  EXPECT_DOUBLE_EQ(eos_excess_free_energy_density(model, rho), rho * eos_excess_free_energy(model, rho));
}

TEST(EquationOfState, DExcessFreeEnergyDensityFormula) {
  EosModel model = PercusYevick(1.0);
  double rho = 0.4;
  double expected = eos_excess_free_energy(model, rho) + rho * eos_excess_free_energy(model, rho, 1);
  EXPECT_DOUBLE_EQ(eos_d_excess_free_energy_density(model, rho), expected);
}

TEST(EquationOfState, D2ExcessFreeEnergyDensityFormula) {
  EosModel model = PercusYevick(1.0);
  double rho = 0.4;
  double expected = 2.0 * eos_excess_free_energy(model, rho, 1) + rho * eos_excess_free_energy(model, rho, 2);
  EXPECT_DOUBLE_EQ(eos_d2_excess_free_energy_density(model, rho), expected);
}

TEST(EquationOfState, VariantUse) {
  EosModel model = LennardJonesJZG(1.0);
  EXPECT_EQ(eos_name(model), "LennardJonesJZG");
  EXPECT_NO_THROW((void)eos_free_energy_per_particle(model, 0.3));
}

// ── IdealGas: derived quantity coverage ─────────────────────────────────────

TEST(IdealGas, ExcessFreeEnergyDensityIsZero) {
  EosModel model = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos_excess_free_energy_density(model, 0.5), 0.0);
}

TEST(IdealGas, DExcessFreeEnergyDensityIsZero) {
  EosModel model = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos_d_excess_free_energy_density(model, 0.5), 0.0);
}

TEST(IdealGas, D2ExcessFreeEnergyDensityIsZero) {
  EosModel model = IdealGas(1.0);
  EXPECT_DOUBLE_EQ(eos_d2_excess_free_energy_density(model, 0.5), 0.0);
}

// ── LennardJonesMecke: missing coverage ─────────────────────────────────────

TEST(LennardJonesMecke, ThirdDerivativeNumerical) {
  EosModel model = LennardJonesMecke(1.0);
  double rho = 0.3;
  double h = 1e-3;
  double numerical =
      (eos_excess_free_energy(model, rho + h, 2) - eos_excess_free_energy(model, rho - h, 2)) / (2.0 * h);
  double analytic = eos_excess_free_energy(model, rho, 3);
  EXPECT_LT(analytic, 0.0);
  EXPECT_LT(numerical, 0.0);
  EXPECT_NEAR(analytic / numerical, 1.0, 0.3);
}

TEST(LennardJonesMecke, PressureIdealLimit) {
  EosModel model = LennardJonesMecke(1.5);
  EXPECT_NEAR(eos_pressure(model, 1e-10), 1.0, 1e-4);
}

TEST(LennardJonesMecke, ShiftedVsTruncated) {
  EosModel eos_truncated = LennardJonesMecke(1.0, 2.5, false);
  EosModel eos_shifted = LennardJonesMecke(1.0, 2.5, true);
  double rho = 0.5;
  EXPECT_NE(eos_excess_free_energy(eos_truncated, rho), eos_excess_free_energy(eos_shifted, rho));
}

// ── Copy/move semantics ─────────────────────────────────────────────────────

TEST(EquationOfState, CopyConstructor) {
  EosModel original = IdealGas(2.0);
  auto copy = original;
  EXPECT_DOUBLE_EQ(eos_temperature(copy), 2.0);
  EXPECT_DOUBLE_EQ(eos_pressure(copy, 0.5), eos_pressure(original, 0.5));
}

TEST(EquationOfState, MoveConstructor) {
  EosModel original = IdealGas(2.0);
  auto moved = std::move(original);
  EXPECT_DOUBLE_EQ(eos_temperature(moved), 2.0);
  EXPECT_DOUBLE_EQ(eos_pressure(moved, 0.5), 1.0);
}
