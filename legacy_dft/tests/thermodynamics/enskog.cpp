#include "classicaldft_bits/thermodynamics/enskog.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft::thermodynamics;

// ── Packing fraction conversion ─────────────────────────────────────────────

TEST(Enskog, PackingFractionFromDensity) {
  EXPECT_DOUBLE_EQ(packing_fraction(0.0), 0.0);
  EXPECT_DOUBLE_EQ(packing_fraction(6.0 / std::numbers::pi), 1.0);
}

TEST(Enskog, DensityFromEtaRoundTrip) {
  for (double eta : {0.0, 0.1, 0.25, 0.4, 0.49}) {
    EXPECT_NEAR(packing_fraction(density_from_eta(eta)), eta, 1e-15);
  }
}

// ── Contact value ───────────────────────────────────────────────────────────

TEST(Enskog, ContactValueAtZeroDensity) {
  CarnahanStarling cs;
  EXPECT_DOUBLE_EQ(cs.contact_value(0.0), 1.0);
}

TEST(Enskog, ContactValueKnown) {
  CarnahanStarling cs;
  double eta = 0.49;
  double expected = (1.0 - 0.5 * eta) / std::pow(1.0 - eta, 3);
  EXPECT_DOUBLE_EQ(cs.contact_value(eta), expected);
}

// ── Excess free energy: dilute limit ────────────────────────────────────────

TEST(Enskog, ExcessFreeEnergyVanishesAtZeroEta) {
  CarnahanStarling cs;
  PercusYevick pyv(PercusYevick::Route::Virial);
  PercusYevick pyc(PercusYevick::Route::Compressibility);

  EXPECT_DOUBLE_EQ(cs.excess_free_energy(0.0), 0.0);
  EXPECT_DOUBLE_EQ(pyv.excess_free_energy(0.0), 0.0);
  EXPECT_DOUBLE_EQ(pyc.excess_free_energy(0.0), 0.0);
}

// ── Excess free energy: known values ────────────────────────────────────────

TEST(Enskog, CSExcessFreeEnergyFormula) {
  CarnahanStarling cs;
  double eta = 0.3;
  double expected = eta * (4.0 - 3.0 * eta) / ((1.0 - eta) * (1.0 - eta));
  EXPECT_DOUBLE_EQ(cs.excess_free_energy(eta), expected);
}

TEST(Enskog, PYVirialExcessFreeEnergyFormula) {
  PercusYevick py(PercusYevick::Route::Virial);
  double eta = 0.2;
  double expected = 2.0 * std::log(1.0 - eta) + 6.0 * eta / (1.0 - eta);
  EXPECT_DOUBLE_EQ(py.excess_free_energy(eta), expected);
}

TEST(Enskog, PYCompressibilityExcessFreeEnergyFormula) {
  PercusYevick py(PercusYevick::Route::Compressibility);
  double eta = 0.25;
  double e2 = (1.0 - eta) * (1.0 - eta);
  double expected = -std::log(1.0 - eta) + 1.5 * eta * (2.0 - eta) / e2;
  EXPECT_DOUBLE_EQ(py.excess_free_energy(eta), expected);
}

// ── Derivatives (d/dη): numerical consistency ───────────────────────────────

TEST(Enskog, CSFirstDerivativeNumericalCheck) {
  CarnahanStarling cs;
  double eta = 0.2;
  double h = 1e-7;
  double numerical = (cs.excess_free_energy(eta + h) - cs.excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(cs.d_excess_free_energy(eta), numerical, 1e-5);
}

TEST(Enskog, CSSecondDerivativeNumericalCheck) {
  CarnahanStarling cs;
  double eta = 0.2;
  double h = 1e-5;
  double numerical = (cs.d_excess_free_energy(eta + h) - cs.d_excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(cs.d2_excess_free_energy(eta), numerical, 1e-4);
}

TEST(Enskog, CSThirdDerivativeNumericalCheck) {
  CarnahanStarling cs;
  double eta = 0.2;
  double h = 1e-4;
  double numerical = (cs.d2_excess_free_energy(eta + h) - cs.d2_excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(cs.d3_excess_free_energy(eta), numerical, 1e-3);
}

TEST(Enskog, PYCompressibilitySecondDerivativeNumerical) {
  PercusYevick py(PercusYevick::Route::Compressibility);
  double eta = 0.2;
  double h = 1e-5;
  double numerical = (py.d_excess_free_energy(eta + h) - py.d_excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(py.d2_excess_free_energy(eta), numerical, 1e-4);
}

TEST(Enskog, PYVirialSecondDerivativeNumerical) {
  PercusYevick py(PercusYevick::Route::Virial);
  double eta = 0.2;
  double h = 1e-5;
  double numerical = (py.d_excess_free_energy(eta + h) - py.d_excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(py.d2_excess_free_energy(eta), numerical, 1e-4);
}

// ── Pressure: dilute limit ──────────────────────────────────────────────────

TEST(Enskog, PressureIdealGasLimit) {
  CarnahanStarling cs;
  PercusYevick pyv(PercusYevick::Route::Virial);
  PercusYevick pyc(PercusYevick::Route::Compressibility);

  EXPECT_DOUBLE_EQ(cs.pressure(0.0), 1.0);
  EXPECT_DOUBLE_EQ(pyv.pressure(0.0), 1.0);
  EXPECT_DOUBLE_EQ(pyc.pressure(0.0), 1.0);
}

// ── Pressure: known values ──────────────────────────────────────────────────

TEST(Enskog, CSPressureFormula) {
  CarnahanStarling cs;
  double eta = 0.35;
  double e3 = std::pow(1.0 - eta, 3);
  double expected = (1.0 + eta + eta * eta - eta * eta * eta) / e3;
  EXPECT_NEAR(cs.pressure(eta), expected, 1e-12);
}

// ── Pressure-free energy thermodynamic consistency ──────────────────────────

TEST(Enskog, CSPressureFromFreeEnergy) {
  CarnahanStarling cs;
  double eta = 0.3;
  double p_direct = cs.pressure(eta);
  double p_from_fe = 1.0 + eta * cs.d_excess_free_energy(eta);
  EXPECT_NEAR(p_direct, p_from_fe, 1e-12);
}

TEST(Enskog, PYVirialPressureFromFreeEnergy) {
  PercusYevick py(PercusYevick::Route::Virial);
  double eta = 0.3;
  double p_direct = py.pressure(eta);
  double p_from_fe = 1.0 + eta * py.d_excess_free_energy(eta);
  EXPECT_NEAR(p_direct, p_from_fe, 1e-12);
}

TEST(Enskog, PYCompressibilityPressureFromFreeEnergy) {
  PercusYevick py(PercusYevick::Route::Compressibility);
  double eta = 0.3;
  double p_direct = py.pressure(eta);
  double p_from_fe = 1.0 + eta * py.d_excess_free_energy(eta);
  EXPECT_NEAR(p_direct, p_from_fe, 1e-12);
}

// ── Chemical potential ──────────────────────────────────────────────────────

TEST(Enskog, ChemicalPotentialDefinition) {
  CarnahanStarling cs;
  double rho = 0.5;
  double mu = cs.chemical_potential(rho);
  double eta = packing_fraction(rho);
  double expected = cs.free_energy(rho) + cs.pressure(eta);
  EXPECT_DOUBLE_EQ(mu, expected);
}

// ── Total free energy includes ideal gas ────────────────────────────────────

TEST(Enskog, TotalFreeEnergyIncludesIdealPart) {
  CarnahanStarling cs;
  double rho = 1.0;
  double eta = packing_fraction(rho);
  EXPECT_DOUBLE_EQ(cs.free_energy(rho), -1.0 + cs.excess_free_energy(eta));
}

// ── Transport coefficients ──────────────────────────────────────────────────

TEST(Enskog, TransportCoefficientsPositive) {
  CarnahanStarling cs;
  double rho = 0.5;
  double chi = cs.contact_value(packing_fraction(rho));
  EXPECT_GT(transport::shear_viscosity(rho, chi), 0.0);
  EXPECT_GT(transport::bulk_viscosity(rho, chi), 0.0);
  EXPECT_GT(transport::thermal_conductivity(rho, chi), 0.0);
  EXPECT_GT(transport::sound_damping(rho, chi), 0.0);
}

TEST(Enskog, BulkViscosityZeroAtZeroDensity) {
  EXPECT_DOUBLE_EQ(transport::bulk_viscosity(0.0, 1.0), 0.0);
}

TEST(Enskog, ShearViscosityDiluteLimit) {
  // At zero density, chi=1, only the kinetic term survives:
  // eta_s = 5/(16*sqrt(pi)) * 1^2 + 0.6 * 0 = 5/(16*sqrt(pi))
  double expected = 5.0 / (16.0 * std::sqrt(std::numbers::pi));
  EXPECT_DOUBLE_EQ(transport::shear_viscosity(0.0, 1.0), expected);
}

// ── All three models agree in dilute limit ──────────────────────────────────

TEST(Enskog, AllModelsAgreeInDiluteLimit) {
  CarnahanStarling cs;
  PercusYevick pyv(PercusYevick::Route::Virial);
  PercusYevick pyc(PercusYevick::Route::Compressibility);

  double rho = 1e-8;
  double ideal = std::log(rho) - 1.0;
  EXPECT_NEAR(cs.free_energy(rho), ideal, 1e-6);
  EXPECT_NEAR(pyv.free_energy(rho), ideal, 1e-6);
  EXPECT_NEAR(pyc.free_energy(rho), ideal, 1e-6);
}

// ── Virial expansion: second virial coefficient ─────────────────────────────

TEST(Enskog, SecondVirialCoefficientCS) {
  CarnahanStarling cs;
  double h = 1e-10;
  double slope = cs.excess_free_energy(h) / h;
  EXPECT_NEAR(slope, 4.0, 1e-4);
}

TEST(Enskog, SecondVirialCoefficientPYVirial) {
  PercusYevick py(PercusYevick::Route::Virial);
  double h = 1e-10;
  double slope = py.excess_free_energy(h) / h;
  EXPECT_NEAR(slope, 4.0, 1e-4);
}

TEST(Enskog, SecondVirialCoefficientPYCompressibility) {
  PercusYevick py(PercusYevick::Route::Compressibility);
  double h = 1e-10;
  double slope = py.excess_free_energy(h) / h;
  EXPECT_NEAR(slope, 4.0, 1e-4);
}

// ── Polymorphism via base pointer ───────────────────────────────────────────

TEST(Enskog, PolymorphicUse) {
  std::unique_ptr<HardSphereFluid> model = std::make_unique<CarnahanStarling>();
  EXPECT_GT(model->pressure(0.3), 1.0);
  EXPECT_NO_THROW((void)model->chemical_potential(0.5));
}

// ── PY Virial: first derivative numerical check ─────────────────────────────

TEST(Enskog, PYVirialFirstDerivativeNumerical) {
  PercusYevick py(PercusYevick::Route::Virial);
  double eta = 0.2;
  double h = 1e-7;
  double numerical = (py.excess_free_energy(eta + h) - py.excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(py.d_excess_free_energy(eta), numerical, 1e-5);
}

TEST(Enskog, PYCompressibilityFirstDerivativeNumerical) {
  PercusYevick py(PercusYevick::Route::Compressibility);
  double eta = 0.2;
  double h = 1e-7;
  double numerical = (py.excess_free_energy(eta + h) - py.excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(py.d_excess_free_energy(eta), numerical, 1e-5);
}

// ── PY third derivative numerical check ─────────────────────────────────────

TEST(Enskog, PYVirialThirdDerivativeNumerical) {
  PercusYevick py(PercusYevick::Route::Virial);
  double eta = 0.2;
  double h = 1e-4;
  double numerical = (py.d2_excess_free_energy(eta + h) - py.d2_excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(py.d3_excess_free_energy(eta), numerical, 1e-3);
}

TEST(Enskog, PYCompressibilityThirdDerivativeNumerical) {
  PercusYevick py(PercusYevick::Route::Compressibility);
  double eta = 0.2;
  double h = 1e-4;
  double numerical = (py.d2_excess_free_energy(eta + h) - py.d2_excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(py.d3_excess_free_energy(eta), numerical, 1e-3);
}

// ── Contact value via PY ────────────────────────────────────────────────────

TEST(Enskog, ContactValueViaPY) {
  PercusYevick py;
  double eta = 0.35;
  double expected = (1.0 - 0.5 * eta) / std::pow(1.0 - eta, 3);
  EXPECT_DOUBLE_EQ(py.contact_value(eta), expected);
}

// ── Chemical potential for PY ───────────────────────────────────────────────

TEST(Enskog, ChemicalPotentialPYVirial) {
  PercusYevick py(PercusYevick::Route::Virial);
  double rho = 0.5;
  double eta = packing_fraction(rho);
  double mu = py.chemical_potential(rho);
  double expected = py.free_energy(rho) + py.pressure(eta);
  EXPECT_DOUBLE_EQ(mu, expected);
}

TEST(Enskog, ChemicalPotentialPYCompressibility) {
  PercusYevick py(PercusYevick::Route::Compressibility);
  double rho = 0.5;
  double eta = packing_fraction(rho);
  double mu = py.chemical_potential(rho);
  double expected = py.free_energy(rho) + py.pressure(eta);
  EXPECT_DOUBLE_EQ(mu, expected);
}

// ── Free energy at general density ──────────────────────────────────────────

TEST(Enskog, CSFreeEnergyAtFiniteDensity) {
  CarnahanStarling cs;
  double rho = 0.5;
  double eta = packing_fraction(rho);
  double expected = std::log(rho) - 1.0 + cs.excess_free_energy(eta);
  EXPECT_DOUBLE_EQ(cs.free_energy(rho), expected);
}

// ── Transport at finite density ─────────────────────────────────────────────

TEST(Enskog, ThermalConductivityDiluteLimit) {
  // At zero density, chi=1: lambda = 75/(64*sqrt(pi)) * 1^2 + 0 = 75/(64*sqrt(pi))
  double expected = 75.0 / (64.0 * std::sqrt(std::numbers::pi));
  EXPECT_DOUBLE_EQ(transport::thermal_conductivity(0.0, 1.0), expected);
}

TEST(Enskog, SoundDampingPositiveAtFiniteDensity) {
  CarnahanStarling cs;
  double rho = 0.3;
  double chi = cs.contact_value(packing_fraction(rho));
  EXPECT_GT(transport::sound_damping(rho, chi), 0.0);
}
