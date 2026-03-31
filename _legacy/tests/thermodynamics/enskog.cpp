#include "dft/thermodynamics/enskog.h"

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
  EXPECT_DOUBLE_EQ(contact_value(0.0), 1.0);
}

TEST(Enskog, ContactValueKnown) {
  double eta = 0.49;
  double expected = (1.0 - 0.5 * eta) / std::pow(1.0 - eta, 3);
  EXPECT_DOUBLE_EQ(contact_value(eta), expected);
}

// ── Excess free energy: dilute limit ────────────────────────────────────────

TEST(Enskog, ExcessFreeEnergyVanishesAtZeroEta) {
  EXPECT_DOUBLE_EQ(CarnahanStarling::excess_free_energy(0.0), 0.0);
  EXPECT_DOUBLE_EQ(PercusYevickVirial::excess_free_energy(0.0), 0.0);
  EXPECT_DOUBLE_EQ(PercusYevickCompressibility::excess_free_energy(0.0), 0.0);
}

// ── Excess free energy: known values ────────────────────────────────────────

TEST(Enskog, CSExcessFreeEnergyFormula) {
  double eta = 0.3;
  double expected = eta * (4.0 - 3.0 * eta) / ((1.0 - eta) * (1.0 - eta));
  EXPECT_DOUBLE_EQ(CarnahanStarling::excess_free_energy(eta), expected);
}

TEST(Enskog, PYVirialExcessFreeEnergyFormula) {
  double eta = 0.2;
  double expected = 2.0 * std::log(1.0 - eta) + 6.0 * eta / (1.0 - eta);
  EXPECT_DOUBLE_EQ(PercusYevickVirial::excess_free_energy(eta), expected);
}

TEST(Enskog, PYCompressibilityExcessFreeEnergyFormula) {
  double eta = 0.25;
  double e2 = (1.0 - eta) * (1.0 - eta);
  double expected = -std::log(1.0 - eta) + 1.5 * eta * (2.0 - eta) / e2;
  EXPECT_DOUBLE_EQ(PercusYevickCompressibility::excess_free_energy(eta), expected);
}

// ── Derivatives (d/dη): numerical consistency via autodiff ──────────────────

TEST(Enskog, CSFirstDerivativeNumericalCheck) {
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.2;
  double h = 1e-7;
  double numerical =
      (CarnahanStarling::excess_free_energy(eta + h) - CarnahanStarling::excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(hs_excess_free_energy(cs, eta, 1), numerical, 1e-5);
}

TEST(Enskog, CSSecondDerivativeNumericalCheck) {
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.2;
  double h = 1e-5;
  double numerical = (hs_excess_free_energy(cs, eta + h, 1) - hs_excess_free_energy(cs, eta - h, 1)) / (2.0 * h);
  EXPECT_NEAR(hs_excess_free_energy(cs, eta, 2), numerical, 1e-4);
}

TEST(Enskog, CSThirdDerivativeNumericalCheck) {
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.2;
  double h = 1e-4;
  double numerical = (hs_excess_free_energy(cs, eta + h, 2) - hs_excess_free_energy(cs, eta - h, 2)) / (2.0 * h);
  EXPECT_NEAR(hs_excess_free_energy(cs, eta, 3), numerical, 1e-3);
}

TEST(Enskog, PYCompressibilitySecondDerivativeNumerical) {
  HardSphereModel py = PercusYevickCompressibility{};
  double eta = 0.2;
  double h = 1e-5;
  double numerical = (hs_excess_free_energy(py, eta + h, 1) - hs_excess_free_energy(py, eta - h, 1)) / (2.0 * h);
  EXPECT_NEAR(hs_excess_free_energy(py, eta, 2), numerical, 1e-4);
}

TEST(Enskog, PYVirialSecondDerivativeNumerical) {
  HardSphereModel py = PercusYevickVirial{};
  double eta = 0.2;
  double h = 1e-5;
  double numerical = (hs_excess_free_energy(py, eta + h, 1) - hs_excess_free_energy(py, eta - h, 1)) / (2.0 * h);
  EXPECT_NEAR(hs_excess_free_energy(py, eta, 2), numerical, 1e-4);
}

// ── Pressure: dilute limit ──────────────────────────────────────────────────

TEST(Enskog, PressureIdealGasLimit) {
  HardSphereModel cs = CarnahanStarling{};
  HardSphereModel pyv = PercusYevickVirial{};
  HardSphereModel pyc = PercusYevickCompressibility{};

  EXPECT_DOUBLE_EQ(hs_pressure(cs, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(hs_pressure(pyv, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(hs_pressure(pyc, 0.0), 1.0);
}

// ── Pressure: known values ──────────────────────────────────────────────────

TEST(Enskog, CSPressureFormula) {
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.35;
  double e3 = std::pow(1.0 - eta, 3);
  double expected = (1.0 + eta + eta * eta - eta * eta * eta) / e3;
  EXPECT_NEAR(hs_pressure(cs, eta), expected, 1e-12);
}

// ── Pressure-free energy thermodynamic consistency ──────────────────────────

TEST(Enskog, CSPressureFromFreeEnergy) {
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.3;
  double p_direct = hs_pressure(cs, eta);
  double p_from_fe = 1.0 + eta * hs_excess_free_energy(cs, eta, 1);
  EXPECT_NEAR(p_direct, p_from_fe, 1e-12);
}

TEST(Enskog, PYVirialPressureFromFreeEnergy) {
  HardSphereModel py = PercusYevickVirial{};
  double eta = 0.3;
  double p_direct = hs_pressure(py, eta);
  double p_from_fe = 1.0 + eta * hs_excess_free_energy(py, eta, 1);
  EXPECT_NEAR(p_direct, p_from_fe, 1e-12);
}

TEST(Enskog, PYCompressibilityPressureFromFreeEnergy) {
  HardSphereModel py = PercusYevickCompressibility{};
  double eta = 0.3;
  double p_direct = hs_pressure(py, eta);
  double p_from_fe = 1.0 + eta * hs_excess_free_energy(py, eta, 1);
  EXPECT_NEAR(p_direct, p_from_fe, 1e-12);
}

// ── Chemical potential ──────────────────────────────────────────────────────

TEST(Enskog, ChemicalPotentialDefinition) {
  HardSphereModel cs = CarnahanStarling{};
  double rho = 0.5;
  double mu = hs_chemical_potential(cs, rho);
  double eta = packing_fraction(rho);
  double expected = hs_free_energy(cs, rho) + hs_pressure(cs, eta);
  EXPECT_DOUBLE_EQ(mu, expected);
}

// ── Total free energy includes ideal gas ────────────────────────────────────

TEST(Enskog, TotalFreeEnergyIncludesIdealPart) {
  HardSphereModel cs = CarnahanStarling{};
  double rho = 1.0;
  double eta = packing_fraction(rho);
  EXPECT_DOUBLE_EQ(hs_free_energy(cs, rho), -1.0 + CarnahanStarling::excess_free_energy(eta));
}

// ── Transport coefficients ──────────────────────────────────────────────────

TEST(Enskog, TransportCoefficientsPositive) {
  double rho = 0.5;
  double chi = contact_value(packing_fraction(rho));
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
  EXPECT_NEAR(transport::shear_viscosity(0.0, 1.0), expected, 1e-12);
}

// ── Polymorphism via variant ────────────────────────────────────────────────

TEST(Enskog, VariantUse) {
  HardSphereModel model = CarnahanStarling{};
  EXPECT_GT(hs_pressure(model, 0.3), 1.0);
  EXPECT_NO_THROW((void)hs_chemical_potential(model, 0.5));
}

// ── PY derivative numerical checks ──────────────────────────────────────────

TEST(Enskog, PYVirialFirstDerivativeNumerical) {
  HardSphereModel py = PercusYevickVirial{};
  double eta = 0.2;
  double h = 1e-7;
  double numerical =
      (PercusYevickVirial::excess_free_energy(eta + h) - PercusYevickVirial::excess_free_energy(eta - h)) / (2.0 * h);
  EXPECT_NEAR(hs_excess_free_energy(py, eta, 1), numerical, 1e-5);
}

TEST(Enskog, PYCompressibilityFirstDerivativeNumerical) {
  HardSphereModel py = PercusYevickCompressibility{};
  double eta = 0.2;
  double h = 1e-7;
  double numerical = (PercusYevickCompressibility::excess_free_energy(eta + h) -
                      PercusYevickCompressibility::excess_free_energy(eta - h)) /
      (2.0 * h);
  EXPECT_NEAR(hs_excess_free_energy(py, eta, 1), numerical, 1e-5);
}

TEST(Enskog, PYVirialThirdDerivativeNumerical) {
  HardSphereModel py = PercusYevickVirial{};
  double eta = 0.2;
  double h = 1e-4;
  double numerical = (hs_excess_free_energy(py, eta + h, 2) - hs_excess_free_energy(py, eta - h, 2)) / (2.0 * h);
  EXPECT_NEAR(hs_excess_free_energy(py, eta, 3), numerical, 1e-3);
}
