#include "cdft/physics/eos.hpp"

#include <gtest/gtest.h>

#include <cmath>

#include "cdft/numerics/autodiff.hpp"

namespace cdft::physics {

  TEST(CarnahanStarlingTest, ZeroPacking) {
    EXPECT_DOUBLE_EQ(CarnahanStarling::excess_free_energy(0.0), 0.0);
  }

  TEST(CarnahanStarlingTest, KnownValue) {
    double eta = 0.3;
    double f = CarnahanStarling::excess_free_energy(eta);
    double e = 1.0 - eta;
    double expected = eta * (4.0 - 3.0 * eta) / (e * e);
    EXPECT_NEAR(f, expected, 1e-14);
  }

  TEST(PercusYevickTest, ZeroPacking) {
    EXPECT_DOUBLE_EQ(PercusYevickVirial::excess_free_energy(0.0), 0.0);
    EXPECT_NEAR(PercusYevickCompressibility::excess_free_energy(0.0), 0.0, 1e-14);
  }

  TEST(HardSphereVariantTest, Dispatch) {
    HardSphereModel cs = CarnahanStarling{};
    double f = hs_excess_free_energy(cs, 0.3);
    EXPECT_NEAR(f, CarnahanStarling::excess_free_energy(0.3), 1e-14);
  }

  TEST(PackingFractionTest, Roundtrip) {
    double rho = 0.5;
    double eta = packing_fraction(rho);
    double rho_back = density_from_eta(eta);
    EXPECT_NEAR(rho_back, rho, 1e-14);
  }

  TEST(IdealGasTest, ZeroExcess) {
    IdealGas ig(1.0);
    EXPECT_DOUBLE_EQ(ig.excess_free_energy_per_particle(0.5), 0.0);
  }

  TEST(LennardJonesJZGTest, Construction) {
    LennardJonesJZG jzg(1.0);
    EXPECT_NO_THROW(jzg.excess_free_energy_per_particle(0.3));
  }

  TEST(LennardJonesMeckeTest, Construction) {
    LennardJonesMecke mecke(1.5);
    EXPECT_NO_THROW(mecke.excess_free_energy_per_particle(0.3));
  }

  TEST(EOSVariantTest, Dispatch) {
    EquationOfState eos = PercusYevickEOS(1.0);
    EXPECT_EQ(eos_name(eos), "PercusYevickEOS");
    EXPECT_GT(eos_temperature(eos), 0.0);
  }

  TEST(TransportTest, NonNegativeViscosity) {
    double rho = 0.5;
    double chi = contact_value(packing_fraction(rho));
    EXPECT_GT(transport::bulk_viscosity(rho, chi), 0.0);
    EXPECT_GT(transport::shear_viscosity(rho, chi), 0.0);
    EXPECT_GT(transport::thermal_conductivity(rho, chi), 0.0);
  }

  // ── AutoDiff tests ──────────────────────────────────────────────────────

  TEST(AutoDiffCSTest, FirstDerivative) {
    double eta = 0.3;
    auto [f, df] = derivatives_up_to_1(
        [](dual x) { return CarnahanStarling::excess_free_energy(x); }, eta);

    // Check value
    double e = 1.0 - eta;
    EXPECT_NEAR(f, eta * (4.0 - 3.0 * eta) / (e * e), 1e-14);

    // Verify derivative against finite difference
    double h = 1e-7;
    double fd = (CarnahanStarling::excess_free_energy(eta + h)
                 - CarnahanStarling::excess_free_energy(eta - h)) / (2.0 * h);
    EXPECT_NEAR(df, fd, 1e-6);
  }

  TEST(AutoDiffCSTest, HigherDerivatives) {
    double eta = 0.3;
    auto [f, df, d2f, d3f] = derivatives_up_to_3(
        [](dual3rd x) { return CarnahanStarling::excess_free_energy(x); }, eta);

    // Verify analytically: d/deta [eta*(4-3*eta)/(1-eta)^2]
    double e = 1.0 - eta;
    double expected_df = (4.0 - 2.0 * eta) / (e * e * e);
    EXPECT_NEAR(df, expected_df, 1e-12);
    EXPECT_FALSE(std::isnan(d2f));
    EXPECT_FALSE(std::isnan(d3f));
  }

  TEST(AutoDiffJZGTest, DerivativesConsistent) {
    LennardJonesJZG jzg(1.0);
    double rho = 0.3;
    auto [f, df, d2f, d3f] = derivatives_up_to_3(
        [&](dual3rd x) { return jzg.excess_free_energy_per_particle(x); }, rho);

    EXPECT_NEAR(f, jzg.excess_free_energy_per_particle(rho), 1e-12);

    // Verify df against finite difference  
    double h = 1e-7;
    double fd = (jzg.excess_free_energy_per_particle(rho + h)
                 - jzg.excess_free_energy_per_particle(rho - h)) / (2.0 * h);
    EXPECT_NEAR(df, fd, 1e-5);
    EXPECT_FALSE(std::isnan(d2f));
    EXPECT_FALSE(std::isnan(d3f));
  }

  TEST(AutoDiffMeckeTest, DerivativesConsistent) {
    LennardJonesMecke mecke(1.5);
    double rho = 0.3;
    auto [f, df] = derivatives_up_to_1(
        [&](dual x) { return mecke.excess_free_energy_per_particle(x); }, rho);

    EXPECT_NEAR(f, mecke.excess_free_energy_per_particle(rho), 1e-14);

    double h = 1e-7;
    double fd = (mecke.excess_free_energy_per_particle(rho + h)
                 - mecke.excess_free_energy_per_particle(rho - h)) / (2.0 * h);
    EXPECT_NEAR(df, fd, 1e-5);
  }

}  // namespace cdft::physics
