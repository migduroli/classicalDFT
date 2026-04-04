#include "dft/physics/hard_spheres.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

using namespace dft::physics::hard_spheres;

// Packing fraction conversion

TEST_CASE("packing fraction round-trip", "[hard_spheres]") {
  double rho = 0.5;
  double eta = packing_fraction(rho);
  double rho_back = density_from_eta(eta);

  CHECK(rho_back == Catch::Approx(rho).margin(1e-14));
}

TEST_CASE("packing fraction at unit density", "[hard_spheres]") {
  CHECK(packing_fraction(1.0) == Catch::Approx(std::numbers::pi / 6.0).margin(1e-14));
}

// Contact value

TEST_CASE("contact value at eta=0 is 1", "[hard_spheres]") {
  CHECK(contact_value(0.0) == Catch::Approx(1.0).margin(1e-14));
}

TEST_CASE("contact value increases with eta", "[hard_spheres]") {
  CHECK(contact_value(0.3) > contact_value(0.1));
}

// Carnahan-Starling excess free energy

TEST_CASE("CS excess free energy is zero at eta=0", "[hard_spheres]") {
  CHECK(CarnahanStarling::excess_free_energy(0.0) == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("CS excess free energy is positive for eta>0", "[hard_spheres]") {
  CHECK(CarnahanStarling::excess_free_energy(0.3) > 0.0);
}

// PercusYevickVirial

TEST_CASE("PY virial excess free energy is zero at eta=0", "[hard_spheres]") {
  CHECK(PercusYevickVirial::excess_free_energy(0.0) == Catch::Approx(0.0).margin(1e-14));
}

// PercusYevickCompressibility

TEST_CASE("PY compressibility excess free energy is zero at eta=0", "[hard_spheres]") {
  CHECK(PercusYevickCompressibility::excess_free_energy(0.0) == Catch::Approx(0.0).margin(1e-14));
}

// Variant-based free functions

TEST_CASE("excess_free_energy dispatches correctly", "[hard_spheres]") {
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.3;
  double expected = CarnahanStarling::excess_free_energy(eta);

  CHECK(excess_free_energy(cs, eta) == Catch::Approx(expected).margin(1e-14));
}

// Autodiff derivatives

TEST_CASE("d_excess_free_energy matches numerical derivative for CS", "[hard_spheres]") {
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.3;
  double h = 1e-6;

  double numerical = (excess_free_energy(cs, eta + h) - excess_free_energy(cs, eta - h)) / (2.0 * h);
  double analytic = d_excess_free_energy(cs, eta);

  CHECK(analytic == Catch::Approx(numerical).margin(1e-5));
}

TEST_CASE("d2_excess_free_energy matches numerical second derivative", "[hard_spheres]") {
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.3;
  double h = 1e-5;

  double numerical =
      (excess_free_energy(cs, eta + h) - 2.0 * excess_free_energy(cs, eta) + excess_free_energy(cs, eta - h)) / (h * h);
  double analytic = d2_excess_free_energy(cs, eta);

  CHECK(analytic == Catch::Approx(numerical).margin(1e-4));
}

TEST_CASE("d3_excess_free_energy is finite", "[hard_spheres]") {
  HardSphereModel cs = CarnahanStarling{};
  CHECK(std::isfinite(d3_excess_free_energy(cs, 0.3)));
}

// Pressure

TEST_CASE("pressure is 1 at eta=0 (ideal gas limit)", "[hard_spheres]") {
  HardSphereModel cs = CarnahanStarling{};
  CHECK(pressure(cs, 0.0) == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("CS pressure at eta=0.3 matches known value", "[hard_spheres]") {
  // P = 1 + eta f'(eta)
  // CS: f' = (8 - 2eta)/(1-eta)^3 at eta = 0.3
  HardSphereModel cs = CarnahanStarling{};
  double eta = 0.3;
  double e = 1.0 - eta;
  double df = (8.0 * eta - 2.0 * eta * eta) / (e * e * e);
  // Exact: f' = d/deta [eta(4-3eta)/(1-eta)^2]
  // = [4 - 6eta] (1-eta)^2 + 2 eta(4-3eta)(1-eta) / (1-eta)^4
  // = [(4-6eta)(1-eta) + 2eta(4-3eta)] / (1-eta)^3
  double fp = ((4.0 - 6.0 * eta) * (1.0 - eta) + 2.0 * eta * (4.0 - 3.0 * eta)) / (e * e * e);
  double expected = 1.0 + eta * fp;

  CHECK(pressure(cs, eta) == Catch::Approx(expected).margin(1e-10));
}

// Free energy and chemical potential

TEST_CASE("free_energy at low density approaches ideal gas", "[hard_spheres]") {
  HardSphereModel cs = CarnahanStarling{};
  double rho = 1e-6;

  double f = free_energy(cs, rho);
  double f_ideal = std::log(rho) - 1.0;

  CHECK(f == Catch::Approx(f_ideal).margin(1e-4));
}

TEST_CASE("chemical_potential exceeds free_energy", "[hard_spheres]") {
  HardSphereModel cs = CarnahanStarling{};
  double rho = 0.5;

  CHECK(chemical_potential(cs, rho) > free_energy(cs, rho));
}

// Model name

TEST_CASE("name returns correct model name", "[hard_spheres]") {
  CHECK(name(HardSphereModel{CarnahanStarling{}}) == "CarnahanStarling");
  CHECK(name(HardSphereModel{PercusYevickVirial{}}) == "PercusYevickVirial");
  CHECK(name(HardSphereModel{PercusYevickCompressibility{}}) == "PercusYevickCompressibility");
}

// Transport coefficients

TEST_CASE("transport bulk viscosity is positive", "[hard_spheres][transport]") {
  CHECK(transport::bulk_viscosity(0.5, 2.0) > 0.0);
}

TEST_CASE("transport shear viscosity is positive", "[hard_spheres][transport]") {
  CHECK(transport::shear_viscosity(0.5, 2.0) > 0.0);
}

TEST_CASE("transport thermal conductivity is positive", "[hard_spheres][transport]") {
  CHECK(transport::thermal_conductivity(0.5, 2.0) > 0.0);
}

TEST_CASE("transport sound damping is positive", "[hard_spheres][transport]") {
  CHECK(transport::sound_damping(0.5, 2.0) > 0.0);
}

// Cross-model: all three agree at very low eta

TEST_CASE("all hard-sphere models agree at low eta", "[hard_spheres]") {
  double eta = 0.01;
  double cs = CarnahanStarling::excess_free_energy(eta);
  double pyv = PercusYevickVirial::excess_free_energy(eta);
  double pyc = PercusYevickCompressibility::excess_free_energy(eta);

  CHECK(cs == Catch::Approx(pyv).margin(1e-3));
  CHECK(cs == Catch::Approx(pyc).margin(1e-3));
}
