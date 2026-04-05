#include "dft/physics/eos.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::physics::eos;

// Factories

TEST_CASE("make_ideal_gas rejects non-positive temperature", "[eos]") {
  REQUIRE_THROWS_AS(make_ideal_gas(0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(make_ideal_gas(-1.0), std::invalid_argument);
}

TEST_CASE("make_percus_yevick rejects non-positive temperature", "[eos]") {
  REQUIRE_THROWS_AS(make_percus_yevick(0.0), std::invalid_argument);
  REQUIRE_THROWS_AS(make_percus_yevick(-1.0), std::invalid_argument);
}

TEST_CASE("make_lennard_jones_jzg rejects non-positive temperature", "[eos]") {
  REQUIRE_THROWS_AS(make_lennard_jones_jzg(0.0), std::invalid_argument);
}

TEST_CASE("make_lennard_jones_mecke rejects non-positive temperature", "[eos]") {
  REQUIRE_THROWS_AS(make_lennard_jones_mecke(0.0), std::invalid_argument);
}

// IdealGas

TEST_CASE("ideal gas excess free energy is zero", "[eos]") {
  auto ig = make_ideal_gas(1.0);
  CHECK(ig.excess_free_energy(0.5) == 0.0);
  CHECK(ig.excess_free_energy(1.0) == 0.0);
}

// PercusYevick

TEST_CASE("PY eos excess free energy is zero at zero density", "[eos]") {
  auto py = make_percus_yevick(1.0);
  CHECK(py.excess_free_energy(0.0) == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("PY eos excess free energy is positive at finite density", "[eos]") {
  auto py = make_percus_yevick(1.0);
  CHECK(py.excess_free_energy(0.5) > 0.0);
}

// LennardJonesJZG

TEST_CASE("JZG excess free energy is finite at moderate density and temperature", "[eos]") {
  auto jzg = make_lennard_jones_jzg(1.5);
  double f = jzg.excess_free_energy(0.3);
  CHECK(std::isfinite(f));
}

TEST_CASE("JZG with tail correction differs from no-tail", "[eos]") {
  auto jzg_notail = make_lennard_jones_jzg(1.5);
  auto jzg_tail = make_lennard_jones_jzg(1.5, 2.5);
  double rho = 0.3;

  CHECK(jzg_notail.excess_free_energy(rho) != jzg_tail.excess_free_energy(rho));
}

// LennardJonesMecke

TEST_CASE("Mecke excess free energy is finite at moderate density and temperature", "[eos]") {
  auto mecke = make_lennard_jones_mecke(1.5);
  double f = mecke.excess_free_energy(0.3);
  CHECK(std::isfinite(f));
}

// Variant-based functions

TEST_CASE("excess_free_energy dispatches via variant", "[eos]") {
  auto model = make_ideal_gas(1.0);
  CHECK(model.excess_free_energy(0.5) == 0.0);
}

TEST_CASE("d_excess_free_energy of ideal gas is zero", "[eos]") {
  auto model = make_ideal_gas(1.0);
  CHECK(model.d_excess_free_energy(0.5) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("d2_excess_free_energy of ideal gas is zero", "[eos]") {
  auto model = make_ideal_gas(1.0);
  CHECK(model.d2_excess_free_energy(0.5) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("temperature returns correct value", "[eos]") {
  auto model = make_ideal_gas(2.5);
  CHECK(model.kT == 2.5);
}

TEST_CASE("name returns correct eos name", "[eos]") {
  CHECK(make_ideal_gas(1.0).NAME == "IdealGas");
  CHECK(make_percus_yevick(1.0).NAME == "PercusYevick");
  CHECK(make_lennard_jones_jzg(1.0).NAME == "LennardJonesJZG");
  CHECK(make_lennard_jones_mecke(1.0).NAME == "LennardJonesMecke");
}

// Derived thermodynamic functions

TEST_CASE("free_energy of ideal gas is log(rho)-1", "[eos]") {
  auto model = make_ideal_gas(1.0);
  double rho = 0.5;
  CHECK(model.free_energy(rho) == Catch::Approx(std::log(rho) - 1.0).margin(1e-14));
}

TEST_CASE("excess_chemical_potential of ideal gas is zero", "[eos]") {
  auto model = make_ideal_gas(1.0);
  CHECK(model.excess_chemical_potential(0.5) == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("chemical_potential of ideal gas is log(rho)", "[eos]") {
  auto model = make_ideal_gas(1.0);
  double rho = 0.5;
  CHECK(model.chemical_potential(rho) == Catch::Approx(std::log(rho)).margin(1e-14));
}

TEST_CASE("pressure of ideal gas is 1", "[eos]") {
  auto model = make_ideal_gas(1.0);
  CHECK(model.pressure(0.5) == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("pressure of PY at finite density exceeds 1", "[eos]") {
  auto model = make_percus_yevick(1.0);
  CHECK(model.pressure(0.5) > 1.0);
}

// JZG autodiff derivatives

TEST_CASE("JZG derivative matches numerical finite difference", "[eos]") {
  auto model = make_lennard_jones_jzg(1.5);
  double rho = 0.3;
  double h = 1e-6;

  double numerical = (model.excess_free_energy(rho + h) - model.excess_free_energy(rho - h)) / (2.0 * h);
  double analytic = model.d_excess_free_energy(rho);

  CHECK(analytic == Catch::Approx(numerical).margin(1e-4));
}

// Mecke autodiff derivatives

TEST_CASE("Mecke derivative matches numerical finite difference", "[eos]") {
  auto model = make_lennard_jones_mecke(1.5);
  double rho = 0.3;
  double h = 1e-6;

  double numerical = (model.excess_free_energy(rho + h) - model.excess_free_energy(rho - h)) / (2.0 * h);
  double analytic = model.d_excess_free_energy(rho);

  CHECK(analytic == Catch::Approx(numerical).margin(1e-4));
}

// excess_chemical_potential consistency

TEST_CASE("excess_chemical_potential matches numerical derivative of rho*f_ex", "[eos]") {
  auto model = make_percus_yevick(1.0);
  double rho = 0.3;
  double h = 1e-6;

  auto density_f = [&](double r) { return r * model.excess_free_energy(r); };
  double numerical = (density_f(rho + h) - density_f(rho - h)) / (2.0 * h);
  double analytic = model.excess_chemical_potential(rho);

  CHECK(analytic == Catch::Approx(numerical).margin(1e-4));
}

// Mecke with cutoff and shifted flag

TEST_CASE("Mecke with cutoff has nonzero tail correction", "[eos]") {
  auto mecke = make_lennard_jones_mecke(1.5, 2.5);
  CHECK(mecke.tail_correction != 0.0);
}

TEST_CASE("Mecke with shifted cutoff differs from unshifted", "[eos]") {
  auto mecke_unshifted = make_lennard_jones_mecke(1.5, 2.5, false);
  auto mecke_shifted = make_lennard_jones_mecke(1.5, 2.5, true);

  CHECK(mecke_unshifted.tail_correction != mecke_shifted.tail_correction);
}

TEST_CASE("JZG with shifted cutoff differs from unshifted", "[eos]") {
  auto jzg_unshifted = make_lennard_jones_jzg(1.5, 2.5, false);
  auto jzg_shifted = make_lennard_jones_jzg(1.5, 2.5, true);

  CHECK(jzg_unshifted.tail_correction != jzg_shifted.tail_correction);
}

// chemical_potential consistency

TEST_CASE("chemical_potential equals log(rho) + excess_chemical_potential", "[eos]") {
  auto model = make_percus_yevick(1.0);
  double rho = 0.3;

  double mu = model.chemical_potential(rho);
  double expected = std::log(rho) + model.excess_chemical_potential(rho);
  CHECK(mu == Catch::Approx(expected).margin(1e-14));
}
