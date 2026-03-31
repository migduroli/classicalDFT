#include "dft/physics/eos.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

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
  EosModel model = make_ideal_gas(1.0);
  CHECK(excess_free_energy(model, 0.5) == 0.0);
}

TEST_CASE("d_excess_free_energy of ideal gas is zero", "[eos]") {
  EosModel model = make_ideal_gas(1.0);
  CHECK(d_excess_free_energy(model, 0.5) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("d2_excess_free_energy of ideal gas is zero", "[eos]") {
  EosModel model = make_ideal_gas(1.0);
  CHECK(d2_excess_free_energy(model, 0.5) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("temperature returns correct value", "[eos]") {
  EosModel model = make_ideal_gas(2.5);
  CHECK(temperature(model) == 2.5);
}

TEST_CASE("name returns correct eos name", "[eos]") {
  CHECK(name(EosModel{make_ideal_gas(1.0)}) == "IdealGas");
  CHECK(name(EosModel{make_percus_yevick(1.0)}) == "PercusYevick");
  CHECK(name(EosModel{make_lennard_jones_jzg(1.0)}) == "LennardJonesJZG");
  CHECK(name(EosModel{make_lennard_jones_mecke(1.0)}) == "LennardJonesMecke");
}

// Derived thermodynamic functions

TEST_CASE("free_energy_per_particle of ideal gas is log(rho)-1", "[eos]") {
  EosModel model = make_ideal_gas(1.0);
  double rho = 0.5;
  CHECK(free_energy_per_particle(model, rho) == Catch::Approx(std::log(rho) - 1.0).margin(1e-14));
}

TEST_CASE("excess_free_energy_density is rho * f_ex", "[eos]") {
  EosModel model = make_percus_yevick(1.0);
  double rho = 0.3;
  double expected = rho * excess_free_energy(model, rho);
  CHECK(excess_free_energy_density(model, rho) == Catch::Approx(expected).margin(1e-14));
}

TEST_CASE("pressure of ideal gas is 1", "[eos]") {
  EosModel model = make_ideal_gas(1.0);
  CHECK(pressure(model, 0.5) == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("pressure of PY at finite density exceeds 1", "[eos]") {
  EosModel model = make_percus_yevick(1.0);
  CHECK(pressure(model, 0.5) > 1.0);
}

// JZG autodiff derivatives

TEST_CASE("JZG derivative matches numerical finite difference", "[eos]") {
  EosModel model = make_lennard_jones_jzg(1.5);
  double rho = 0.3;
  double h = 1e-6;

  double numerical = (excess_free_energy(model, rho + h) - excess_free_energy(model, rho - h)) / (2.0 * h);
  double analytic = d_excess_free_energy(model, rho);

  CHECK(analytic == Catch::Approx(numerical).margin(1e-4));
}

// Mecke autodiff derivatives

TEST_CASE("Mecke derivative matches numerical finite difference", "[eos]") {
  EosModel model = make_lennard_jones_mecke(1.5);
  double rho = 0.3;
  double h = 1e-6;

  double numerical = (excess_free_energy(model, rho + h) - excess_free_energy(model, rho - h)) / (2.0 * h);
  double analytic = d_excess_free_energy(model, rho);

  CHECK(analytic == Catch::Approx(numerical).margin(1e-4));
}

// d_excess_free_energy_density consistency

TEST_CASE("d_excess_free_energy_density matches numerical derivative of density*f_ex", "[eos]") {
  EosModel model = make_percus_yevick(1.0);
  double rho = 0.3;
  double h = 1e-6;

  auto density_f = [&](double r) { return r * excess_free_energy(model, r); };
  double numerical = (density_f(rho + h) - density_f(rho - h)) / (2.0 * h);
  double analytic = d_excess_free_energy_density(model, rho);

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

// d2_excess_free_energy_density consistency

TEST_CASE("d2_excess_free_energy_density matches numerical second derivative", "[eos]") {
  EosModel model = make_percus_yevick(1.0);
  double rho = 0.3;
  double h = 1e-5;

  auto density_f = [&](double r) { return r * excess_free_energy(model, r); };
  double numerical = (density_f(rho + h) - 2.0 * density_f(rho) + density_f(rho - h)) / (h * h);
  double analytic = d2_excess_free_energy_density(model, rho);

  CHECK(analytic == Catch::Approx(numerical).margin(1e-3));
}
