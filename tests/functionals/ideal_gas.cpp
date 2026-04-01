#include "dft/functionals/ideal_gas.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace dft;
using namespace dft::functionals;

static auto make_single_species_state(double rho_val, long n, double kT, double mu = 0.0) -> State {
  arma::vec rho(n, arma::fill::value(rho_val));
  arma::vec vext(n, arma::fill::zeros);
  return State{
      .species = {SpeciesState{
          .density = Density{.values = rho, .external_field = vext},
          .force = arma::vec(n, arma::fill::zeros),
          .chemical_potential = mu,
      }},
      .temperature = kT,
  };
}

// Ideal gas free energy

TEST_CASE("ideal gas free energy for uniform density", "[functionals][ideal_gas]") {
  double rho = 0.5;
  double kT = 1.0;
  long n = 1000;
  auto grid = make_grid(0.1, {1.0, 1.0, 1.0});
  auto state = make_single_species_state(rho, grid.total_points(), kT);

  auto result = ideal_gas(grid, state);

  double dv = grid.cell_volume();
  double expected = kT * rho * (std::log(rho) - 1.0) * dv * grid.total_points();
  CHECK(result.free_energy == Catch::Approx(expected).margin(1e-10));
}

TEST_CASE("ideal gas returns one force vector per species", "[functionals][ideal_gas]") {
  auto grid = make_grid(0.5, {1.0, 1.0, 1.0});
  auto state = make_single_species_state(0.5, grid.total_points(), 1.0);

  auto result = ideal_gas(grid, state);
  REQUIRE(result.forces.size() == 1);
  CHECK(result.forces[0].n_elem == static_cast<arma::uword>(grid.total_points()));
}

TEST_CASE("ideal gas force is log(rho) dV for zero chemical potential", "[functionals][ideal_gas]") {
  double rho = 0.3;
  auto grid = make_grid(0.5, {1.0, 1.0, 1.0});
  auto state = make_single_species_state(rho, grid.total_points(), 1.0);

  auto result = ideal_gas(grid, state);
  double expected_force = std::log(rho) * grid.cell_volume();
  CHECK(result.forces[0](0) == Catch::Approx(expected_force).margin(1e-14));
}

TEST_CASE("ideal gas force includes chemical potential", "[functionals][ideal_gas]") {
  double rho = 0.3;
  double mu = 2.0;
  double kT = 1.5;
  auto grid = make_grid(0.5, {1.0, 1.0, 1.0});
  auto state = make_single_species_state(rho, grid.total_points(), kT, mu);

  auto result = ideal_gas(grid, state);
  double dv = grid.cell_volume();
  double expected_force = (std::log(rho) - mu / kT) * dv;
  CHECK(result.forces[0](0) == Catch::Approx(expected_force).margin(1e-14));
}

TEST_CASE("ideal gas free energy scales with temperature", "[functionals][ideal_gas]") {
  double rho = 0.5;
  auto grid = make_grid(0.5, {1.0, 1.0, 1.0});

  auto state1 = make_single_species_state(rho, grid.total_points(), 1.0);
  auto state2 = make_single_species_state(rho, grid.total_points(), 2.0);

  auto r1 = ideal_gas(grid, state1);
  auto r2 = ideal_gas(grid, state2);
  CHECK(r2.free_energy == Catch::Approx(2.0 * r1.free_energy).margin(1e-10));
}

TEST_CASE("ideal gas handles multiple species", "[functionals][ideal_gas]") {
  auto grid = make_grid(0.5, {1.0, 1.0, 1.0});
  long n = grid.total_points();
  arma::vec rho1(n, arma::fill::value(0.3));
  arma::vec rho2(n, arma::fill::value(0.7));
  arma::vec vext(n, arma::fill::zeros);
  arma::vec f(n, arma::fill::zeros);

  State state{
      .species =
          {
              SpeciesState{.density = {.values = rho1, .external_field = vext}, .force = f},
              SpeciesState{.density = {.values = rho2, .external_field = vext}, .force = f},
          },
      .temperature = 1.0,
  };

  auto result = ideal_gas(grid, state);
  REQUIRE(result.forces.size() == 2);

  double dv = grid.cell_volume();
  double expected = (0.3 * (std::log(0.3) - 1.0) + 0.7 * (std::log(0.7) - 1.0)) * dv * n;
  CHECK(result.free_energy == Catch::Approx(expected).margin(1e-10));
}
