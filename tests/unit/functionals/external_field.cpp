#include "dft/functionals/external_field.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;
using namespace dft::functionals;

static auto make_state_with_field(double rho_val, double vext_val, long n, double kT) -> State {
  arma::vec rho(n, arma::fill::value(rho_val));
  arma::vec vext(n, arma::fill::value(vext_val));
  return State{
    .species = { SpeciesState{
        .density = Density{ .values = rho, .external_field = vext },
        .force = arma::vec(n, arma::fill::zeros),
    } },
    .temperature = kT,
  };
}

// External field free energy

TEST_CASE("external field energy for uniform density and field", "[functionals][external_field]") {
  double rho = 0.5;
  double vext = 3.0;
  auto grid = make_grid(0.5, { 1.0, 1.0, 1.0 });
  auto state = make_state_with_field(rho, vext, grid.total_points(), 1.0);

  auto result = external_field(grid, state);

  double dv = grid.cell_volume();
  double expected = rho * vext * dv * grid.total_points();
  CHECK(result.free_energy == Catch::Approx(expected).margin(1e-12));
}

TEST_CASE("external field energy is zero when field is zero", "[functionals][external_field]") {
  auto grid = make_grid(0.5, { 1.0, 1.0, 1.0 });
  auto state = make_state_with_field(0.8, 0.0, grid.total_points(), 1.0);

  auto result = external_field(grid, state);
  CHECK(result.free_energy == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("external field force is V_ext times dV", "[functionals][external_field]") {
  double vext = 2.5;
  auto grid = make_grid(0.5, { 1.0, 1.0, 1.0 });
  auto state = make_state_with_field(0.5, vext, grid.total_points(), 1.0);

  auto result = external_field(grid, state);
  REQUIRE(result.forces.size() == 1);
  CHECK(result.forces[0](0) == Catch::Approx(vext * grid.cell_volume()).margin(1e-14));
}

TEST_CASE("external field handles non-uniform field", "[functionals][external_field]") {
  auto grid = make_grid(0.5, { 1.0, 1.0, 1.0 });
  long n = grid.total_points();
  arma::vec rho(n, arma::fill::ones);
  arma::vec vext = arma::linspace(0.0, 1.0, n);
  arma::vec force(n, arma::fill::zeros);

  State state{
    .species = { SpeciesState{
        .density = { .values = rho, .external_field = vext },
        .force = force,
    } },
    .temperature = 1.0,
  };

  auto result = external_field(grid, state);
  double dv = grid.cell_volume();
  double expected = arma::dot(rho, vext) * dv;
  CHECK(result.free_energy == Catch::Approx(expected).margin(1e-12));
}

TEST_CASE("external field handles multiple species", "[functionals][external_field]") {
  auto grid = make_grid(0.5, { 1.0, 1.0, 1.0 });
  long n = grid.total_points();
  arma::vec rho1(n, arma::fill::value(0.3));
  arma::vec rho2(n, arma::fill::value(0.6));
  arma::vec v1(n, arma::fill::value(1.0));
  arma::vec v2(n, arma::fill::value(2.0));
  arma::vec f(n, arma::fill::zeros);

  State state{
      .species =
          {
              SpeciesState{.density = {.values = rho1, .external_field = v1}, .force = f},
              SpeciesState{.density = {.values = rho2, .external_field = v2}, .force = f},
          },
      .temperature = 1.0,
  };

  auto result = external_field(grid, state);
  REQUIRE(result.forces.size() == 2);

  double dv = grid.cell_volume();
  double expected = (0.3 * 1.0 + 0.6 * 2.0) * dv * n;
  CHECK(result.free_energy == Catch::Approx(expected).margin(1e-12));
}
