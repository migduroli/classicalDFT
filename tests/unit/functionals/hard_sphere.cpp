#include "dft/functionals/hard_sphere.hpp"

#include "dft/functionals/fmt/models.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

using namespace dft::functionals;
using namespace dft::functionals::fmt;
using dft::Density;
using dft::Grid;
using dft::Species;
using dft::SpeciesState;
using dft::State;

static constexpr double DX = 0.1;
static constexpr double DIAMETER = 1.0;
static const Grid GRID = Grid{ .dx = DX, .box_size = { 1.6, 1.6, 1.6 }, .shape = { 16, 16, 16 } };
static constexpr long N = 16 * 16 * 16;

static auto uniform_state(double rho0, double kT = 1.0) -> State {
  arma::vec rho(N, arma::fill::value(rho0));
  return State{
    .species = { SpeciesState{
        .density = Density{ .values = rho, .external_field = arma::zeros(N) },
        .force = arma::zeros(N),
    } },
    .temperature = kT,
  };
}

// Workspace allocation

TEST_CASE("make_fmt_weights generates one WeightSet per species", "[fmt][hard_sphere]") {
  std::vector<Species> species = { { .name = "A", .hard_sphere_diameter = 1.0 } };
  auto w = make_fmt_weights(GRID, species);
  CHECK(w.per_species.size() == 1);
}

TEST_CASE("make_fmt_weights handles multiple species", "[fmt][hard_sphere]") {
  std::vector<Species> species = {
    { .name = "A", .hard_sphere_diameter = 1.0 },
    { .name = "B", .hard_sphere_diameter = 0.8 },
  };
  auto w = make_fmt_weights(GRID, species);
  CHECK(w.per_species.size() == 2);
}

// Uniform density: free energy matches bulk free_energy_density

TEST_CASE("hard_sphere free energy matches bulk for Rosenfeld", "[fmt][hard_sphere]") {
  double rho0 = 0.5;
  FMTModel model = Rosenfeld{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(rho0);

  auto result = hard_sphere(model, GRID, state, species, w);

  double volume = GRID.box_size[0] * GRID.box_size[1] * GRID.box_size[2];
  double phi_bulk = model.free_energy_density(rho0, DIAMETER);
  double expected = phi_bulk * volume;

  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-6));
}

TEST_CASE("hard_sphere free energy matches bulk for WhiteBearII", "[fmt][hard_sphere]") {
  double rho0 = 0.5;
  FMTModel model = WhiteBearII{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(rho0);

  auto result = hard_sphere(model, GRID, state, species, w);

  double volume = GRID.box_size[0] * GRID.box_size[1] * GRID.box_size[2];
  double phi_bulk = model.free_energy_density(rho0, DIAMETER);
  double expected = phi_bulk * volume;

  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-6));
}

TEST_CASE("hard_sphere free energy matches bulk for RSLT", "[fmt][hard_sphere]") {
  double rho0 = 0.5;
  FMTModel model = RSLT{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(rho0);

  auto result = hard_sphere(model, GRID, state, species, w);

  double volume = GRID.box_size[0] * GRID.box_size[1] * GRID.box_size[2];
  double phi_bulk = model.free_energy_density(rho0, DIAMETER);
  double expected = phi_bulk * volume;

  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-6));
}

TEST_CASE("hard_sphere free energy matches bulk for WhiteBearI", "[fmt][hard_sphere]") {
  double rho0 = 0.5;
  FMTModel model = WhiteBearI{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(rho0);

  auto result = hard_sphere(model, GRID, state, species, w);

  double volume = GRID.box_size[0] * GRID.box_size[1] * GRID.box_size[2];
  double phi_bulk = model.free_energy_density(rho0, DIAMETER);
  double expected = phi_bulk * volume;

  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-6));
}

// Uniform density: forces are spatially uniform

TEST_CASE("hard_sphere forces are uniform for uniform density", "[fmt][hard_sphere]") {
  double rho0 = 0.4;
  FMTModel model = Rosenfeld{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(rho0);

  auto result = hard_sphere(model, GRID, state, species, w);

  REQUIRE(result.forces.size() == 1);
  double mean_force = arma::mean(result.forces[0]);
  double max_dev = arma::max(arma::abs(result.forces[0] - mean_force));
  CHECK(max_dev == Catch::Approx(0.0).margin(1e-8));
}

// Uniform density: force equals excess chemical potential * dV

TEST_CASE("uniform force equals excess_chemical_potential * dV for Rosenfeld", "[fmt][hard_sphere]") {
  double rho0 = 0.4;
  FMTModel model = Rosenfeld{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(rho0);

  auto result = hard_sphere(model, GRID, state, species, w);

  double mu_ex = model.excess_chemical_potential(rho0, DIAMETER);
  double dv = GRID.cell_volume();

  CHECK(result.forces[0](0) == Catch::Approx(mu_ex * dv).epsilon(1e-6));
}

TEST_CASE("uniform force equals excess_chemical_potential * dV for WhiteBearII", "[fmt][hard_sphere]") {
  double rho0 = 0.4;
  FMTModel model = WhiteBearII{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(rho0);

  auto result = hard_sphere(model, GRID, state, species, w);

  double mu_ex = model.excess_chemical_potential(rho0, DIAMETER);
  double dv = GRID.cell_volume();

  CHECK(result.forces[0](0) == Catch::Approx(mu_ex * dv).epsilon(1e-6));
}

// Multiple densities: free energy scales correctly

TEST_CASE("hard_sphere free energy increases with density", "[fmt][hard_sphere]") {
  FMTModel model = Rosenfeld{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);

  auto result_low = hard_sphere(model, GRID, uniform_state(0.2), species, w);
  auto result_high = hard_sphere(model, GRID, uniform_state(0.6), species, w);

  CHECK(result_high.free_energy > result_low.free_energy);
}

// Zero density gives zero free energy and zero forces

TEST_CASE("hard_sphere returns zero for zero density", "[fmt][hard_sphere]") {
  FMTModel model = Rosenfeld{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(0.0);

  auto result = hard_sphere(model, GRID, state, species, w);

  CHECK(result.free_energy == Catch::Approx(0.0).margin(1e-14));
  CHECK(arma::max(arma::abs(result.forces[0])) == Catch::Approx(0.0).margin(1e-14));
}

// Forces have correct size

TEST_CASE("hard_sphere returns one force vector per species", "[fmt][hard_sphere]") {
  FMTModel model = Rosenfeld{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(0.5);

  auto result = hard_sphere(model, GRID, state, species, w);

  REQUIRE(result.forces.size() == 1);
  CHECK(result.forces[0].n_elem == static_cast<arma::uword>(N));
}

// Multiple species: binary mixture

TEST_CASE("hard_sphere handles binary mixture", "[fmt][hard_sphere]") {
  double rho_a = 0.3;
  double rho_b = 0.2;
  FMTModel model = Rosenfeld{};
  std::vector<Species> species = {
    { .name = "A", .hard_sphere_diameter = 1.0 },
    { .name = "B", .hard_sphere_diameter = 0.8 },
  };
  auto w = make_fmt_weights(GRID, species);

  arma::vec rho_a_vec(N, arma::fill::value(rho_a));
  arma::vec rho_b_vec(N, arma::fill::value(rho_b));
  State state{
      .species =
          {
              SpeciesState{
                  .density = Density{.values = rho_a_vec, .external_field = arma::zeros(N)},
                  .force = arma::zeros(N),
              },
              SpeciesState{
                  .density = Density{.values = rho_b_vec, .external_field = arma::zeros(N)},
                  .force = arma::zeros(N),
              },
          },
      .temperature = 1.0,
  };

  auto result = hard_sphere(model, GRID, state, species, w);

  REQUIRE(result.forces.size() == 2);
  CHECK(result.free_energy > 0.0);

  // Forces should be uniform for uniform densities
  double max_dev_a = arma::max(arma::abs(result.forces[0] - arma::mean(result.forces[0])));
  double max_dev_b = arma::max(arma::abs(result.forces[1] - arma::mean(result.forces[1])));
  CHECK(max_dev_a == Catch::Approx(0.0).margin(1e-8));
  CHECK(max_dev_b == Catch::Approx(0.0).margin(1e-8));

  // Different species should have different forces
  CHECK(result.forces[0](0) != Catch::Approx(result.forces[1](0)).margin(1e-10));
}

// Reusability: weights can be reused for multiple evaluations

TEST_CASE("weights are reusable across evaluations", "[fmt][hard_sphere]") {
  FMTModel model = Rosenfeld{};
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);

  auto result1 = hard_sphere(model, GRID, uniform_state(0.4), species, w);
  auto result2 = hard_sphere(model, GRID, uniform_state(0.4), species, w);

  CHECK(result1.free_energy == Catch::Approx(result2.free_energy).margin(1e-14));
}

TEST_CASE("hard_sphere free energy matches bulk for EsFMT", "[fmt][hard_sphere]") {
  double rho0 = 0.5;
  FMTModel model = EsFMT{ .A = 1.0, .B = 0.0 };
  std::vector<Species> species = { { .name = "HS", .hard_sphere_diameter = DIAMETER } };
  auto w = make_fmt_weights(GRID, species);
  auto state = uniform_state(rho0);

  auto result = hard_sphere(model, GRID, state, species, w);

  double volume = GRID.box_size[0] * GRID.box_size[1] * GRID.box_size[2];
  double phi_bulk = model.free_energy_density(rho0, DIAMETER);
  double expected = phi_bulk * volume;

  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-6));
}
