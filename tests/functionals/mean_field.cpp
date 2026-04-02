#include "dft/functionals/mean_field.hpp"

#include "dft/physics/interactions.hpp"
#include "dft/physics/potentials.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

using namespace dft::functionals;
using namespace dft::physics;
using namespace dft::physics::potentials;
using dft::Density;
using dft::Grid;
using dft::Species;
using dft::SpeciesState;
using dft::State;

static constexpr double DX = 0.1;
static constexpr double KT = 1.0;
static constexpr double SIGMA = 1.0;
static constexpr double EPSILON = 1.0;
static constexpr double R_CUTOFF = 2.5;
static const Grid GRID = Grid{.dx = DX, .box_size = {1.6, 1.6, 1.6}, .shape = {16, 16, 16}};
static constexpr long N = 16 * 16 * 16;

static auto make_lj() -> Potential {
  return make_lennard_jones(SIGMA, EPSILON, R_CUTOFF);
}

static auto uniform_state(double rho0, int n_species = 1) -> State {
  State state;
  state.temperature = KT;
  for (int s = 0; s < n_species; ++s) {
    arma::vec rho(N, arma::fill::value(rho0));
    state.species.push_back(SpeciesState{
        .density = Density{.values = rho, .external_field = arma::zeros(N)},
        .force = arma::zeros(N),
    });
  }
  return state;
}

// Weight generation

TEST_CASE("make_mean_field_weights generates one weight per interaction", "[mean_field]") {
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = make_lj()},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);
  CHECK(w.interactions.size() == 1);
}

TEST_CASE("make_mean_field_weights handles multiple interactions", "[mean_field]") {
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = make_lj()},
      {.species_i = 0, .species_j = 1, .potential = make_lj()},
      {.species_i = 1, .species_j = 1, .potential = make_lj()},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);
  CHECK(w.interactions.size() == 3);
}

TEST_CASE("a_vdw is negative for attractive LJ tail", "[mean_field]") {
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = make_lj()},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);
  CHECK(w.interactions[0].a_vdw < 0.0);
}

TEST_CASE("a_vdw matches 2 * vdw_integral when potential fits the grid", "[mean_field]") {
  // Use a small sigma so the attractive tail (r_min to r_cutoff) fits
  // entirely within the grid half-box (0.8).
  auto pot = make_lennard_jones(0.5, 1.0, 0.75);
  auto split = SplitScheme::WeeksChandlerAndersen;

  // vdw_integral = (2pi/kT) int r^2 u_att(r) dr
  // a_vdw = int (u_att(r)/kT) d^3r = 2 * vdw_integral
  double a_continuous = 2.0 * vdw_integral(pot, KT, split);

  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot, .split = split},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);

  CHECK(w.interactions[0].a_vdw == Catch::Approx(a_continuous).epsilon(0.05));
}

// Uniform density: free energy

TEST_CASE("mean_field free energy is (1/2) a_vdw rho^2 V for uniform density", "[mean_field]") {
  double rho0 = 0.5;
  auto pot = make_lj();
  std::vector<Species> species = {{.name = "A", .hard_sphere_diameter = 1.0}};
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);
  auto state = uniform_state(rho0);

  auto result = mean_field(GRID, state, species, w);

  double volume = GRID.box_size[0] * GRID.box_size[1] * GRID.box_size[2];
  double expected = 0.5 * w.interactions[0].a_vdw * rho0 * rho0 * volume;
  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-6));
}

TEST_CASE("mean_field free energy scales with rho squared", "[mean_field]") {
  auto pot = make_lj();
  std::vector<Species> species = {{.name = "A", .hard_sphere_diameter = 1.0}};
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);

  auto r1 = mean_field(GRID, uniform_state(0.3), species, w);
  auto r2 = mean_field(GRID, uniform_state(0.6), species, w);

  // F ~ rho^2, so doubling rho should quadruple F
  double ratio = r2.free_energy / r1.free_energy;
  CHECK(ratio == Catch::Approx(4.0).epsilon(1e-6));
}

// Uniform density: forces are spatially uniform

TEST_CASE("mean_field forces are uniform for uniform density", "[mean_field]") {
  double rho0 = 0.4;
  auto pot = make_lj();
  std::vector<Species> species = {{.name = "A", .hard_sphere_diameter = 1.0}};
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);
  auto state = uniform_state(rho0);

  auto result = mean_field(GRID, state, species, w);

  REQUIRE(result.forces.size() == 1);
  double mean_force = arma::mean(result.forces[0]);
  double max_dev = arma::max(arma::abs(result.forces[0] - mean_force));
  CHECK(max_dev == Catch::Approx(0.0).margin(1e-8));
}

// Uniform density: force equals a_vdw * rho * dV

TEST_CASE("uniform force equals a_vdw * rho * dV for self-interaction", "[mean_field]") {
  double rho0 = 0.4;
  auto pot = make_lj();
  std::vector<Species> species = {{.name = "A", .hard_sphere_diameter = 1.0}};
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);
  auto state = uniform_state(rho0);

  auto result = mean_field(GRID, state, species, w);

  double dv = GRID.cell_volume();
  double expected_force = w.interactions[0].a_vdw * rho0 * dv;
  CHECK(result.forces[0](0) == Catch::Approx(expected_force).epsilon(1e-6));
}

// Zero density

TEST_CASE("mean_field returns zero for zero density", "[mean_field]") {
  auto pot = make_lj();
  std::vector<Species> species = {{.name = "A", .hard_sphere_diameter = 1.0}};
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);
  auto state = uniform_state(0.0);

  auto result = mean_field(GRID, state, species, w);

  CHECK(result.free_energy == Catch::Approx(0.0).margin(1e-14));
  CHECK(arma::max(arma::abs(result.forces[0])) == Catch::Approx(0.0).margin(1e-14));
}

// Cross-interaction: binary mixture

TEST_CASE("mean_field handles cross-interaction in binary mixture", "[mean_field]") {
  double rho_a = 0.3;
  double rho_b = 0.2;
  auto pot = make_lj();
  std::vector<Species> species = {
      {.name = "A", .hard_sphere_diameter = 1.0},
      {.name = "B", .hard_sphere_diameter = 0.8},
  };
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 1, .potential = pot},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);

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
      .temperature = KT,
  };

  auto result = mean_field(GRID, state, species, w);

  // F = (1/2) a_vdw rho_a rho_b V
  double volume = GRID.box_size[0] * GRID.box_size[1] * GRID.box_size[2];
  double expected = 0.5 * w.interactions[0].a_vdw * rho_a * rho_b * volume;
  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-6));

  // Forces should be uniform
  double max_dev_a = arma::max(arma::abs(result.forces[0] - arma::mean(result.forces[0])));
  double max_dev_b = arma::max(arma::abs(result.forces[1] - arma::mean(result.forces[1])));
  CHECK(max_dev_a == Catch::Approx(0.0).margin(1e-8));
  CHECK(max_dev_b == Catch::Approx(0.0).margin(1e-8));

  // dF/d rho_a = (1/2) a_vdw rho_b dV
  double dv = GRID.cell_volume();
  CHECK(result.forces[0](0) == Catch::Approx(0.5 * w.interactions[0].a_vdw * rho_b * dv).epsilon(1e-6));
  CHECK(result.forces[1](0) == Catch::Approx(0.5 * w.interactions[0].a_vdw * rho_a * dv).epsilon(1e-6));
}

// Different weight schemes produce different weights

TEST_CASE("InterpolationZero produces different weight than Linear", "[mean_field]") {
  auto pot = make_lj();
  std::vector<Interaction> inter_zero = {
      {.species_i = 0, .species_j = 0, .potential = pot, .weight_scheme = WeightScheme::InterpolationZero},
  };
  std::vector<Interaction> inter_linear = {
      {.species_i = 0, .species_j = 0, .potential = pot, .weight_scheme = WeightScheme::InterpolationLinearF},
  };

  auto w_zero = make_mean_field_weights(GRID, inter_zero, KT);
  auto w_linear = make_mean_field_weights(GRID, inter_linear, KT);

  // Both should be negative, but different values
  CHECK(w_zero.interactions[0].a_vdw < 0.0);
  CHECK(w_linear.interactions[0].a_vdw < 0.0);
  CHECK(w_zero.interactions[0].a_vdw != Catch::Approx(w_linear.interactions[0].a_vdw).margin(1e-10));
}

// Weights are reusable

TEST_CASE("mean_field weights are reusable across evaluations", "[mean_field]") {
  auto pot = make_lj();
  std::vector<Species> species = {{.name = "A", .hard_sphere_diameter = 1.0}};
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);

  auto r1 = mean_field(GRID, uniform_state(0.4), species, w);
  auto r2 = mean_field(GRID, uniform_state(0.4), species, w);

  CHECK(r1.free_energy == Catch::Approx(r2.free_energy).margin(1e-14));
}

// Force has correct size

TEST_CASE("mean_field returns one force vector per species", "[mean_field]") {
  auto pot = make_lj();
  std::vector<Species> species = {
      {.name = "A", .hard_sphere_diameter = 1.0},
      {.name = "B", .hard_sphere_diameter = 0.8},
  };
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 1, .potential = pot},
  };
  auto w = make_mean_field_weights(GRID, interactions, KT);
  auto state = uniform_state(0.3, 2);

  auto result = mean_field(GRID, state, species, w);

  REQUIRE(result.forces.size() == 2);
  CHECK(result.forces[0].n_elem == static_cast<arma::uword>(N));
  CHECK(result.forces[1].n_elem == static_cast<arma::uword>(N));
}
