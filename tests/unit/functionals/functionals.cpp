#include "dft/functionals/functionals.hpp"

#include "dft/functionals/external_field.hpp"
#include "dft/functionals/hard_sphere.hpp"
#include "dft/functionals/ideal_gas.hpp"
#include "dft/functionals/mean_field.hpp"
#include "dft/physics/interactions.hpp"
#include "dft/physics/potentials.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

using namespace dft::functionals;
using namespace dft::functionals::fmt;
using namespace dft::physics;
using namespace dft::physics::potentials;
using dft::Density;
using dft::Grid;
using dft::Species;
using dft::SpeciesState;
using dft::State;

static constexpr double DX = 0.1;
static constexpr double DIAMETER = 1.0;
static constexpr double KT = 1.0;
static const Grid GRID = Grid{.dx = DX, .box_size = {1.6, 1.6, 1.6}, .shape = {16, 16, 16}};
static constexpr long N = 16 * 16 * 16;
static const std::vector<Species> SPECIES = {{.name = "HS", .hard_sphere_diameter = DIAMETER}};

static auto uniform_state(double rho0, double mu = 0.0) -> State {
  arma::vec rho(N, arma::fill::value(rho0));
  return State{
      .species = {SpeciesState{
          .density = Density{.values = rho, .external_field = arma::zeros(N)},
          .force = arma::zeros(N),
          .chemical_potential = mu,
      }},
      .temperature = KT,
  };
}

// make_weights

TEST_CASE("make_weights creates FMT and mean-field weights", "[functionals]") {
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  CHECK(w.fmt.per_species.size() == 1);
  CHECK(w.mean_field.interactions.empty());
}

TEST_CASE("make_weights with interactions generates mean-field weights", "[functionals]") {
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  Model model{.grid = GRID, .species = SPECIES, .interactions = interactions, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  CHECK(w.fmt.per_species.size() == 1);
  CHECK(w.mean_field.interactions.size() == 1);
}

// total: returns correct number of forces

TEST_CASE("total returns one force vector per species", "[functionals]") {
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(0.5);

  auto result = total(model, state, w);
  CHECK(result.forces.size() == 1);
  CHECK(result.forces[0].n_elem == static_cast<arma::uword>(N));
}

// total: free energy is sum of individual contributions

TEST_CASE("total free energy equals sum of individual contributions", "[functionals]") {
  double rho0 = 0.4;
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(rho0);

  auto result = total(model, state, w);

  auto id = ideal_gas(model.grid, state);
  auto ext = external_field(model.grid, state);
  auto hs = hard_sphere(w.fmt_model, model.grid, state, model.species, w.fmt);

  double expected = id.free_energy + ext.free_energy + hs.free_energy;
  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-12));
}

// total: forces are sum of individual contributions

TEST_CASE("total forces equal sum of individual forces", "[functionals]") {
  double rho0 = 0.3;
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(rho0);

  auto result = total(model, state, w);

  auto id = ideal_gas(model.grid, state);
  auto ext = external_field(model.grid, state);
  auto hs = hard_sphere(w.fmt_model, model.grid, state, model.species, w.fmt);

  arma::vec expected = id.forces[0] + ext.forces[0] + hs.forces[0];
  double max_diff = arma::max(arma::abs(result.forces[0] - expected));
  CHECK(max_diff < 1e-12);
}

// total: with mean-field interactions

TEST_CASE("total includes mean-field contribution when interactions present", "[functionals]") {
  double rho0 = 0.3;
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  Model model{.grid = GRID, .species = SPECIES, .interactions = interactions, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(rho0);

  auto result = total(model, state, w);

  auto id = ideal_gas(model.grid, state);
  auto ext = external_field(model.grid, state);
  auto hs = hard_sphere(w.fmt_model, model.grid, state, model.species, w.fmt);
  auto mf = mean_field(model.grid, state, model.species, w.mean_field);

  double expected = id.free_energy + ext.free_energy + hs.free_energy + mf.free_energy;
  CHECK(result.free_energy == Catch::Approx(expected).epsilon(1e-12));
}

TEST_CASE("total forces include mean-field when interactions present", "[functionals]") {
  double rho0 = 0.3;
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };
  Model model{.grid = GRID, .species = SPECIES, .interactions = interactions, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(rho0);

  auto result = total(model, state, w);

  auto id = ideal_gas(model.grid, state);
  auto ext = external_field(model.grid, state);
  auto hs = hard_sphere(w.fmt_model, model.grid, state, model.species, w.fmt);
  auto mf = mean_field(model.grid, state, model.species, w.mean_field);

  arma::vec expected = id.forces[0] + ext.forces[0] + hs.forces[0] + mf.forces[0];
  double max_diff = arma::max(arma::abs(result.forces[0] - expected));
  CHECK(max_diff < 1e-12);
}

// grand potential

TEST_CASE("grand potential equals free energy when mu is zero", "[functionals]") {
  double rho0 = 0.4;
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(rho0, 0.0);

  auto result = total(model, state, w);
  CHECK(result.grand_potential == Catch::Approx(result.free_energy).epsilon(1e-14));
}

TEST_CASE("grand potential equals F minus mu*N", "[functionals]") {
  double rho0 = 0.4;
  double mu = 2.5;
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(rho0, mu);

  auto result = total(model, state, w);

  double dv = GRID.cell_volume();
  double N_total = arma::accu(state.species[0].density.values) * dv;
  double expected = result.free_energy - mu * N_total;
  CHECK(result.grand_potential == Catch::Approx(expected).epsilon(1e-12));
}

// uniform forces are constant across all grid points

TEST_CASE("uniform density produces spatially constant forces", "[functionals]") {
  double rho0 = 0.3;
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(rho0);

  auto result = total(model, state, w);

  double mean = arma::mean(result.forces[0]);
  double max_dev = arma::max(arma::abs(result.forces[0] - mean));
  CHECK(max_dev < 1e-10);
}

// zero density gives zero excess contributions

TEST_CASE("near-zero density gives dominant ideal-gas contribution", "[functionals]") {
  double rho0 = 1e-10;
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);
  auto state = uniform_state(rho0);

  auto result = total(model, state, w);
  auto id = ideal_gas(model.grid, state);

  // At very low density, hard-sphere excess should be negligible
  CHECK(std::abs(result.free_energy - id.free_energy) / std::abs(id.free_energy) < 1e-6);
}

// multi-species

TEST_CASE("total handles two species", "[functionals]") {
  std::vector<Species> species2 = {
      {.name = "A", .hard_sphere_diameter = 1.0},
      {.name = "B", .hard_sphere_diameter = 0.8},
  };
  Model model{.grid = GRID, .species = species2, .temperature = KT};
  auto w = make_weights(Rosenfeld{}, model);

  arma::vec rho_a(N, arma::fill::value(0.3));
  arma::vec rho_b(N, arma::fill::value(0.2));
  State state{
      .species =
          {
              SpeciesState{
                  .density = Density{.values = rho_a, .external_field = arma::zeros(N)},
                  .force = arma::zeros(N),
              },
              SpeciesState{
                  .density = Density{.values = rho_b, .external_field = arma::zeros(N)},
                  .force = arma::zeros(N),
              },
          },
      .temperature = KT,
  };

  auto result = total(model, state, w);
  CHECK(result.forces.size() == 2);
  CHECK(result.forces[0].n_elem == static_cast<arma::uword>(N));
  CHECK(result.forces[1].n_elem == static_cast<arma::uword>(N));
}

// different FMT models

TEST_CASE("total works with all FMT models", "[functionals]") {
  double rho0 = 0.4;
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto state = uniform_state(rho0);

  auto [label, fmt_model] = GENERATE(table<std::string, FMTModel>({
      {"Rosenfeld", Rosenfeld{}},
      {"RSLT", RSLT{}},
      {"WhiteBearI", WhiteBearI{}},
      {"WhiteBearII", WhiteBearII{}},
  }));

  CAPTURE(label);
  auto w = make_weights(fmt_model, model);
  auto result = total(model, state, w);

  CHECK(result.free_energy != 0.0);
  CHECK(result.forces[0].n_elem == static_cast<arma::uword>(N));
}

TEST_CASE("total works with EsFMT model", "[functionals]") {
  double rho0 = 0.4;
  Model model{.grid = GRID, .species = SPECIES, .temperature = KT};
  auto state = uniform_state(rho0);

  auto w = make_weights(EsFMT{.A = 1.0, .B = 0.0}, model);
  auto result = total(model, state, w);

  CHECK(result.free_energy != 0.0);
  CHECK(result.forces[0].n_elem == static_cast<arma::uword>(N));
}

TEST_CASE("make_bulk_weights creates weights with a_vdw", "[functionals]") {
  auto pot = make_lennard_jones(1.0, 1.0, 2.5);
  std::vector<Interaction> interactions = {
      {.species_i = 0, .species_j = 0, .potential = pot},
  };

  auto w = make_bulk_weights(Rosenfeld{}, interactions, KT);
  CHECK(w.fmt.per_species.empty());
  CHECK(w.mean_field.interactions.size() == 1);
  CHECK(w.mean_field.interactions[0].a_vdw != 0.0);
}
