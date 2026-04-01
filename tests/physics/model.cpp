#include "dft/physics/model.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace dft;
using namespace dft::physics;
using namespace dft::physics::potentials;

// Model construction

TEST_CASE("Model can be constructed with designated initialisers", "[model]") {
  auto grid = make_grid(0.1, {1.0, 1.0, 1.0});
  Model model{
      .grid = grid,
      .species = {Species{.name = "Argon", .hard_sphere_diameter = 1.0}},
      .interactions = {},
      .temperature = 1.5,
  };

  CHECK(model.species.size() == 1);
  CHECK(model.species[0].name == "Argon");
  CHECK(model.temperature == 1.5);
  CHECK(model.interactions.empty());
  CHECK(model.grid.dx == 0.1);
}

TEST_CASE("Model supports multiple species", "[model]") {
  auto grid = make_grid(0.1, {1.0, 1.0, 1.0});
  Model model{
      .grid = grid,
      .species =
          {Species{.name = "Argon", .hard_sphere_diameter = 1.0},
           Species{.name = "Xenon", .hard_sphere_diameter = 1.2}},
      .interactions = {},
      .temperature = 2.0,
  };

  CHECK(model.species.size() == 2);
  CHECK(model.species[1].name == "Xenon");
}

TEST_CASE("Model supports interactions", "[model]") {
  auto grid = make_grid(0.1, {1.0, 1.0, 1.0});
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);

  Model model{
      .grid = grid,
      .species = {Species{.name = "A", .hard_sphere_diameter = 1.0}, Species{.name = "B", .hard_sphere_diameter = 1.0}},
      .interactions = {Interaction{
          .species_i = 0,
          .species_j = 1,
          .potential = Potential{lj},
      }},
      .temperature = 1.0,
  };

  CHECK(model.interactions.size() == 1);
  CHECK(model.interactions[0].species_i == 0);
  CHECK(model.interactions[0].species_j == 1);
}
