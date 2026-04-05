#include "dft/physics/interactions.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace dft::physics;
using namespace dft::physics::potentials;

// Interaction struct construction

TEST_CASE("Interaction can be constructed with designated initialisers", "[interactions]") {
  auto lj = make_lennard_jones(1.0, 1.0, 2.5);
  Interaction inter{
      .species_i = 0,
      .species_j = 1,
      .potential = Potential{lj},
      .split = potentials::SplitScheme::BarkerHenderson,
      .weight_scheme = WeightScheme::GaussE,
      .gauss_order = 10,
  };

  CHECK(inter.species_i == 0);
  CHECK(inter.species_j == 1);
  CHECK(inter.gauss_order == 10);
  CHECK(inter.weight_scheme == WeightScheme::GaussE);
  CHECK(inter.split == potentials::SplitScheme::BarkerHenderson);
}

TEST_CASE("Interaction defaults are sensible", "[interactions]") {
  Interaction inter{
      .species_i = 0,
      .species_j = 0,
      .potential = Potential{make_lennard_jones(1.0, 1.0, 2.5)},
  };

  CHECK(inter.split == potentials::SplitScheme::WeeksChandlerAndersen);
  CHECK(inter.weight_scheme == WeightScheme::InterpolationQuadraticF);
  CHECK(inter.gauss_order == 5);
}

TEST_CASE("Interaction stores variant potential correctly", "[interactions]") {
  auto twf = make_ten_wolde_frenkel(1.0, 1.0, 2.5, 50.0);
  Interaction inter{
      .species_i = 0,
      .species_j = 0,
      .potential = Potential{twf},
  };

  CHECK(inter.potential.name() == "TenWoldeFrenkel");
}
