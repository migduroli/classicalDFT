#include "dft/types.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace dft;

TEST_CASE("state default-constructs with empty species list", "[state]") {
  State s;
  CHECK(s.species.empty());
}

TEST_CASE("state supports designated initializer construction", "[state]") {
  State s{
      .species = {SpeciesState{
          .density =
              Density{.values = arma::vec(8, arma::fill::ones), .external_field = arma::vec(8, arma::fill::zeros)},
          .force = arma::vec(8, arma::fill::zeros),
          .chemical_potential = -1.0,
      }},
      .temperature = 1.5,
  };
  CHECK(s.species.size() == 1);
  CHECK(s.temperature == 1.5);
  CHECK(s.species[0].density.values.n_elem == 8);
  CHECK(s.species[0].chemical_potential == -1.0);
}

TEST_CASE("state with multiple species", "[state]") {
  State s{
      .species =
          {
              SpeciesState{
                  .density = Density{.values = arma::vec(4, arma::fill::ones)},
                  .force = arma::vec(4, arma::fill::zeros),
                  .chemical_potential = -1.0,
              },
              SpeciesState{
                  .density = Density{.values = arma::vec(4, arma::fill::zeros)},
                  .force = arma::vec(4, arma::fill::zeros),
                  .chemical_potential = -2.0,
              },
          },
      .temperature = 0.8,
  };
  CHECK(s.species.size() == 2);
  CHECK(s.species[0].chemical_potential == -1.0);
  CHECK(s.species[1].chemical_potential == -2.0);
}

TEST_CASE("state is movable", "[state]") {
  State original{
      .species = {SpeciesState{
          .density = Density{.values = arma::vec{1.0, 2.0, 3.0}},
          .force = arma::vec(3, arma::fill::zeros),
      }},
      .temperature = 1.0,
  };
  State moved = std::move(original);
  CHECK(moved.species.size() == 1);
  CHECK(moved.species[0].density.values(1) == 2.0);
  CHECK(moved.temperature == 1.0);
}
