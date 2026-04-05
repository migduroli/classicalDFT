#include "dft/types.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace dft;

TEST_CASE("species stores identity data", "[species]") {
  Species s{.name = "Argon", .hard_sphere_diameter = 3.405};
  CHECK(s.name == "Argon");
  CHECK(s.hard_sphere_diameter == 3.405);
}

TEST_CASE("species supports direct field modification", "[species]") {
  Species s{.name = "Krypton", .hard_sphere_diameter = 1.0};
  s.hard_sphere_diameter = 3.6;
  CHECK(s.hard_sphere_diameter == 3.6);
}

TEST_CASE("species state default-constructs with zero chemical potential", "[species]") {
  SpeciesState ss;
  CHECK(ss.chemical_potential == 0.0);
  CHECK(!ss.fixed_mass.has_value());
  CHECK(ss.density.values.is_empty());
  CHECK(ss.force.is_empty());
}

TEST_CASE("species state supports designated initializer construction", "[species]") {
  SpeciesState ss{
      .density = Density{.values = arma::vec(10, arma::fill::ones), .external_field = arma::vec(10, arma::fill::zeros)},
      .force = arma::vec(10, arma::fill::zeros),
      .chemical_potential = -2.5,
      .fixed_mass = 100.0,
  };
  CHECK(ss.density.values.n_elem == 10);
  CHECK(ss.chemical_potential == -2.5);
  CHECK(ss.fixed_mass.has_value());
  CHECK(ss.fixed_mass.value() == 100.0);
}

TEST_CASE("species state fixed mass is optional", "[species]") {
  SpeciesState ss;
  CHECK(!ss.fixed_mass.has_value());
  ss.fixed_mass = 42.0;
  CHECK(ss.fixed_mass.value() == 42.0);
  ss.fixed_mass = std::nullopt;
  CHECK(!ss.fixed_mass.has_value());
}
