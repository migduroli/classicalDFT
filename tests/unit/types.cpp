#include "dft/types.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fstream>

using namespace dft;

// --- Density ---

TEST_CASE("density default-constructs with empty vectors", "[density]") {
  Density d;
  CHECK(d.values.is_empty());
  CHECK(d.external_field.is_empty());
}

TEST_CASE("density values are directly assignable", "[density]") {
  Density d;
  d.values = arma::vec{1.0, 2.0, 3.0};
  CHECK(d.values.n_elem == 3);
  CHECK(d.values(0) == 1.0);
  CHECK(d.values(2) == 3.0);
}

TEST_CASE("density supports designated initializer construction", "[density]") {
  Density d{
      .values = arma::vec(100, arma::fill::ones),
      .external_field = arma::vec(100, arma::fill::zeros),
  };
  CHECK(d.values.n_elem == 100);
  CHECK(d.external_field.n_elem == 100);
}

TEST_CASE("density is copyable", "[density]") {
  Density original{
      .values = arma::vec{1.0, 2.0, 3.0},
      .external_field = arma::vec{0.1, 0.2, 0.3},
  };
  Density copy = original;
  CHECK(copy.values.n_elem == 3);
  CHECK(copy.values(1) == 2.0);

  // Modifying copy does not affect original
  copy.values(0) = 99.0;
  CHECK(original.values(0) == 1.0);
}

TEST_CASE("density is movable", "[density]") {
  Density original{
      .values = arma::vec{1.0, 2.0, 3.0},
      .external_field = arma::vec(3, arma::fill::zeros),
  };
  Density moved = std::move(original);
  CHECK(moved.values.n_elem == 3);
  CHECK(moved.values(2) == 3.0);
}

// --- Species ---

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

// --- State ---

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

// --- Lattice ---

TEST_CASE("build_lattice BCC 001 single cell has 2 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_001);
  CHECK(lattice.positions.n_rows == 2);
  CHECK(lattice.positions.n_cols == 3);
  CHECK(lattice.shape == std::vector<long>{1, 1, 1});
}

TEST_CASE("build_lattice FCC 001 single cell has 4 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_001);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice HCP 001 single cell has 4 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::HCP, Orientation::_001);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice replicates unit cells", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_001, {2, 2, 2});
  CHECK(lattice.positions.n_rows == 2 * 8);
}

TEST_CASE("build_lattice BCC nearest neighbor distance is 1", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_001, {3, 3, 3});
  double min_dist = arma::datum::inf;
  for (arma::uword i = 0; i < lattice.positions.n_rows; ++i) {
    for (arma::uword j = i + 1; j < lattice.positions.n_rows; ++j) {
      double dist = arma::norm(lattice.positions.row(i) - lattice.positions.row(j));
      if (dist < min_dist)
        min_dist = dist;
    }
  }
  CHECK(min_dist == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("build_lattice FCC nearest neighbor distance is 1", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_001, {3, 3, 3});
  double min_dist = arma::datum::inf;
  for (arma::uword i = 0; i < lattice.positions.n_rows; ++i) {
    for (arma::uword j = i + 1; j < lattice.positions.n_rows; ++j) {
      double dist = arma::norm(lattice.positions.row(i) - lattice.positions.row(j));
      if (dist < min_dist)
        min_dist = dist;
    }
  }
  CHECK(min_dist == Catch::Approx(1.0).margin(1e-10));
}

TEST_CASE("build_lattice dimensions scale with shape", "[lattice]") {
  auto single = build_lattice(Structure::BCC, Orientation::_001, {1, 1, 1});
  auto doubled = build_lattice(Structure::BCC, Orientation::_001, {2, 3, 4});
  CHECK(doubled.dimensions(0) == Catch::Approx(2.0 * single.dimensions(0)));
  CHECK(doubled.dimensions(1) == Catch::Approx(3.0 * single.dimensions(1)));
  CHECK(doubled.dimensions(2) == Catch::Approx(4.0 * single.dimensions(2)));
}

TEST_CASE("build_lattice rejects invalid shape", "[lattice]") {
  REQUIRE_THROWS_AS(build_lattice(Structure::BCC, Orientation::_001, {1, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(build_lattice(Structure::BCC, Orientation::_001, {0, 1, 1}), std::invalid_argument);
}

TEST_CASE("build_lattice HCP rejects unsupported orientations", "[lattice]") {
  REQUIRE_THROWS_AS(build_lattice(Structure::HCP, Orientation::_110), std::invalid_argument);
  REQUIRE_THROWS_AS(build_lattice(Structure::HCP, Orientation::_111), std::invalid_argument);
}

TEST_CASE("scaled_positions by dnn multiplies uniformly", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_001);
  auto scaled = lattice.scaled_positions(2.5);
  CHECK(scaled(0, 0) == Catch::Approx(lattice.positions(0, 0) * 2.5));
  CHECK(scaled(1, 2) == Catch::Approx(lattice.positions(1, 2) * 2.5));
}

TEST_CASE("scaled_positions by box rescales anisotropically", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_001, {2, 2, 2});
  arma::rowvec3 box = {10.0, 20.0, 30.0};
  auto scaled = lattice.scaled_positions(box);
  CHECK(arma::max(scaled.col(0)) < 10.0 + 1e-10);
  CHECK(arma::max(scaled.col(1)) < 20.0 + 1e-10);
  CHECK(arma::max(scaled.col(2)) < 30.0 + 1e-10);
}

TEST_CASE("export_lattice writes XYZ format", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_001);
  std::string path = "/tmp/test_lattice.xyz";
  lattice.export_to(path, ExportFormat::XYZ);

  std::ifstream file(path);
  REQUIRE(file.is_open());
  int n_atoms = 0;
  file >> n_atoms;
  CHECK(n_atoms == 2);
}

TEST_CASE("export_lattice writes CSV format", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_001);
  std::string path = "/tmp/test_lattice.csv";
  lattice.export_to(path, ExportFormat::CSV);

  std::ifstream file(path);
  REQUIRE(file.is_open());
  std::string header;
  std::getline(file, header);
  CHECK(header == "x,y,z");
}

// BCC orientations

TEST_CASE("build_lattice BCC 010 has same atom count as 001", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_010);
  CHECK(lattice.positions.n_rows == 2);
}

TEST_CASE("build_lattice BCC 100 has same atom count as 001", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_100);
  CHECK(lattice.positions.n_rows == 2);
}

TEST_CASE("build_lattice BCC 110 single cell has 4 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_110);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice BCC 101 single cell has 4 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_101);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice BCC 011 single cell has 4 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_011);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice BCC 111 single cell has 12 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_111);
  CHECK(lattice.positions.n_rows == 12);
}

// FCC orientations

TEST_CASE("build_lattice FCC 010 has same atom count as 001", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_010);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice FCC 100 has same atom count as 001", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_100);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice FCC 110 single cell has 2 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_110);
  CHECK(lattice.positions.n_rows == 2);
}

TEST_CASE("build_lattice FCC 101 single cell has 2 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_101);
  CHECK(lattice.positions.n_rows == 2);
}

TEST_CASE("build_lattice FCC 011 single cell has 2 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_011);
  CHECK(lattice.positions.n_rows == 2);
}

TEST_CASE("build_lattice FCC 111 single cell has 6 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_111);
  CHECK(lattice.positions.n_rows == 6);
}

// HCP orientations

TEST_CASE("build_lattice HCP 010 single cell has 4 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::HCP, Orientation::_010);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice HCP 100 single cell has 4 atoms", "[lattice]") {
  auto lattice = build_lattice(Structure::HCP, Orientation::_100);
  CHECK(lattice.positions.n_rows == 4);
}

TEST_CASE("build_lattice HCP 010 dimensions are swapped relative to 001", "[lattice]") {
  auto hcp001 = build_lattice(Structure::HCP, Orientation::_001);
  auto hcp010 = build_lattice(Structure::HCP, Orientation::_010);
  // 010 swaps y and z dimensions
  CHECK(hcp010.dimensions(0) == Catch::Approx(hcp001.dimensions(0)).margin(1e-10));
  CHECK(hcp010.dimensions(1) == Catch::Approx(hcp001.dimensions(2)).margin(1e-10));
  CHECK(hcp010.dimensions(2) == Catch::Approx(hcp001.dimensions(1)).margin(1e-10));
}

TEST_CASE("build_lattice HCP 100 dimensions are swapped relative to 001", "[lattice]") {
  auto hcp001 = build_lattice(Structure::HCP, Orientation::_001);
  auto hcp100 = build_lattice(Structure::HCP, Orientation::_100);
  // 100 swaps x and z dimensions
  CHECK(hcp100.dimensions(0) == Catch::Approx(hcp001.dimensions(2)).margin(1e-10));
  CHECK(hcp100.dimensions(1) == Catch::Approx(hcp001.dimensions(1)).margin(1e-10));
  CHECK(hcp100.dimensions(2) == Catch::Approx(hcp001.dimensions(0)).margin(1e-10));
}
