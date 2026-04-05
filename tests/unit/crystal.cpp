#include "dft/types.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;

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
  auto scaled = scaled_positions(lattice, 2.5);
  CHECK(scaled(0, 0) == Catch::Approx(lattice.positions(0, 0) * 2.5));
  CHECK(scaled(1, 2) == Catch::Approx(lattice.positions(1, 2) * 2.5));
}

TEST_CASE("scaled_positions by box rescales anisotropically", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_001, {2, 2, 2});
  arma::rowvec3 box = {10.0, 20.0, 30.0};
  auto scaled = scaled_positions(lattice, box);
  CHECK(arma::max(scaled.col(0)) < 10.0 + 1e-10);
  CHECK(arma::max(scaled.col(1)) < 20.0 + 1e-10);
  CHECK(arma::max(scaled.col(2)) < 30.0 + 1e-10);
}

TEST_CASE("export_lattice writes XYZ format", "[lattice]") {
  auto lattice = build_lattice(Structure::BCC, Orientation::_001);
  std::string path = "/tmp/test_lattice.xyz";
  export_lattice(lattice, path, ExportFormat::XYZ);

  std::ifstream file(path);
  REQUIRE(file.is_open());
  int n_atoms = 0;
  file >> n_atoms;
  CHECK(n_atoms == 2);
}

TEST_CASE("export_lattice writes CSV format", "[lattice]") {
  auto lattice = build_lattice(Structure::FCC, Orientation::_001);
  std::string path = "/tmp/test_lattice.csv";
  export_lattice(lattice, path, ExportFormat::CSV);

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
