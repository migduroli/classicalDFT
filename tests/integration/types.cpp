// Cross-validation of crystal lattice positions against Jim's code.
// Compares atom positions for BCC, FCC, and HCP in all supported orientations.

#include "dft.hpp"
#include "legacy/classicaldft.hpp"

#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <string>

using namespace dft;
using Catch::Approx;

// Convert legacy orientation string to our enum.
static auto orient_enum(const std::string& s) -> Orientation {
  if (s == "001")
    return Orientation::_001;
  if (s == "010")
    return Orientation::_010;
  if (s == "100")
    return Orientation::_100;
  if (s == "110")
    return Orientation::_110;
  if (s == "101")
    return Orientation::_101;
  if (s == "011")
    return Orientation::_011;
  if (s == "111")
    return Orientation::_111;
  throw std::invalid_argument("Unknown orientation: " + s);
}

// Sort positions for deterministic comparison. Lexicographic on (x, y, z).
static auto sorted_positions(const arma::mat& pos) -> arma::mat {
  arma::mat out = pos;
  arma::uvec order = arma::sort_index(out.col(0) * 1e6 + out.col(1) * 1e3 + out.col(2));
  return out.rows(order);
}

static void compare_lattices(Structure structure, const std::string& orient, int nx = 2, int ny = 2, int nz = 2) {
  std::string lattice_name = structure == Structure::BCC ? "BCC" : structure == Structure::FCC ? "FCC" : "HCP";

  // Our code
  auto lattice = build_lattice(structure, orient_enum(orient), {nx, ny, nz});
  auto ours = sorted_positions(lattice.positions);

  // Jim's code
  auto jim_lat = legacy::crystal::build(lattice_name, orient, nx, ny, nz);

  REQUIRE(ours.n_rows == jim_lat.atoms.size());

  // Convert Jim's atoms to arma::mat for sorting.
  arma::mat jim_mat(jim_lat.atoms.size(), 3);
  for (size_t i = 0; i < jim_lat.atoms.size(); ++i) {
    jim_mat(i, 0) = jim_lat.atoms[i][0];
    jim_mat(i, 1) = jim_lat.atoms[i][1];
    jim_mat(i, 2) = jim_lat.atoms[i][2];
  }
  auto jim_sorted = sorted_positions(jim_mat);

  for (arma::uword i = 0; i < ours.n_rows; ++i) {
    INFO("atom " << i << " in " << lattice_name << " " << orient);
    CHECK(ours(i, 0) == Approx(jim_sorted(i, 0)).margin(1e-12));
    CHECK(ours(i, 1) == Approx(jim_sorted(i, 1)).margin(1e-12));
    CHECK(ours(i, 2) == Approx(jim_sorted(i, 2)).margin(1e-12));
  }

  // Box dimensions should also match.
  CHECK(lattice.dimensions(0) == Approx(jim_lat.L[0]).margin(1e-12));
  CHECK(lattice.dimensions(1) == Approx(jim_lat.L[1]).margin(1e-12));
  CHECK(lattice.dimensions(2) == Approx(jim_lat.L[2]).margin(1e-12));
}

// ---- BCC ----

TEST_CASE("BCC 001 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::BCC, "001");
}
TEST_CASE("BCC 010 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::BCC, "010");
}
TEST_CASE("BCC 100 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::BCC, "100");
}
TEST_CASE("BCC 110 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::BCC, "110");
}
TEST_CASE("BCC 101 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::BCC, "101");
}
TEST_CASE("BCC 011 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::BCC, "011");
}
TEST_CASE("BCC 111 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::BCC, "111");
}

// ---- FCC ----

TEST_CASE("FCC 001 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::FCC, "001");
}
TEST_CASE("FCC 010 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::FCC, "010");
}
TEST_CASE("FCC 100 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::FCC, "100");
}
TEST_CASE("FCC 110 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::FCC, "110");
}
TEST_CASE("FCC 101 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::FCC, "101");
}
TEST_CASE("FCC 011 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::FCC, "011");
}
TEST_CASE("FCC 111 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::FCC, "111");
}

// ---- HCP ----

TEST_CASE("HCP 001 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::HCP, "001");
}
TEST_CASE("HCP 010 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::HCP, "010");
}
TEST_CASE("HCP 100 lattice matches legacy", "[integration][crystal]") {
  compare_lattices(Structure::HCP, "100");
}

// ---- Scaled positions ----

TEST_CASE("Scaled BCC positions match legacy", "[integration][crystal]") {
  double dnn = 1.5;
  auto lattice = build_lattice(Structure::BCC, Orientation::_001, {2, 2, 2});
  auto ours = sorted_positions(lattice.scaled_positions(dnn));

  auto jim_lat = legacy::crystal::build("BCC", "001", 2, 2, 2);
  auto jim_scaled = legacy::crystal::scaled(jim_lat, dnn);
  arma::mat jim_mat(jim_scaled.atoms.size(), 3);
  for (size_t i = 0; i < jim_scaled.atoms.size(); ++i) {
    jim_mat(i, 0) = jim_scaled.atoms[i][0];
    jim_mat(i, 1) = jim_scaled.atoms[i][1];
    jim_mat(i, 2) = jim_scaled.atoms[i][2];
  }
  auto jim_sorted = sorted_positions(jim_mat);

  REQUIRE(ours.n_rows == jim_sorted.n_rows);
  for (arma::uword i = 0; i < ours.n_rows; ++i) {
    CHECK(ours(i, 0) == Approx(jim_sorted(i, 0)).margin(1e-12));
    CHECK(ours(i, 1) == Approx(jim_sorted(i, 1)).margin(1e-12));
    CHECK(ours(i, 2) == Approx(jim_sorted(i, 2)).margin(1e-12));
  }
}
