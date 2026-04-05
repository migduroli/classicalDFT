// check.cpp — Cross-validation of crystal lattice positions.
//
// Compares our build_lattice() output against Jim's Crystal_Lattice
// logic for every valid (structure, orientation) combination.
//
// Checks:
//   1. Same number of atoms per unit cell
//   2. Same lattice dimensions (L)
//   3. Atom positions match (sorted, within tolerance)
//   4. Scaled positions match at dnn = 1.3
//   5. Tiled supercell (2x2x2) has correct atom count and dimensions

#include "dft.hpp"
#include "legacy/classicaldft.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace dft;

static int g_failures = 0;
static int g_checks = 0;

static void check_close(const std::string& label, double got, double ref, double tol = 1e-12) {
  ++g_checks;
  double diff = std::abs(got - ref);
  if (diff > tol) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": got=" << got << " ref=" << ref << " diff=" << diff << "\n";
  }
}

static void check_eq(const std::string& label, long got, long ref) {
  ++g_checks;
  if (got != ref) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": got=" << got << " ref=" << ref << "\n";
  }
}

static void section(const std::string& title) {
  std::cout << "\n" << title << "\n";
  std::cout << std::string(title.size(), '-') << "\n";
}

// Sort atoms by (z, y, x) for order-independent comparison.
struct AtomSorter {
  bool operator()(const std::array<double, 3>& a, const std::array<double, 3>& b) const {
    if (std::abs(a[2] - b[2]) > 1e-10)
      return a[2] < b[2];
    if (std::abs(a[1] - b[1]) > 1e-10)
      return a[1] < b[1];
    return a[0] < b[0];
  }
};

// Convert our Lattice positions (arma::mat) to vector<array<double,3>>.
static auto to_atoms(const arma::mat& positions) -> std::vector<std::array<double, 3>> {
  std::vector<std::array<double, 3>> result(positions.n_rows);
  for (arma::uword i = 0; i < positions.n_rows; ++i) {
    result[i] = { positions(i, 0), positions(i, 1), positions(i, 2) };
  }
  return result;
}

// Map our enum to Jim's string convention.
static auto structure_str(Structure s) -> std::string {
  switch (s) {
    case Structure::BCC:
      return "BCC";
    case Structure::FCC:
      return "FCC";
    case Structure::HCP:
      return "HCP";
  }
  return "?";
}

static auto orient_str(Orientation o) -> std::string {
  switch (o) {
    case Orientation::_001:
      return "001";
    case Orientation::_010:
      return "010";
    case Orientation::_100:
      return "100";
    case Orientation::_110:
      return "110";
    case Orientation::_101:
      return "101";
    case Orientation::_011:
      return "011";
    case Orientation::_111:
      return "111";
  }
  return "?";
}

struct TestCase {
  Structure structure;
  Orientation orientation;
};

int main() {
  std::cout << std::setprecision(15);

  std::vector<TestCase> cases = {
    // BCC: all 7 orientations.
    { Structure::BCC, Orientation::_001 },
    { Structure::BCC, Orientation::_010 },
    { Structure::BCC, Orientation::_100 },
    { Structure::BCC, Orientation::_110 },
    { Structure::BCC, Orientation::_101 },
    { Structure::BCC, Orientation::_011 },
    { Structure::BCC, Orientation::_111 },
    // FCC: all 7 orientations.
    { Structure::FCC, Orientation::_001 },
    { Structure::FCC, Orientation::_010 },
    { Structure::FCC, Orientation::_100 },
    { Structure::FCC, Orientation::_110 },
    { Structure::FCC, Orientation::_101 },
    { Structure::FCC, Orientation::_011 },
    { Structure::FCC, Orientation::_111 },
    // HCP: 3 orientations.
    { Structure::HCP, Orientation::_001 },
    { Structure::HCP, Orientation::_010 },
    { Structure::HCP, Orientation::_100 },
  };

  // ------------------------------------------------------------------
  // Step 1: Unit cell comparison (1x1x1)
  // ------------------------------------------------------------------

  for (const auto& tc : cases) {
    auto label = structure_str(tc.structure) + "-" + orient_str(tc.orientation);
    section("Unit cell: " + label);

    auto ours = build_lattice(tc.structure, tc.orientation, { 1, 1, 1 });
    auto jims = legacy::crystal::build(structure_str(tc.structure), orient_str(tc.orientation), 1, 1, 1);

    // Atom count.
    check_eq(label + " atom count", static_cast<long>(ours.positions.n_rows), static_cast<long>(jims.atoms.size()));

    // Dimensions.
    check_close(label + " Lx", ours.dimensions(0), jims.L[0]);
    check_close(label + " Ly", ours.dimensions(1), jims.L[1]);
    check_close(label + " Lz", ours.dimensions(2), jims.L[2]);

    // Sorted positions.
    auto our_atoms = to_atoms(ours.positions);
    auto jim_atoms = jims.atoms;
    std::sort(our_atoms.begin(), our_atoms.end(), AtomSorter{});
    std::sort(jim_atoms.begin(), jim_atoms.end(), AtomSorter{});

    for (size_t i = 0; i < std::min(our_atoms.size(), jim_atoms.size()); ++i) {
      for (int d = 0; d < 3; ++d) {
        check_close(
            label + " atom[" + std::to_string(i) + "][" + std::to_string(d) + "]",
            our_atoms[i][d],
            jim_atoms[i][d]
        );
      }
    }
  }

  // ------------------------------------------------------------------
  // Step 2: Supercell (2x2x2) for a few representative cases
  // ------------------------------------------------------------------

  section("Supercell 2x2x2");

  std::vector<TestCase> super_cases = {
    { Structure::BCC, Orientation::_001 },
    { Structure::FCC, Orientation::_111 },
    { Structure::HCP, Orientation::_001 },
  };

  for (const auto& tc : super_cases) {
    auto label = structure_str(tc.structure) + "-" + orient_str(tc.orientation) + " 2x2x2";

    auto ours = build_lattice(tc.structure, tc.orientation, { 2, 2, 2 });
    auto jims = legacy::crystal::build(structure_str(tc.structure), orient_str(tc.orientation), 2, 2, 2);

    check_eq(label + " atom count", static_cast<long>(ours.positions.n_rows), static_cast<long>(jims.atoms.size()));

    check_close(label + " Lx", ours.dimensions(0), jims.L[0]);
    check_close(label + " Ly", ours.dimensions(1), jims.L[1]);
    check_close(label + " Lz", ours.dimensions(2), jims.L[2]);

    auto our_atoms = to_atoms(ours.positions);
    auto jim_atoms = jims.atoms;
    std::sort(our_atoms.begin(), our_atoms.end(), AtomSorter{});
    std::sort(jim_atoms.begin(), jim_atoms.end(), AtomSorter{});

    for (size_t i = 0; i < std::min(our_atoms.size(), jim_atoms.size()); ++i) {
      for (int d = 0; d < 3; ++d) {
        check_close(
            label + " atom[" + std::to_string(i) + "][" + std::to_string(d) + "]",
            our_atoms[i][d],
            jim_atoms[i][d]
        );
      }
    }
  }

  // ------------------------------------------------------------------
  // Step 3: Scaled positions at dnn = 1.3
  // ------------------------------------------------------------------

  section("Scaled positions (dnn = 1.3)");

  double dnn = 1.3;
  for (const auto& tc : super_cases) {
    auto label = structure_str(tc.structure) + "-" + orient_str(tc.orientation) + " dnn=" + std::to_string(dnn);

    auto ours_lat = build_lattice(tc.structure, tc.orientation, { 1, 1, 1 });
    auto ours_scaled = ours_lat.scaled_positions(dnn);

    auto jims_lat = legacy::crystal::build(structure_str(tc.structure), orient_str(tc.orientation), 1, 1, 1);
    auto jims_scaled = legacy::crystal::scaled(jims_lat, dnn);

    auto our_atoms = to_atoms(ours_scaled);
    auto jim_atoms = jims_scaled.atoms;
    std::sort(our_atoms.begin(), our_atoms.end(), AtomSorter{});
    std::sort(jim_atoms.begin(), jim_atoms.end(), AtomSorter{});

    for (size_t i = 0; i < std::min(our_atoms.size(), jim_atoms.size()); ++i) {
      for (int d = 0; d < 3; ++d) {
        check_close(
            label + " atom[" + std::to_string(i) + "][" + std::to_string(d) + "]",
            our_atoms[i][d],
            jim_atoms[i][d]
        );
      }
    }
  }

  // ------------------------------------------------------------------
  // Summary
  // ------------------------------------------------------------------

  std::cout << "\n========================================\n";
  std::cout << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures > 0 ? 1 : 0;
}
