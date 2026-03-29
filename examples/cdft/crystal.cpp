// ── Crystal lattice example (modern cdft API) ───────────────────────────────
//
// Demonstrates:
//   - Lattice generation for BCC, FCC, HCP crystal structures
//   - Orientation control and supercell shapes
//   - Position scaling to nearest-neighbour distance
//   - Export of lattice positions

#include <cdft.hpp>

#include <iomanip>
#include <iostream>

int main() {
  using namespace cdft::physics;

  std::cout << std::fixed << std::setprecision(6);

  // ── Single unit cell: BCC, FCC, HCP ─────────────────────────────────────

  std::cout << "Single unit cells (orientation 001)\n";
  std::cout << std::string(60, '-') << "\n";

  for (auto structure : {CrystalStructure::BCC, CrystalStructure::FCC, CrystalStructure::HCP}) {
    auto lattice = Lattice(structure, Orientation::_001);
    auto name = (structure == CrystalStructure::BCC) ? "BCC"
              : (structure == CrystalStructure::FCC) ? "FCC"
                                                     : "HCP";

    std::cout << "\n  " << name << ": " << lattice.size() << " atoms\n";
    std::cout << "  dimensions: [" << lattice.dimensions()(0) << ", "
              << lattice.dimensions()(1) << ", " << lattice.dimensions()(2) << "]\n";

    auto pos = lattice.positions();
    for (arma::uword i = 0; i < pos.n_rows; ++i) {
      std::cout << "    (" << pos(i, 0) << ", " << pos(i, 1) << ", " << pos(i, 2) << ")\n";
    }
  }

  // ── Supercell: 3x3x3 FCC ───────────────────────────────────────────────

  std::cout << "\n\n3x3x3 FCC supercell\n";
  std::cout << std::string(40, '-') << "\n";

  auto fcc_333 = Lattice(CrystalStructure::FCC, Orientation::_001, {3, 3, 3});

  std::cout << "  Atoms: " << fcc_333.size() << "\n";
  std::cout << "  Dimensions: [" << fcc_333.dimensions()(0) << ", "
            << fcc_333.dimensions()(1) << ", " << fcc_333.dimensions()(2) << "]\n";

  // ── Scaled positions (nearest-neighbour distance = 1.0) ─────────────────

  constexpr double nn_distance = 1.5;
  auto scaled_pos = fcc_333.positions(nn_distance);

  std::cout << "\n  Scaled to nearest-neighbour distance = " << nn_distance << "\n";
  std::cout << "  First 5 positions:\n";
  for (arma::uword i = 0; i < std::min<arma::uword>(5, scaled_pos.n_rows); ++i) {
    std::cout << "    (" << scaled_pos(i, 0) << ", " << scaled_pos(i, 1) << ", " << scaled_pos(i, 2) << ")\n";
  }

  // ── Different orientations ──────────────────────────────────────────────

  std::cout << "\n\nBCC unit cells with different orientations\n";
  std::cout << std::string(40, '-') << "\n";

  for (auto orient : {Orientation::_001, Orientation::_110, Orientation::_111}) {
    auto lattice = Lattice(CrystalStructure::BCC, orient);
    auto label = (orient == Orientation::_001) ? "001"
               : (orient == Orientation::_110) ? "110"
                                               : "111";

    std::cout << "  " << label << ": " << lattice.size() << " atoms, dims = ["
              << lattice.dimensions()(0) << ", " << lattice.dimensions()(1) << ", "
              << lattice.dimensions()(2) << "]\n";
  }

  return 0;
}
