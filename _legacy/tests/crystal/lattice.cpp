#include "dft/crystal/lattice.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace dft::crystal;

// ── Construction ────────────────────────────────────────────────────────────

TEST(Lattice, InvalidCellCountThrows) {
  EXPECT_THROW(Lattice(Structure::BCC, Orientation::_001, {0, 1, 1}), std::invalid_argument);
  EXPECT_THROW(Lattice(Structure::BCC, Orientation::_001, {1, -1, 1}), std::invalid_argument);
}

TEST(Lattice, HCPInvalidOrientationThrows) {
  EXPECT_THROW(Lattice(Structure::HCP, Orientation::_110), std::invalid_argument);
  EXPECT_THROW(Lattice(Structure::HCP, Orientation::_111), std::invalid_argument);
}

TEST(Lattice, Accessors) {
  auto lattice = Lattice(Structure::FCC, Orientation::_001, {2, 3, 4});
  EXPECT_EQ(lattice.structure(), Structure::FCC);
  EXPECT_EQ(lattice.orientation(), Orientation::_001);
  EXPECT_EQ(lattice.shape()[0], 2);
  EXPECT_EQ(lattice.shape()[1], 3);
  EXPECT_EQ(lattice.shape()[2], 4);
}

// ── BCC ─────────────────────────────────────────────────────────────────────

TEST(Lattice, BCC001SingleCellAtomCount) {
  auto lattice = Lattice(Structure::BCC, Orientation::_001);
  EXPECT_EQ(lattice.size(), 2U);
}

TEST(Lattice, BCC001ReplicatedAtomCount) {
  auto lattice = Lattice(Structure::BCC, Orientation::_001, {2, 2, 2});
  EXPECT_EQ(lattice.size(), 2U * 8);
}

TEST(Lattice, BCC110SingleCellAtomCount) {
  auto lattice = Lattice(Structure::BCC, Orientation::_110);
  EXPECT_EQ(lattice.size(), 4U);
}

TEST(Lattice, BCC111SingleCellAtomCount) {
  auto lattice = Lattice(Structure::BCC, Orientation::_111);
  EXPECT_EQ(lattice.size(), 12U);
}

// ── FCC ─────────────────────────────────────────────────────────────────────

TEST(Lattice, FCC001SingleCellAtomCount) {
  auto lattice = Lattice(Structure::FCC, Orientation::_001);
  EXPECT_EQ(lattice.size(), 4U);
}

TEST(Lattice, FCC110SingleCellAtomCount) {
  auto lattice = Lattice(Structure::FCC, Orientation::_110);
  EXPECT_EQ(lattice.size(), 2U);
}

TEST(Lattice, FCC111SingleCellAtomCount) {
  auto lattice = Lattice(Structure::FCC, Orientation::_111);
  EXPECT_EQ(lattice.size(), 6U);
}

TEST(Lattice, FCC001ReplicatedAtomCount) {
  auto lattice = Lattice(Structure::FCC, Orientation::_001, {3, 3, 3});
  EXPECT_EQ(lattice.size(), 4U * 27);
}

// ── HCP ─────────────────────────────────────────────────────────────────────

TEST(Lattice, HCP001SingleCellAtomCount) {
  auto lattice = Lattice(Structure::HCP, Orientation::_001);
  EXPECT_EQ(lattice.size(), 4U);
}

TEST(Lattice, HCP010SingleCellAtomCount) {
  auto lattice = Lattice(Structure::HCP, Orientation::_010);
  EXPECT_EQ(lattice.size(), 4U);
}

TEST(Lattice, HCP100SingleCellAtomCount) {
  auto lattice = Lattice(Structure::HCP, Orientation::_100);
  EXPECT_EQ(lattice.size(), 4U);
}

// ── Nearest-neighbor distance ───────────────────────────────────────────────

namespace {

  double min_distance(const arma::mat& atoms, const arma::rowvec3& L) {
    double dmin = 1e30;
    for (arma::uword i = 0; i < atoms.n_rows; ++i) {
      for (arma::uword j = i + 1; j < atoms.n_rows; ++j) {
        arma::rowvec3 dr = atoms.row(i) - atoms.row(j);
        dr -= L % arma::round(dr / L);
        dmin = std::min(dmin, arma::dot(dr, dr));
      }
    }
    return std::sqrt(dmin);
  }

}  // namespace

TEST(Lattice, BCCNearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::BCC, Orientation::_001, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, FCCNearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::FCC, Orientation::_001, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, HCPNearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::HCP, Orientation::_001, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

// ── Coordination number ─────────────────────────────────────────────────────

namespace {

  int count_nearest_neighbors(const arma::mat& atoms, const arma::rowvec3& L, double dnn, double tolerance = 0.01) {
    double cutoff2 = (dnn + tolerance) * (dnn + tolerance);
    int count = 0;
    for (arma::uword j = 1; j < atoms.n_rows; ++j) {
      arma::rowvec3 dr = atoms.row(0) - atoms.row(j);
      dr -= L % arma::round(dr / L);
      if (arma::dot(dr, dr) < cutoff2)
        ++count;
    }
    return count;
  }

}  // namespace

TEST(Lattice, BCCCoordinationNumber) {
  auto lattice = Lattice(Structure::BCC, Orientation::_001, {3, 3, 3});
  EXPECT_EQ(count_nearest_neighbors(lattice.positions(), lattice.dimensions(), 1.0), 8);
}

TEST(Lattice, FCCCoordinationNumber) {
  auto lattice = Lattice(Structure::FCC, Orientation::_001, {3, 3, 3});
  EXPECT_EQ(count_nearest_neighbors(lattice.positions(), lattice.dimensions(), 1.0), 12);
}

TEST(Lattice, HCPCoordinationNumber) {
  auto lattice = Lattice(Structure::HCP, Orientation::_001, {3, 3, 3});
  EXPECT_EQ(count_nearest_neighbors(lattice.positions(), lattice.dimensions(), 1.0), 12);
}

// ── positions() overloads ───────────────────────────────────────────────────

TEST(Lattice, PositionsWithDNNScalesUniformly) {
  auto lattice = Lattice(Structure::FCC, Orientation::_001);
  double dnn = 2.5;
  arma::mat scaled = lattice.positions(dnn);
  EXPECT_EQ(scaled.n_rows, lattice.size());
  EXPECT_TRUE(arma::approx_equal(scaled, lattice.positions() * dnn, "absdiff", 1e-14));
}

TEST(Lattice, PositionsWithBoxScalesAnisotropically) {
  auto lattice = Lattice(Structure::BCC, Orientation::_001);
  arma::rowvec3 box = {5.0, 6.0, 7.0};
  arma::mat scaled = lattice.positions(box);
  // Second atom in BCC 001 is at center -> scales to (Lx/2, Ly/2, Lz/2)
  EXPECT_NEAR(scaled(1, 0), box(0) / 2.0, 1e-10);
  EXPECT_NEAR(scaled(1, 1), box(1) / 2.0, 1e-10);
  EXPECT_NEAR(scaled(1, 2), box(2) / 2.0, 1e-10);
}

// ── Replication ─────────────────────────────────────────────────────────────

TEST(Lattice, LatticeScalesWithReplication) {
  auto single = Lattice(Structure::FCC, Orientation::_001, {1, 1, 1});
  auto multi = Lattice(Structure::FCC, Orientation::_001, {2, 3, 4});

  EXPECT_NEAR(multi.dimensions()(0), 2.0 * single.dimensions()(0), 1e-12);
  EXPECT_NEAR(multi.dimensions()(1), 3.0 * single.dimensions()(1), 1e-12);
  EXPECT_NEAR(multi.dimensions()(2), 4.0 * single.dimensions()(2), 1e-12);
}

TEST(Lattice, AllAtomsInsideBox) {
  auto lattice = Lattice(Structure::FCC, Orientation::_111, {2, 2, 2});
  const auto& L = lattice.dimensions();
  const auto& pos = lattice.positions();
  for (arma::uword i = 0; i < pos.n_rows; ++i) {
    for (arma::uword d = 0; d < 3; ++d) {
      EXPECT_GE(pos(i, d), -1e-10);
      EXPECT_LT(pos(i, d), L(d) + 1e-10);
    }
  }
}

// ── Export ───────────────────────────────────────────────────────────────────

TEST(Lattice, ExportXYZWritesFile) {
  auto lattice = Lattice(Structure::BCC, Orientation::_001);
  std::string filename = "test_crystal.xyz";
  lattice.export_to(filename, ExportFormat::XYZ);

  std::ifstream in(filename);
  ASSERT_TRUE(in.good());

  int n;
  in >> n;
  EXPECT_EQ(n, 2);

  std::string comment;
  std::getline(in, comment);  // rest of first line
  std::getline(in, comment);  // comment line
  EXPECT_EQ(comment, "Crystal lattice");

  std::string element;
  double x, y, z;
  in >> element >> x >> y >> z;
  EXPECT_EQ(element, "Ar");
  EXPECT_NEAR(x, lattice.positions()(0, 0), 1e-10);
  EXPECT_NEAR(y, lattice.positions()(0, 1), 1e-10);
  EXPECT_NEAR(z, lattice.positions()(0, 2), 1e-10);

  std::filesystem::remove(filename);
}

TEST(Lattice, ExportXYZDefaultFormat) {
  auto lattice = Lattice(Structure::BCC, Orientation::_001);
  std::string filename = "test_crystal_default.xyz";
  lattice.export_to(filename);  // default = XYZ

  std::ifstream in(filename);
  ASSERT_TRUE(in.good());

  int n;
  in >> n;
  EXPECT_EQ(n, 2);

  std::filesystem::remove(filename);
}

TEST(Lattice, ExportCSVWritesFile) {
  auto lattice = Lattice(Structure::FCC, Orientation::_001);
  std::string filename = "test_crystal.csv";
  lattice.export_to(filename, ExportFormat::CSV);

  std::ifstream in(filename);
  ASSERT_TRUE(in.good());

  std::string header;
  std::getline(in, header);
  EXPECT_EQ(header, "x,y,z");

  // Read first data line
  std::string line;
  std::getline(in, line);
  ASSERT_FALSE(line.empty());

  // Count total data lines (FCC 001 has 4 atoms)
  int count = 1;
  while (std::getline(in, line)) {
    if (!line.empty())
      ++count;
  }
  EXPECT_EQ(count, 4);

  std::filesystem::remove(filename);
}

// ── All BCC orientations produce unit nearest-neighbor distance ─────────────

TEST(Lattice, BCC010NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::BCC, Orientation::_010, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, BCC100NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::BCC, Orientation::_100, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, BCC110NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::BCC, Orientation::_110, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, BCC101NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::BCC, Orientation::_101, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, BCC011NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::BCC, Orientation::_011, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, BCC111NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::BCC, Orientation::_111, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

// ── All FCC orientations produce unit nearest-neighbor distance ─────────────

TEST(Lattice, FCC010NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::FCC, Orientation::_010, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, FCC100NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::FCC, Orientation::_100, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, FCC110NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::FCC, Orientation::_110, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, FCC101NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::FCC, Orientation::_101, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, FCC011NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::FCC, Orientation::_011, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, FCC111NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::FCC, Orientation::_111, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

// ── HCP non-001 orientations ────────────────────────────────────────────────

TEST(Lattice, HCP010NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::HCP, Orientation::_010, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

TEST(Lattice, HCP100NearestNeighborDistanceIsOne) {
  auto lattice = Lattice(Structure::HCP, Orientation::_100, {3, 3, 3});
  EXPECT_NEAR(min_distance(lattice.positions(), lattice.dimensions()), 1.0, 1e-10);
}

// ── Positions(box) for FCC ──────────────────────────────────────────────────

TEST(Lattice, PositionsWithBoxFCCScalesCorrectly) {
  auto lattice = Lattice(Structure::FCC, Orientation::_001);
  arma::rowvec3 box = {10.0, 20.0, 30.0};
  arma::mat scaled = lattice.positions(box);
  // All positions should be within [0, box)
  for (arma::uword i = 0; i < scaled.n_rows; ++i) {
    for (arma::uword d = 0; d < 3; ++d) {
      EXPECT_GE(scaled(i, d), -1e-10);
      EXPECT_LT(scaled(i, d), box(d) + 1e-10);
    }
  }
}

// ── Shape validation ────────────────────────────────────────────────────────

TEST(Lattice, InvalidShapeSizeThrows) {
  EXPECT_THROW(Lattice(Structure::BCC, Orientation::_001, {1, 1}), std::invalid_argument);
  EXPECT_THROW(Lattice(Structure::BCC, Orientation::_001, {1, 1, 1, 1}), std::invalid_argument);
}
