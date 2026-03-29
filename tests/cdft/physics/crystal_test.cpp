#include "cdft/physics/crystal.hpp"

#include <gtest/gtest.h>

namespace cdft::physics {

  TEST(LatticeTest, BCCSingleUnit) {
    Lattice lat(CrystalStructure::BCC, Orientation::_001, {1, 1, 1});
    EXPECT_EQ(lat.size(), 2u);
    EXPECT_EQ(lat.shape().size(), 3u);
  }

  TEST(LatticeTest, FCCSingleUnit) {
    Lattice lat(CrystalStructure::FCC, Orientation::_001, {1, 1, 1});
    EXPECT_EQ(lat.size(), 4u);
  }

  TEST(LatticeTest, HCPSingleUnit) {
    Lattice lat(CrystalStructure::HCP, Orientation::_001, {1, 1, 1});
    EXPECT_EQ(lat.size(), 4u);
  }

  TEST(LatticeTest, MultipleUnitCells) {
    Lattice lat(CrystalStructure::BCC, Orientation::_001, {2, 2, 2});
    EXPECT_EQ(lat.size(), 2u * 8u);
  }

  TEST(LatticeTest, ScaledPositions) {
    Lattice lat(CrystalStructure::FCC, Orientation::_001, {1, 1, 1});
    auto pos = lat.positions(2.0);
    EXPECT_EQ(pos.n_rows, 4u);
    EXPECT_EQ(pos.n_cols, 3u);
  }

  TEST(LatticeTest, InvalidShapeThrows) {
    EXPECT_THROW(Lattice(CrystalStructure::BCC, Orientation::_001, {0, 1, 1}), std::invalid_argument);
  }

  TEST(LatticeTest, HCPInvalidOrientation) {
    EXPECT_THROW(Lattice(CrystalStructure::HCP, Orientation::_111, {1, 1, 1}), std::invalid_argument);
  }

}  // namespace cdft::physics
