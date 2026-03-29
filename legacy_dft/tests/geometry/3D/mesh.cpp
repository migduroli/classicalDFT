#include "classicaldft_bits/geometry/3D/mesh.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace dft::geometry;

// Test wrapper to access protected method
class TestableSUQMesh3D : public three_dimensional::SUQMesh {
 public:
  using three_dimensional::SUQMesh::global_index_to_cartesian;
  using three_dimensional::SUQMesh::SUQMesh;
};

TEST(ThreeDimensionalMesh, ConstructionAndBasicProperties) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{1, 1, 1};
  auto m = three_dimensional::SUQMesh(0.5, dimensions, origin);
  ASSERT_EQ(27, m.number_vertices());
  ASSERT_DOUBLE_EQ(1.0, m.volume());
}

TEST(ThreeDimensionalMesh, ShapeIsCorrect) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = three_dimensional::SUQMesh(1.0, dimensions, origin);
  ASSERT_EQ(m.shape().size(), 3U);
  EXPECT_EQ(m.shape()[0], 3);
  EXPECT_EQ(m.shape()[1], 3);
  EXPECT_EQ(m.shape()[2], 3);
}

TEST(ThreeDimensionalMesh, IndexerReturnsVertex) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = three_dimensional::SUQMesh(1.0, dimensions, origin);
  auto v = m[{0, 0, 0}];
  EXPECT_DOUBLE_EQ(v.coordinates()[0], 0.0);
  EXPECT_DOUBLE_EQ(v.coordinates()[1], 0.0);
  EXPECT_DOUBLE_EQ(v.coordinates()[2], 0.0);
}

TEST(ThreeDimensionalMesh, NegativeIndexing) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = three_dimensional::SUQMesh(1.0, dimensions, origin);
  auto last = m[{-1, -1, -1}];
  EXPECT_DOUBLE_EQ(last.coordinates()[0], 2.0);
  EXPECT_DOUBLE_EQ(last.coordinates()[1], 2.0);
  EXPECT_DOUBLE_EQ(last.coordinates()[2], 2.0);
}

TEST(ThreeDimensionalMesh, ElementsAndVolume) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = three_dimensional::SUQMesh(1.0, dimensions, origin);
  EXPECT_EQ(m.elements().size(), 8U);
  EXPECT_DOUBLE_EQ(m.element_volume(), 1.0);
}

TEST(ThreeDimensionalMesh, GlobalIndexToCartesian) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = TestableSUQMesh3D(1.0, dimensions, origin);
  // shape is [3, 3, 3]; global index 13 = cartesian [1, 1, 1] (1*9 + 1*3 + 1 = 13)
  auto cart = m.global_index_to_cartesian(13, m.shape());
  ASSERT_EQ(cart.size(), 3U);
  EXPECT_EQ(cart[0], 1);
  EXPECT_EQ(cart[1], 1);
  EXPECT_EQ(cart[2], 1);
}

#ifdef DFT_HAS_MATPLOTLIB
TEST(ThreeDimensionalMesh, PlotWithMatplotlib) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{1, 1, 1};
  auto m = three_dimensional::SUQMesh(1.0, dimensions, origin);
  EXPECT_NO_THROW(m.plot("exports/test_mesh_3d.png", false));
}
#else
TEST(ThreeDimensionalMesh, PlotWithoutBackendThrows) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{1, 1, 1};
  auto m = three_dimensional::SUQMesh(1.0, dimensions, origin);
  EXPECT_THROW(m.plot("exports/test_mesh_3d.png", false), std::runtime_error);
}
#endif

TEST(ThreeDimensionalMesh, StreamOutput) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{1, 1, 1};
  auto m = three_dimensional::SUQMesh(1.0, dimensions, origin);
  std::ostringstream ss;
  ss << static_cast<const Mesh&>(m);
  auto output = ss.str();
  EXPECT_NE(output.find("Mesh object"), std::string::npos);
}
