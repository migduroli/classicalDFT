#include "dft/geometry/2D/mesh.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace dft::geometry;

// Test wrapper to access protected method
class TestableSUQMesh2D : public two_dimensional::SUQMesh {
 public:
  using two_dimensional::SUQMesh::global_index_to_cartesian;
  using two_dimensional::SUQMesh::SUQMesh;
};

// ── 2D SUQ Mesh construction ────────────────────────────────────────────────

TEST(TwoDimensionalMesh, ConstructionAndBasicProperties) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::SUQMesh(0.5, dimensions, origin);
  ASSERT_EQ(9, m.number_vertices());
  ASSERT_DOUBLE_EQ(1.0, m.volume());
}

TEST(TwoDimensionalMesh, ShapeIsCorrect) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 3};
  auto m = two_dimensional::SUQMesh(1.0, dimensions, origin);
  ASSERT_EQ(m.shape().size(), 2U);
  EXPECT_EQ(m.shape()[0], 3);
  EXPECT_EQ(m.shape()[1], 4);
}

TEST(TwoDimensionalMesh, DimensionsAccessor) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 3};
  auto m = two_dimensional::SUQMesh(1.0, dimensions, origin);
  ASSERT_EQ(m.dimensions().size(), 2U);
  EXPECT_DOUBLE_EQ(m.dimensions()[0], 2.0);
  EXPECT_DOUBLE_EQ(m.dimensions()[1], 3.0);
}

TEST(TwoDimensionalMesh, VerticesAccessor) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::SUQMesh(0.5, dimensions, origin);
  EXPECT_EQ(m.vertices().size(), static_cast<size_t>(m.number_vertices()));
}

// ── Indexing ────────────────────────────────────────────────────────────────

TEST(TwoDimensionalMesh, IndexerReturnsVertex) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 2};
  auto m = two_dimensional::SUQMesh(1.0, dimensions, origin);
  auto v = m[{0, 0}];
  EXPECT_DOUBLE_EQ(v.coordinates()[0], 0.0);
  EXPECT_DOUBLE_EQ(v.coordinates()[1], 0.0);
}

TEST(TwoDimensionalMesh, NegativeIndexing) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 2};
  auto m = two_dimensional::SUQMesh(1.0, dimensions, origin);
  auto last = m[{-1, -1}];
  EXPECT_DOUBLE_EQ(last.coordinates()[0], 2.0);
  EXPECT_DOUBLE_EQ(last.coordinates()[1], 2.0);
}

TEST(TwoDimensionalMesh, OutOfBoundsThrows) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::SUQMesh(0.5, dimensions, origin);
  auto oob = std::vector<long>{100, 100};
  EXPECT_THROW(m[oob], std::runtime_error);
}

TEST(TwoDimensionalMesh, WrongDimensionIndexThrows) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::SUQMesh(0.5, dimensions, origin);
  auto bad = std::vector<long>{0, 0, 0};
  EXPECT_THROW(m[bad], std::runtime_error);
}

// ── Elements ────────────────────────────────────────────────────────────────

TEST(TwoDimensionalMesh, ElementsAccessor) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 2};
  auto m = two_dimensional::SUQMesh(1.0, dimensions, origin);
  EXPECT_EQ(m.elements().size(), 4U);
}

TEST(TwoDimensionalMesh, ElementVolume) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 2};
  auto m = two_dimensional::SUQMesh(0.5, dimensions, origin);
  EXPECT_DOUBLE_EQ(m.element_volume(), 0.25);
}

// ── Global index conversion ──────────────────────────────────────────────────

TEST(TwoDimensionalMesh, GlobalIndexToCartesian) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 3};
  auto m = TestableSUQMesh2D(1.0, dimensions, origin);
  // shape is [3, 4]; global index 5 = cartesian [1, 1] (row-major: 1*4 + 1 = 5)
  auto cart = m.global_index_to_cartesian(5, m.shape());
  ASSERT_EQ(cart.size(), 2U);
  EXPECT_EQ(cart[0], 1);
  EXPECT_EQ(cart[1], 1);
  // Test index 0 = [0, 0]
  auto first = m.global_index_to_cartesian(0, m.shape());
  EXPECT_EQ(first[0], 0);
  EXPECT_EQ(first[1], 0);
}

// ── Stream output ───────────────────────────────────────────────────────────

TEST(TwoDimensionalMesh, StreamOutput) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::SUQMesh(0.5, dimensions, origin);
  std::ostringstream ss;
  ss << static_cast<const Mesh&>(m);
  auto output = ss.str();
  EXPECT_NE(output.find("Mesh object"), std::string::npos);
  EXPECT_NE(output.find("Volume:"), std::string::npos);
  EXPECT_NE(output.find("Number of vertices:"), std::string::npos);
  EXPECT_NE(output.find("Shape:"), std::string::npos);
  EXPECT_NE(output.find("Dimensions:"), std::string::npos);
}

// ── Plot (without Grace) ────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
TEST(TwoDimensionalMesh, PlotWithMatplotlib) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::SUQMesh(0.5, dimensions, origin);
  EXPECT_NO_THROW(m.plot("exports/test_mesh_2d.png", false));
}
#elif !defined(DFT_HAS_GRACE)
TEST(TwoDimensionalMesh, PlotWithoutBackendThrows) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::SUQMesh(0.5, dimensions, origin);
  EXPECT_THROW(m.plot("exports/test_mesh_2d.png", false), std::runtime_error);
}
#endif
