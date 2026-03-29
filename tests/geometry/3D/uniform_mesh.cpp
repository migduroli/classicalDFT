#include "dft/geometry/3D/uniform_mesh.h"

#include <gtest/gtest.h>

using namespace dft::geometry;

// ── 3D UniformMesh construction ─────────────────────────────────────────────

TEST(ThreeDimensionalUniformMesh, ConstructionAndBasicProperties) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{1, 1, 1};
  auto m = three_dimensional::UniformMesh(0.5, dimensions, origin);
  ASSERT_EQ(27, m.number_vertices());
  ASSERT_DOUBLE_EQ(1.0, m.volume());
}

TEST(ThreeDimensionalUniformMesh, ShapeIsCorrect) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  ASSERT_EQ(m.shape().size(), 3U);
  EXPECT_EQ(m.shape()[0], 3);
  EXPECT_EQ(m.shape()[1], 3);
  EXPECT_EQ(m.shape()[2], 3);
}

TEST(ThreeDimensionalUniformMesh, Spacing) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{1, 1, 1};
  auto m = three_dimensional::UniformMesh(0.25, dimensions, origin);
  EXPECT_DOUBLE_EQ(m.spacing(), 0.25);
}

// ── Elements ────────────────────────────────────────────────────────────────

TEST(ThreeDimensionalUniformMesh, ElementsAndVolume) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  EXPECT_EQ(m.elements().size(), 8U);
  EXPECT_DOUBLE_EQ(m.element_volume(), 1.0);
}

// ── Indexing ────────────────────────────────────────────────────────────────

TEST(ThreeDimensionalUniformMesh, IndexerReturnsVertex) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  auto v = m[{0, 0, 0}];
  EXPECT_DOUBLE_EQ(v.coordinates()[0], 0.0);
  EXPECT_DOUBLE_EQ(v.coordinates()[1], 0.0);
  EXPECT_DOUBLE_EQ(v.coordinates()[2], 0.0);
}

TEST(ThreeDimensionalUniformMesh, NegativeIndexing) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{2, 2, 2};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  auto last = m[{-1, -1, -1}];
  EXPECT_DOUBLE_EQ(last.coordinates()[0], 2.0);
  EXPECT_DOUBLE_EQ(last.coordinates()[1], 2.0);
  EXPECT_DOUBLE_EQ(last.coordinates()[2], 2.0);
}

// ── PBC wrap ────────────────────────────────────────────────────────────────

TEST(ThreeDimensionalUniformMesh, WrapInsideIsIdentity) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{4, 6, 8};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  auto w = m.wrap(Vertex({1.5, 2.5, 3.5}));
  EXPECT_DOUBLE_EQ(w.coordinates()[0], 1.5);
  EXPECT_DOUBLE_EQ(w.coordinates()[1], 2.5);
  EXPECT_DOUBLE_EQ(w.coordinates()[2], 3.5);
}

TEST(ThreeDimensionalUniformMesh, WrapPositiveOverflow) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{4, 4, 4};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  auto w = m.wrap(Vertex({5.5, 7.0, 12.5}));
  EXPECT_NEAR(w.coordinates()[0], 1.5, 1e-12);
  EXPECT_NEAR(w.coordinates()[1], 3.0, 1e-12);
  EXPECT_NEAR(w.coordinates()[2], 0.5, 1e-12);
}

TEST(ThreeDimensionalUniformMesh, WrapNegative) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{4, 4, 4};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  auto w = m.wrap(Vertex({-1.0, -0.5, -5.0}));
  EXPECT_NEAR(w.coordinates()[0], 3.0, 1e-12);
  EXPECT_NEAR(w.coordinates()[1], 3.5, 1e-12);
  EXPECT_NEAR(w.coordinates()[2], 3.0, 1e-12);
}

TEST(ThreeDimensionalUniformMesh, WrapExactBoundary) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{4, 4, 4};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  auto w = m.wrap(Vertex({4.0, 0.0, 4.0}));
  EXPECT_NEAR(w.coordinates()[0], 0.0, 1e-12);
  EXPECT_NEAR(w.coordinates()[1], 0.0, 1e-12);
  EXPECT_NEAR(w.coordinates()[2], 0.0, 1e-12);
}

#ifdef DFT_HAS_MATPLOTLIB
TEST(ThreeDimensionalUniformMesh, PlotWithMatplotlib) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{1, 1, 1};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  EXPECT_NO_THROW(m.plot("exports/test_uniform_mesh_3d.png", false));
}
#else
TEST(ThreeDimensionalUniformMesh, PlotWithoutBackendThrows) {
  auto origin = std::vector<double>{0, 0, 0};
  auto dimensions = std::vector<double>{1, 1, 1};
  auto m = three_dimensional::UniformMesh(1.0, dimensions, origin);
  EXPECT_THROW(m.plot("exports/test_uniform_mesh_3d.png", false), std::runtime_error);
}
#endif
