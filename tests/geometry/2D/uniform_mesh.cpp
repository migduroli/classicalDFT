#include "dft/geometry/2D/uniform_mesh.h"

#include <gtest/gtest.h>

using namespace dft::geometry;

// ── 2D UniformMesh construction ─────────────────────────────────────────────

TEST(TwoDimensionalUniformMesh, ConstructionAndBasicProperties) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::UniformMesh(0.5, dimensions, origin);
  ASSERT_EQ(9, m.number_vertices());
  ASSERT_DOUBLE_EQ(1.0, m.volume());
}

TEST(TwoDimensionalUniformMesh, ShapeIsCorrect) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 3};
  auto m = two_dimensional::UniformMesh(1.0, dimensions, origin);
  ASSERT_EQ(m.shape().size(), 2U);
  EXPECT_EQ(m.shape()[0], 3);
  EXPECT_EQ(m.shape()[1], 4);
}

TEST(TwoDimensionalUniformMesh, Spacing) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::UniformMesh(0.25, dimensions, origin);
  EXPECT_DOUBLE_EQ(m.spacing(), 0.25);
}

// ── Elements ────────────────────────────────────────────────────────────────

TEST(TwoDimensionalUniformMesh, ElementsAccessor) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 2};
  auto m = two_dimensional::UniformMesh(1.0, dimensions, origin);
  EXPECT_EQ(m.elements().size(), 4U);
}

TEST(TwoDimensionalUniformMesh, ElementVolume) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 2};
  auto m = two_dimensional::UniformMesh(0.5, dimensions, origin);
  EXPECT_DOUBLE_EQ(m.element_volume(), 0.25);
}

// ── Indexing ────────────────────────────────────────────────────────────────

TEST(TwoDimensionalUniformMesh, IndexerReturnsVertex) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 2};
  auto m = two_dimensional::UniformMesh(1.0, dimensions, origin);
  auto v = m[{0, 0}];
  EXPECT_DOUBLE_EQ(v.coordinates()[0], 0.0);
  EXPECT_DOUBLE_EQ(v.coordinates()[1], 0.0);
}

TEST(TwoDimensionalUniformMesh, NegativeIndexing) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{2, 2};
  auto m = two_dimensional::UniformMesh(1.0, dimensions, origin);
  auto last = m[{-1, -1}];
  EXPECT_DOUBLE_EQ(last.coordinates()[0], 2.0);
  EXPECT_DOUBLE_EQ(last.coordinates()[1], 2.0);
}

// ── PBC wrap ────────────────────────────────────────────────────────────────

TEST(TwoDimensionalUniformMesh, WrapInsideIsIdentity) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{4, 6};
  auto m = two_dimensional::UniformMesh(1.0, dimensions, origin);
  auto w = m.wrap(Vertex({1.5, 2.5}));
  EXPECT_DOUBLE_EQ(w.coordinates()[0], 1.5);
  EXPECT_DOUBLE_EQ(w.coordinates()[1], 2.5);
}

TEST(TwoDimensionalUniformMesh, WrapPositiveOverflow) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{4, 4};
  auto m = two_dimensional::UniformMesh(1.0, dimensions, origin);
  auto w = m.wrap(Vertex({5.5, 7.0}));
  EXPECT_NEAR(w.coordinates()[0], 1.5, 1e-12);
  EXPECT_NEAR(w.coordinates()[1], 3.0, 1e-12);
}

TEST(TwoDimensionalUniformMesh, WrapNegative) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{4, 4};
  auto m = two_dimensional::UniformMesh(1.0, dimensions, origin);
  auto w = m.wrap(Vertex({-1.0, -0.5}));
  EXPECT_NEAR(w.coordinates()[0], 3.0, 1e-12);
  EXPECT_NEAR(w.coordinates()[1], 3.5, 1e-12);
}

// ── Plot (without Grace) ────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
TEST(TwoDimensionalUniformMesh, PlotWithMatplotlib) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::UniformMesh(0.5, dimensions, origin);
  EXPECT_NO_THROW(m.plot("exports/test_uniform_mesh_2d.png", false));
}
#elif !defined(DFT_HAS_GRACE)
TEST(TwoDimensionalUniformMesh, PlotWithoutBackendThrows) {
  auto origin = std::vector<double>{0, 0};
  auto dimensions = std::vector<double>{1, 1};
  auto m = two_dimensional::UniformMesh(0.5, dimensions, origin);
  EXPECT_THROW(m.plot("exports/test_uniform_mesh_2d.png", false), std::runtime_error);
}
#endif
