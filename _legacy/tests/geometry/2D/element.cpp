#include "dft/geometry/2D/element.h"

#include <gtest/gtest.h>

using namespace dft::geometry;

// ── Default constructor ─────────────────────────────────────────────────────

TEST(TwoDimensionalSquareBox, DefaultConstructor) {
  auto e = two_dimensional::SquareBox();
  ASSERT_EQ(4, e.number_of_vertices());
  ASSERT_EQ(2, e.dimension());
  ASSERT_DOUBLE_EQ(1.0, e.volume());
}

// ── Parameterized constructor ───────────────────────────────────────────────

TEST(TwoDimensionalSquareBox, ParameterizedConstructor) {
  auto e = two_dimensional::SquareBox(2.0, {1.0, 1.0});
  ASSERT_EQ(4, e.number_of_vertices());
  ASSERT_EQ(2, e.dimension());
  ASSERT_DOUBLE_EQ(4.0, e.volume());
}

// ── Move constructor ────────────────────────────────────────────────────────

TEST(TwoDimensionalSquareBox, MoveConstructorWith4Vertices) {
  vertex_vec verts = {Vertex({0, 0}), Vertex({1, 0}), Vertex({1, 1}), Vertex({0, 1})};
  auto e = two_dimensional::SquareBox(std::move(verts));
  ASSERT_EQ(4, e.number_of_vertices());
  ASSERT_EQ(2, e.dimension());
}

TEST(TwoDimensionalSquareBox, MoveConstructorWrongCountThrows) {
  vertex_vec verts = {Vertex({0, 0}), Vertex({1, 0}), Vertex({1, 1})};
  EXPECT_THROW(two_dimensional::SquareBox(std::move(verts)), std::runtime_error);
}

// ── Indexer ──────────────────────────────────────────────────────────────────

TEST(TwoDimensionalSquareBox, IndexerReturnsCorrectVertices) {
  auto e = two_dimensional::SquareBox();
  EXPECT_DOUBLE_EQ(0.0, e[0].coordinates()[0]);
  EXPECT_DOUBLE_EQ(0.0, e[0].coordinates()[1]);
}
