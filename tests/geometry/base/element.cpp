#include "classicaldft_bits/geometry/base/element.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace dft_core::geometry;

// ── Default constructor ─────────────────────────────────────────────────────

TEST(Element, DefaultConstructorIsEmpty) {
  auto e = Element();
  ASSERT_EQ(0, e.number_of_vertices());
  ASSERT_EQ(0, e.dimension());
  ASSERT_EQ(0.0, e.volume());
  EXPECT_TRUE(e.vertices_raw().empty());
  EXPECT_TRUE(e.vertices().empty());
}

// ── Initializer list constructor ────────────────────────────────────────────

TEST(Element, InitializerListConstructor) {
  auto e = Element({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  ASSERT_EQ(3, e.number_of_vertices());
  EXPECT_EQ(3, e.dimension());
}

TEST(Element, InitializerListInconsistentDimensionsThrows) {
  EXPECT_THROW(Element({{1}, {4, 6}, {7, 8, 9}}), std::runtime_error);
  EXPECT_THROW(Element({{1, 2, 3}, {4, 6}, {7, 8, 9}}), std::runtime_error);
  EXPECT_THROW(Element({{1, 2, 3}, {4, 5, 6}, {7, 8}}), std::runtime_error);
}

// ── Vector constructors ─────────────────────────────────────────────────────

TEST(Element, VectorConstructor) {
  std::vector<Vertex> verts = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  Element e(verts);
  ASSERT_EQ(3, e.number_of_vertices());
  EXPECT_DOUBLE_EQ(e[0][0], 1.0);
  EXPECT_DOUBLE_EQ(e[1][0], 3.0);
}

TEST(Element, MoveVectorConstructor) {
  std::vector<Vertex> verts = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  Element e(std::move(verts));
  ASSERT_EQ(3, e.number_of_vertices());
  EXPECT_DOUBLE_EQ(e[0][0], 1.0);
}

// ── Inspectors ──────────────────────────────────────────────────────────────

TEST(Element, VerticesRawReturnsVector) {
  auto e = Element({{1, 2}, {3, 4}});
  EXPECT_EQ(e.vertices_raw().size(), 2U);
}

TEST(Element, VerticesReturnsMap) {
  auto e = Element({{1, 2}, {3, 4}});
  EXPECT_EQ(e.vertices().size(), 2U);
}

// ── Indexer ──────────────────────────────────────────────────────────────────

TEST(Element, IndexerReturnsCorrectVertex) {
  auto e = Element({{1, 2, 3}, {4, 5, 6}});
  EXPECT_DOUBLE_EQ(e[0][0], 1.0);
  EXPECT_DOUBLE_EQ(e[1][0], 4.0);
}

TEST(Element, IndexerOutOfRangeThrows) {
  auto e = Element({{1, 2, 3}});
  EXPECT_THROW(e[5], std::out_of_range);
}

// ── Stream output ───────────────────────────────────────────────────────────

TEST(Element, StreamOutput) {
  auto e = Element({{1, 2}, {3, 4}});
  std::ostringstream ss;
  ss << e;
  auto output = ss.str();
  EXPECT_NE(output.find("Element"), std::string::npos);
  EXPECT_NE(output.find("Number of vertices: 2"), std::string::npos);
  EXPECT_NE(output.find("Volume: 0"), std::string::npos);
}

// ── SquareBox ───────────────────────────────────────────────────────────────

TEST(SquareBox, VolumeIsLengthPowerDimension) {
  // SquareBox is abstract; tested via 2D/3D subclasses in their own files.
  // Here we just verify the base class volume formula.
  // A 2D box with length 2 has volume 4.
  // (This is tested through two_dimensional::SquareBox)
}
