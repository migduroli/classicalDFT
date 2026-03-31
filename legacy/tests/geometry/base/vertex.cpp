#include "dft/geometry/base/vertex.h"

#include <gtest/gtest.h>
#include <sstream>

using namespace dft::geometry;

// ── Constructors ────────────────────────────────────────────────────────────

TEST(Vertex, DefaultConstructorHasZeroDimension) {
  auto p = Vertex();
  ASSERT_EQ(p.dimension(), 0);
  EXPECT_TRUE(p.coordinates().empty());
}

TEST(Vertex, ConstVectorConstructor) {
  auto x = std::vector<double>{0, 1, 2};
  auto p = Vertex(x);
  ASSERT_EQ(p.dimension(), 3);
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_DOUBLE_EQ(x[i], p.coordinates()[i]);
  }
}

TEST(Vertex, MoveVectorConstructor) {
  auto x = std::vector<double>{0, 1, 2};
  auto p = Vertex(std::move(x));
  ASSERT_EQ(p.dimension(), 3);
  ASSERT_EQ(x.size(), 0U);
  EXPECT_DOUBLE_EQ(p[0], 0.0);
  EXPECT_DOUBLE_EQ(p[1], 1.0);
  EXPECT_DOUBLE_EQ(p[2], 2.0);
}

TEST(Vertex, InitializerListConstructor) {
  Vertex p = {0, 1, 2};
  ASSERT_EQ(p.dimension(), 3);
  EXPECT_DOUBLE_EQ(p[0], 0.0);
  EXPECT_DOUBLE_EQ(p[1], 1.0);
  EXPECT_DOUBLE_EQ(p[2], 2.0);
}

// ── Indexer ─────────────────────────────────────────────────────────────────

TEST(Vertex, IndexerReturnsCorrectCoordinate) {
  Vertex p = {0.0, 1.0, 2.0};
  ASSERT_DOUBLE_EQ(0.0, p[0]);
  ASSERT_DOUBLE_EQ(1.0, p[1]);
  ASSERT_DOUBLE_EQ(2.0, p[2]);
}

TEST(Vertex, IndexerThrowsOutOfRange) {
  Vertex p = {0.0, 1.0};
  EXPECT_THROW(p[5], std::out_of_range);
}

// ── Arithmetic operators ────────────────────────────────────────────────────

TEST(Vertex, AdditionOfSameDimension) {
  Vertex p1 = {0.0, 1.0, 2.0};
  Vertex p2 = {2.0, 1.0, 0.0};
  auto sum = p1 + p2;
  ASSERT_DOUBLE_EQ(2.0, sum[0]);
  ASSERT_DOUBLE_EQ(2.0, sum[1]);
  ASSERT_DOUBLE_EQ(2.0, sum[2]);
}

TEST(Vertex, AdditionOfDifferentDimensionThrows) {
  Vertex p1 = {0.0, 1.0, 2.0};
  Vertex p2 = {2.0, 1.0};
  EXPECT_THROW(p1 + p2, std::runtime_error);
}

TEST(Vertex, SubtractionOfSameDimension) {
  Vertex p1 = {0.0, 1.0, 2.0};
  Vertex p2 = {2.0, 1.0, 0.0};
  auto result = p2 - p1;
  ASSERT_DOUBLE_EQ(2.0, result[0]);
  ASSERT_DOUBLE_EQ(0.0, result[1]);
  ASSERT_DOUBLE_EQ(-2.0, result[2]);
}

TEST(Vertex, SubtractionOfDifferentDimensionThrows) {
  Vertex p1 = {0.0, 1.0, 2.0};
  Vertex p2 = {2.0, 1.0};
  EXPECT_THROW(p1 - p2, std::runtime_error);
}

// ── Stream output ───────────────────────────────────────────────────────────

TEST(Vertex, StreamOutputForNonEmpty) {
  Vertex p = {1.0, 2.0, 3.0};
  std::ostringstream ss;
  ss << p;
  auto output = ss.str();
  EXPECT_NE(output.find("dimensions = 3"), std::string::npos);
  EXPECT_NE(output.find("(1, 2, 3)"), std::string::npos);
}

TEST(Vertex, StreamOutputForEmpty) {
  Vertex p;
  std::ostringstream ss;
  ss << p;
  auto output = ss.str();
  EXPECT_NE(output.find("dimensions = 0"), std::string::npos);
  EXPECT_NE(output.find("()"), std::string::npos);
}
