#include "classicaldft_bits/geometry/3D/element.h"

#include <gtest/gtest.h>

using namespace dft::geometry;

TEST(ThreeDimensionalSquareBox, DefaultConstructor) {
  auto e = three_dimensional::SquareBox();
  ASSERT_EQ(8, e.number_of_vertices());
  ASSERT_EQ(3, e.dimension());
  ASSERT_DOUBLE_EQ(1.0, e.volume());
}

TEST(ThreeDimensionalSquareBox, ParameterizedConstructor) {
  auto e = three_dimensional::SquareBox(2.0, {0.0, 0.0, 0.0});
  ASSERT_EQ(8, e.number_of_vertices());
  ASSERT_DOUBLE_EQ(8.0, e.volume());
}

TEST(ThreeDimensionalSquareBox, MoveConstructorWith8Vertices) {
  vertex_vec verts = {
      Vertex({0, 0, 0}),
      Vertex({1, 0, 0}),
      Vertex({1, 1, 0}),
      Vertex({0, 1, 0}),
      Vertex({0, 1, 1}),
      Vertex({1, 1, 1}),
      Vertex({1, 0, 1}),
      Vertex({0, 0, 1})};
  auto e = three_dimensional::SquareBox(std::move(verts));
  ASSERT_EQ(8, e.number_of_vertices());
  ASSERT_EQ(3, e.dimension());
}

TEST(ThreeDimensionalSquareBox, MoveConstructorWrongCountThrows) {
  vertex_vec verts = {Vertex({0, 0, 0}), Vertex({1, 0, 0})};
  EXPECT_THROW(three_dimensional::SquareBox(std::move(verts)), std::runtime_error);
}
