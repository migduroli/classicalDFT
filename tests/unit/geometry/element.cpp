#include "dft/geometry/element.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::geometry;

TEST_CASE("make_square_box_2d produces 4 vertices", "[element]") {
  auto box = make_square_box_2d(1.0, { 0.0, 0.0 });
  CHECK(box.vertices.size() == 4);
  CHECK(box.length == 1.0);
  CHECK(box.origin == std::vector<double>{ 0.0, 0.0 });
}

TEST_CASE("make_square_box_2d vertices are anti-clockwise", "[element]") {
  auto box = make_square_box_2d(2.0, { 1.0, 3.0 });
  CHECK(box.vertices[0].coordinates == std::vector<double>{ 1.0, 3.0 });
  CHECK(box.vertices[1].coordinates == std::vector<double>{ 3.0, 3.0 });
  CHECK(box.vertices[2].coordinates == std::vector<double>{ 3.0, 5.0 });
  CHECK(box.vertices[3].coordinates == std::vector<double>{ 1.0, 5.0 });
}

TEST_CASE("make_square_box_3d produces 8 vertices", "[element]") {
  auto box = make_square_box_3d(1.0, { 0.0, 0.0, 0.0 });
  CHECK(box.vertices.size() == 8);
  CHECK(box.length == 1.0);
  CHECK(box.origin == std::vector<double>{ 0.0, 0.0, 0.0 });
}

TEST_CASE("volume of SquareBox2D is length squared", "[element]") {
  auto box = make_square_box_2d(3.0, { 0.0, 0.0 });
  CHECK(box.volume() == Catch::Approx(9.0));
}

TEST_CASE("volume of SquareBox3D is length cubed", "[element]") {
  auto box = make_square_box_3d(2.0, { 0.0, 0.0, 0.0 });
  CHECK(box.volume() == Catch::Approx(8.0));
}

TEST_CASE("volume of ElementVariant dispatches correctly", "[element]") {
  auto sq = make_square_box_2d(4.0, { 0.0, 0.0 });
  auto cb = make_square_box_3d(2.0, { 0.0, 0.0, 0.0 });

  CHECK(sq.volume() == Catch::Approx(16.0));
  CHECK(cb.volume() == Catch::Approx(8.0));
}

TEST_CASE("dimension of ElementVariant", "[element]") {
  auto sq = make_square_box_2d(1.0, { 0.0, 0.0 });
  auto cb = make_square_box_3d(1.0, { 0.0, 0.0, 0.0 });
  Element empty{};

  CHECK(sq.vertices.front().dimension() == 2);
  CHECK(cb.vertices.front().dimension() == 3);
  CHECK(empty.vertices.empty());
}

TEST_CASE("element struct supports direct member access", "[element]") {
  Element e{ .vertices = { Vertex{ { 1.0, 2.0 } }, Vertex{ { 3.0, 4.0 } } } };
  CHECK(e.vertices.size() == 2);
  CHECK(e.vertices[0][0] == 1.0);
}
