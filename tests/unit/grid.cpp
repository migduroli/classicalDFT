#include "dft/grid.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace dft;

TEST_CASE("make_grid computes shape from box and dx", "[grid]") {
  auto g = make_grid(0.1, { 1.0, 2.0, 3.0 });

  CHECK(g.dx == 0.1);
  CHECK(g.box_size[0] == 1.0);
  CHECK(g.box_size[1] == 2.0);
  CHECK(g.box_size[2] == 3.0);
  CHECK(g.shape[0] == 10);
  CHECK(g.shape[1] == 20);
  CHECK(g.shape[2] == 30);
}

TEST_CASE("make_grid rejects non-positive dx", "[grid]") {
  REQUIRE_THROWS_AS(make_grid(0.0, { 1.0, 1.0, 1.0 }), std::invalid_argument);
  REQUIRE_THROWS_AS(make_grid(-0.1, { 1.0, 1.0, 1.0 }), std::invalid_argument);
}

TEST_CASE("make_grid rejects non-positive box dimensions", "[grid]") {
  REQUIRE_THROWS_AS(make_grid(0.1, { 0.0, 1.0, 1.0 }), std::invalid_argument);
  REQUIRE_THROWS_AS(make_grid(0.1, { 1.0, -1.0, 1.0 }), std::invalid_argument);
}

TEST_CASE("make_grid rejects incommensurate box", "[grid]") {
  REQUIRE_THROWS_AS(make_grid(0.3, { 1.0, 1.0, 1.0 }), std::invalid_argument);
}

TEST_CASE("cell_volume is dx cubed", "[grid]") {
  auto g = make_grid(0.5, { 1.0, 1.0, 1.0 });
  CHECK(g.cell_volume() == 0.5 * 0.5 * 0.5);
}

TEST_CASE("total_points is product of shape", "[grid]") {
  auto g = make_grid(0.5, { 1.0, 2.0, 3.0 });
  CHECK(g.total_points() == 2 * 4 * 6);
}

TEST_CASE("flat_index is row-major with z fastest", "[grid]") {
  auto g = make_grid(1.0, { 3.0, 4.0, 5.0 });
  CHECK(g.flat_index(0, 0, 0) == 0);
  CHECK(g.flat_index(0, 0, 1) == 1);
  CHECK(g.flat_index(0, 1, 0) == 5);
  CHECK(g.flat_index(1, 0, 0) == 20);
  CHECK(g.flat_index(2, 3, 4) == 2 * 20 + 3 * 5 + 4);
}

TEST_CASE("grid supports designated initializer construction", "[grid]") {
  Grid g{ .dx = 0.25, .box_size = { 1.0, 1.0, 1.0 }, .shape = { 4, 4, 4 } };
  CHECK(g.total_points() == 64);
}
