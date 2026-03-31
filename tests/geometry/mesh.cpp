#include "dft/geometry/mesh.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::geometry;

// 2D mesh tests

TEST_CASE("uniform_mesh_2d computes correct shape", "[mesh]") {
  auto mesh = uniform_mesh_2d(0.5, {1.0, 1.0}, {0.0, 0.0});
  CHECK(mesh.shape[0] == 3);
  CHECK(mesh.shape[1] == 3);
  CHECK(mesh.vertices.size() == 9);
  CHECK(mesh.elements.size() == 4);
}

TEST_CASE("uniform_mesh_2d volume equals product of dimensions", "[mesh]") {
  auto mesh = uniform_mesh_2d(0.5, {2.0, 3.0}, {0.0, 0.0});
  CHECK(volume(mesh) == Catch::Approx(6.0));
}

TEST_CASE("uniform_mesh_2d element volume is dx squared", "[mesh]") {
  auto mesh = uniform_mesh_2d(0.25, {1.0, 1.0}, {0.0, 0.0});
  CHECK(element_volume(mesh) == Catch::Approx(0.0625));
}

TEST_CASE("uniform_mesh_2d spacing returns dx", "[mesh]") {
  auto mesh = uniform_mesh_2d(0.5, {1.0, 1.0}, {0.0, 0.0});
  CHECK(spacing(mesh) == Catch::Approx(0.5));
}

// 3D mesh tests

TEST_CASE("uniform_mesh_3d computes correct shape", "[mesh]") {
  auto mesh = uniform_mesh_3d(0.5, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
  CHECK(mesh.shape[0] == 3);
  CHECK(mesh.shape[1] == 3);
  CHECK(mesh.shape[2] == 3);
  CHECK(mesh.vertices.size() == 27);
  CHECK(mesh.elements.size() == 8);
}

TEST_CASE("uniform_mesh_3d volume equals product of dimensions", "[mesh]") {
  auto mesh = uniform_mesh_3d(0.5, {2.0, 3.0, 4.0}, {0.0, 0.0, 0.0});
  CHECK(volume(mesh) == Catch::Approx(24.0));
}

TEST_CASE("uniform_mesh_3d element volume is dx cubed", "[mesh]") {
  auto mesh = uniform_mesh_3d(0.5, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
  CHECK(element_volume(mesh) == Catch::Approx(0.125));
}

// Variant-based tests

TEST_CASE("mesh variant volume dispatches correctly", "[mesh]") {
  Mesh m2d = uniform_mesh_2d(0.5, {2.0, 3.0}, {0.0, 0.0});
  Mesh m3d = uniform_mesh_3d(0.5, {2.0, 3.0, 4.0}, {0.0, 0.0, 0.0});
  CHECK(volume(m2d) == Catch::Approx(6.0));
  CHECK(volume(m3d) == Catch::Approx(24.0));
}

TEST_CASE("mesh variant element_volume dispatches correctly", "[mesh]") {
  Mesh m2d = uniform_mesh_2d(0.5, {1.0, 1.0}, {0.0, 0.0});
  Mesh m3d = uniform_mesh_3d(0.5, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
  CHECK(element_volume(m2d) == Catch::Approx(0.25));
  CHECK(element_volume(m3d) == Catch::Approx(0.125));
}

// Indexing tests

TEST_CASE("flat_index is row-major with last index fastest", "[mesh]") {
  Mesh mesh = uniform_mesh_2d(0.5, {1.0, 1.0}, {0.0, 0.0});
  CHECK(flat_index(mesh, {0, 0}) == 0);
  CHECK(flat_index(mesh, {0, 1}) == 1);
  CHECK(flat_index(mesh, {0, 2}) == 2);
  CHECK(flat_index(mesh, {1, 0}) == 3);
  CHECK(flat_index(mesh, {2, 2}) == 8);
}

TEST_CASE("flat_index for 3D mesh", "[mesh]") {
  Mesh mesh = uniform_mesh_3d(0.5, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
  CHECK(flat_index(mesh, {0, 0, 0}) == 0);
  CHECK(flat_index(mesh, {0, 0, 1}) == 1);
  CHECK(flat_index(mesh, {0, 1, 0}) == 3);
  CHECK(flat_index(mesh, {1, 0, 0}) == 9);
}

TEST_CASE("cartesian_index is inverse of flat_index", "[mesh]") {
  Mesh mesh = uniform_mesh_3d(0.5, {1.0, 1.0, 1.0}, {0.0, 0.0, 0.0});
  auto idx = cartesian_index(mesh, 14);
  CHECK(flat_index(mesh, idx) == 14);
}

TEST_CASE("vertex access by cartesian index", "[mesh]") {
  Mesh mesh = uniform_mesh_2d(1.0, {2.0, 2.0}, {0.0, 0.0});
  auto& v = vertex(mesh, {1, 2});
  CHECK(v[0] == Catch::Approx(1.0));
  CHECK(v[1] == Catch::Approx(2.0));
}

TEST_CASE("vertex access with negative indices", "[mesh]") {
  Mesh mesh = uniform_mesh_2d(1.0, {2.0, 2.0}, {0.0, 0.0});
  // shape is {3, 3}, so -1 means index 2
  auto& v = vertex(mesh, {-1, -1});
  CHECK(v[0] == Catch::Approx(2.0));
  CHECK(v[1] == Catch::Approx(2.0));
}

TEST_CASE("vertex access out of bounds throws", "[mesh]") {
  Mesh mesh = uniform_mesh_2d(1.0, {2.0, 2.0}, {0.0, 0.0});
  REQUIRE_THROWS_AS(vertex(mesh, {3, 0}), std::out_of_range);
}

TEST_CASE("flat_index with wrong dimension throws", "[mesh]") {
  Mesh mesh = uniform_mesh_2d(1.0, {2.0, 2.0}, {0.0, 0.0});
  REQUIRE_THROWS_AS(flat_index(mesh, {0, 0, 0}), std::invalid_argument);
}

// Periodic boundary conditions

TEST_CASE("wrap folds position into periodic box (2D)", "[mesh]") {
  Mesh mesh = uniform_mesh_2d(1.0, {4.0, 4.0}, {0.0, 0.0});

  auto wrapped = wrap(mesh, Vertex{{5.0, 6.0}});
  CHECK(wrapped[0] == Catch::Approx(1.0));
  CHECK(wrapped[1] == Catch::Approx(2.0));
}

TEST_CASE("wrap handles negative coordinates", "[mesh]") {
  Mesh mesh = uniform_mesh_2d(1.0, {4.0, 4.0}, {0.0, 0.0});

  auto wrapped = wrap(mesh, Vertex{{-1.0, -2.0}});
  CHECK(wrapped[0] == Catch::Approx(3.0));
  CHECK(wrapped[1] == Catch::Approx(2.0));
}

TEST_CASE("wrap leaves interior position unchanged", "[mesh]") {
  Mesh mesh = uniform_mesh_2d(1.0, {4.0, 4.0}, {0.0, 0.0});

  auto wrapped = wrap(mesh, Vertex{{2.5, 1.5}});
  CHECK(wrapped[0] == Catch::Approx(2.5));
  CHECK(wrapped[1] == Catch::Approx(1.5));
}

TEST_CASE("wrap for 3D mesh", "[mesh]") {
  Mesh mesh = uniform_mesh_3d(1.0, {4.0, 4.0, 4.0}, {0.0, 0.0, 0.0});

  auto wrapped = wrap(mesh, Vertex{{5.5, -0.5, 8.5}});
  CHECK(wrapped[0] == Catch::Approx(1.5));
  CHECK(wrapped[1] == Catch::Approx(3.5));
  CHECK(wrapped[2] == Catch::Approx(0.5));
}
