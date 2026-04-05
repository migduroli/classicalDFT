#include "dft/geometry/vertex.hpp"

#include <catch2/catch_test_macros.hpp>
#include <sstream>

using namespace dft::geometry;

TEST_CASE("vertex default-constructs with empty coordinates", "[vertex]") {
  Vertex v;
  CHECK(v.coordinates.empty());
  CHECK(dimension(v) == 0);
}

TEST_CASE("vertex supports designated initializer construction", "[vertex]") {
  Vertex v{.coordinates = {1.0, 2.0, 3.0}};
  CHECK(dimension(v) == 3);
  CHECK(v.coordinates[0] == 1.0);
  CHECK(v.coordinates[1] == 2.0);
  CHECK(v.coordinates[2] == 3.0);
}

TEST_CASE("vertex operator[] accesses coordinates", "[vertex]") {
  Vertex v{{4.0, 5.0}};
  CHECK(v[0] == 4.0);
  CHECK(v[1] == 5.0);
  REQUIRE_THROWS_AS(v[2], std::out_of_range);
}

TEST_CASE("vertex operator[] allows mutable access", "[vertex]") {
  Vertex v{{1.0, 2.0}};
  v[0] = 10.0;
  CHECK(v[0] == 10.0);
}

TEST_CASE("vertex addition produces element-wise sum", "[vertex]") {
  Vertex a{{1.0, 2.0, 3.0}};
  Vertex b{{4.0, 5.0, 6.0}};
  auto c = a + b;
  CHECK(c[0] == 5.0);
  CHECK(c[1] == 7.0);
  CHECK(c[2] == 9.0);
}

TEST_CASE("vertex subtraction produces element-wise difference", "[vertex]") {
  Vertex a{{10.0, 20.0}};
  Vertex b{{3.0, 7.0}};
  auto c = a - b;
  CHECK(c[0] == 7.0);
  CHECK(c[1] == 13.0);
}

TEST_CASE("vertex addition throws for mismatched dimensions", "[vertex]") {
  Vertex a{{1.0, 2.0}};
  Vertex b{{1.0, 2.0, 3.0}};
  REQUIRE_THROWS_AS(a + b, std::invalid_argument);
}

TEST_CASE("vertex subtraction throws for mismatched dimensions", "[vertex]") {
  Vertex a{{1.0}};
  Vertex b{{1.0, 2.0}};
  REQUIRE_THROWS_AS(a - b, std::invalid_argument);
}

TEST_CASE("vertex stream output produces readable format", "[vertex]") {
  Vertex v{{1.5, 2.5, 3.5}};
  std::ostringstream os;
  os << v;
  CHECK(os.str() == "(1.5, 2.5, 3.5)");
}

TEST_CASE("vertex stream output for empty vertex", "[vertex]") {
  Vertex v;
  std::ostringstream os;
  os << v;
  CHECK(os.str() == "()");
}
