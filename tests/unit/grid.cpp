#include "dft/grid.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;

TEST_CASE("make_grid computes shape from box and dx", "[grid]") {
  auto g = make_grid(0.1, {1.0, 2.0, 3.0});

  CHECK(g.dx == 0.1);
  CHECK(g.box_size[0] == 1.0);
  CHECK(g.box_size[1] == 2.0);
  CHECK(g.box_size[2] == 3.0);
  CHECK(g.shape[0] == 10);
  CHECK(g.shape[1] == 20);
  CHECK(g.shape[2] == 30);
}

TEST_CASE("make_grid rejects non-positive dx", "[grid]") {
  REQUIRE_THROWS_AS(make_grid(0.0, {1.0, 1.0, 1.0}), std::invalid_argument);
  REQUIRE_THROWS_AS(make_grid(-0.1, {1.0, 1.0, 1.0}), std::invalid_argument);
}

TEST_CASE("make_grid rejects non-positive box dimensions", "[grid]") {
  REQUIRE_THROWS_AS(make_grid(0.1, {0.0, 1.0, 1.0}), std::invalid_argument);
  REQUIRE_THROWS_AS(make_grid(0.1, {1.0, -1.0, 1.0}), std::invalid_argument);
}

TEST_CASE("make_grid rejects incommensurate box", "[grid]") {
  REQUIRE_THROWS_AS(make_grid(0.3, {1.0, 1.0, 1.0}), std::invalid_argument);
}

TEST_CASE("cell_volume is dx cubed", "[grid]") {
  auto g = make_grid(0.5, {1.0, 1.0, 1.0});
  CHECK(g.cell_volume() == 0.5 * 0.5 * 0.5);
}

TEST_CASE("total_points is product of shape", "[grid]") {
  auto g = make_grid(0.5, {1.0, 2.0, 3.0});
  CHECK(g.total_points() == 2 * 4 * 6);
}

TEST_CASE("flat_index is row-major with z fastest", "[grid]") {
  auto g = make_grid(1.0, {3.0, 4.0, 5.0});
  CHECK(g.flat_index(0, 0, 0) == 0);
  CHECK(g.flat_index(0, 0, 1) == 1);
  CHECK(g.flat_index(0, 1, 0) == 5);
  CHECK(g.flat_index(1, 0, 0) == 20);
  CHECK(g.flat_index(2, 3, 4) == 2 * 20 + 3 * 5 + 4);
}

TEST_CASE("grid supports designated initializer construction", "[grid]") {
  Grid g{.dx = 0.25, .box_size = {1.0, 1.0, 1.0}, .shape = {4, 4, 4}};
  CHECK(g.total_points() == 64);
}

// Face mask

TEST_CASE("face_mask selects only the requested face", "[grid]") {
  auto g = make_grid(1.0, {3.0, 3.0, 3.0});
  auto lower_z = g.face_mask(2, true);
  auto upper_z = g.face_mask(2, false);
  arma::uword lower_count = arma::accu(lower_z);
  arma::uword upper_count = arma::accu(upper_z);
  CHECK(lower_count == 9);
  CHECK(upper_count == 9);
  CHECK(lower_z(static_cast<arma::uword>(g.flat_index(0, 0, 0))) == 1);
  CHECK(lower_z(static_cast<arma::uword>(g.flat_index(1, 1, 2))) == 0);
  CHECK(upper_z(static_cast<arma::uword>(g.flat_index(1, 1, 2))) == 1);
}

// Radial distances

TEST_CASE("radial_distances from center is zero at center", "[grid]") {
  auto g = make_grid(1.0, {4.0, 4.0, 4.0});
  auto r = g.radial_distances({2.0, 2.0, 2.0});
  CHECK(r(static_cast<arma::uword>(g.flat_index(2, 2, 2))) == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("radial_distances from center gives correct diagonal", "[grid]") {
  auto g = make_grid(1.0, {4.0, 4.0, 4.0});
  auto r = g.radial_distances();
  // center is (2,2,2), point (1,1,1) is at distance sqrt(3)
  double expected = std::sqrt(3.0);
  CHECK(r(static_cast<arma::uword>(g.flat_index(1, 1, 1))) == Catch::Approx(expected).margin(1e-12));
}

// Wrap periodic and minimum image

TEST_CASE("wrap_periodic maps into [0, L)", "[grid]") {
  CHECK(Grid::wrap_periodic(-0.5, 3.0) == Catch::Approx(2.5).margin(1e-14));
  CHECK(Grid::wrap_periodic(3.5, 3.0) == Catch::Approx(0.5).margin(1e-14));
  CHECK(Grid::wrap_periodic(1.0, 3.0) == Catch::Approx(1.0).margin(1e-14));
}

TEST_CASE("minimum_image returns shortest distance", "[grid]") {
  CHECK(Grid::minimum_image(2.8, 3.0) == Catch::Approx(-0.2).margin(1e-14));
  CHECK(Grid::minimum_image(-0.2, 3.0) == Catch::Approx(-0.2).margin(1e-14));
  CHECK(Grid::minimum_image(0.5, 3.0) == Catch::Approx(0.5).margin(1e-14));
}

// Plane slices

TEST_CASE("plane_slice produces correct meshgrid dimensions", "[grid]") {
  auto g = make_grid(1.0, {3.0, 4.0, 5.0});
  arma::vec field(static_cast<arma::uword>(g.total_points()), arma::fill::ones);
  auto slice = g.plane_slice(field, {0, 2});
  CHECK(slice.nx == 3);
  CHECK(slice.ny == 5);
  CHECK(slice.x_label == "x");
  CHECK(slice.y_label == "z");
  CHECK(static_cast<long>(slice.z.size()) == slice.ny);
  CHECK(static_cast<long>(slice.z[0].size()) == slice.nx);
}

TEST_CASE("xy_slice picks the center z by default", "[grid]") {
  auto g = make_grid(1.0, {3.0, 3.0, 3.0});
  arma::vec field(static_cast<arma::uword>(g.total_points()), arma::fill::zeros);
  field(static_cast<arma::uword>(g.flat_index(1, 1, 1))) = 42.0;
  auto slice = g.xy_slice(field);
  CHECK(slice.z[1][1] == 42.0);
  CHECK(slice.x_label == "x");
  CHECK(slice.y_label == "y");
}

// Face average

TEST_CASE("face_average computes mean of masked elements", "[grid]") {
  arma::vec field = {1.0, 2.0, 3.0, 4.0};
  arma::uvec mask = {1, 0, 0, 1};
  CHECK(Grid::face_average(field, mask) == Catch::Approx(2.5).margin(1e-14));
}

// Pad / unpad as methods

TEST_CASE("pad and unpad round-trip for non-periodic grid", "[grid]") {
  auto g = make_grid(1.0, {3.0, 3.0, 3.0}, {true, true, false});
  arma::vec field(static_cast<arma::uword>(g.total_points()), arma::fill::randu);
  auto padded = g.pad(field);
  auto recovered = g.unpad(padded);
  CHECK(arma::approx_equal(recovered, field, "absdiff", 1e-14));
}

// Slice1D and line_slice

TEST_CASE("line_slice extracts a 1D profile along each axis", "[grid]") {
  auto g = make_grid(1.0, {4.0, 4.0, 4.0});
  arma::vec field(static_cast<arma::uword>(g.total_points()), arma::fill::zeros);

  long cx = 2, cy = 2, cz = 2;
  field(static_cast<arma::uword>(g.flat_index(cx, cy, cz))) = 7.0;

  auto sx = g.line_slice(field, 0);
  CHECK(sx.axis == "x");
  CHECK(sx.x.size() == 4);
  CHECK(sx.values.size() == 4);
  CHECK(sx.values[static_cast<std::size_t>(cx)] == Catch::Approx(7.0));

  auto sy = g.y_line(field);
  CHECK(sy.axis == "y");
  CHECK(sy.values[static_cast<std::size_t>(cy)] == Catch::Approx(7.0));

  auto sz = g.z_line(field);
  CHECK(sz.axis == "z");
  CHECK(sz.values[static_cast<std::size_t>(cz)] == Catch::Approx(7.0));
}

// RadialProfile and radial_profile

TEST_CASE("radial_profile bins a uniform field correctly", "[grid]") {
  auto g = make_grid(1.0, {6.0, 6.0, 6.0});
  arma::vec field(static_cast<arma::uword>(g.total_points()), arma::fill::value(2.5));

  auto profile = g.radial_profile(field);
  CHECK(!profile.r.empty());
  CHECK(!profile.values.empty());
  for (double v : profile.values) {
    CHECK(v == Catch::Approx(2.5));
  }
}

TEST_CASE("RadialProfile::equimolar_radius finds crossing", "[grid]") {
  RadialProfile profile;
  for (int i = 0; i < 20; ++i) {
    double r = (i + 0.5) * 0.5;
    profile.r.push_back(r);
    profile.values.push_back(r < 3.0 ? 1.0 : 0.0);
  }
  double R = profile.equimolar_radius();
  CHECK(R == Catch::Approx(3.0).margin(0.5));
}

TEST_CASE("RadialProfile::equimolar_radius returns 0 for flat profile", "[grid]") {
  RadialProfile profile;
  for (int i = 0; i < 10; ++i) {
    profile.r.push_back((i + 0.5) * 1.0);
    profile.values.push_back(0.5);
  }
  CHECK(profile.equimolar_radius() == 0.0);
}

// center_value and value_at

TEST_CASE("center_value returns field at box center", "[grid]") {
  auto g = make_grid(1.0, {4.0, 4.0, 4.0});
  arma::vec field(static_cast<arma::uword>(g.total_points()), arma::fill::zeros);
  field(static_cast<arma::uword>(g.flat_index(2, 2, 2))) = 42.0;
  CHECK(g.center_value(field) == Catch::Approx(42.0));
}

TEST_CASE("value_at returns field at nearest grid point", "[grid]") {
  auto g = make_grid(1.0, {4.0, 4.0, 4.0});
  arma::vec field(static_cast<arma::uword>(g.total_points()), arma::fill::zeros);
  field(static_cast<arma::uword>(g.flat_index(1, 2, 3))) = 99.0;
  CHECK(g.value_at(field, {1.0, 2.0, 3.0}) == Catch::Approx(99.0));
  CHECK(g.value_at(field, {1.4, 2.1, 2.6}) == Catch::Approx(99.0));
}
