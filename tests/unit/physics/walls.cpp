#include "dft/physics/walls.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;
using namespace dft::physics::walls;

// LJ93 concrete type

TEST_CASE("LJ93 energy at sigma is negative", "[walls]") {
  LJ93 lj{.sigma = 1.0, .epsilon = 1.0, .density = 1.0, .cutoff = 5.0};
  double v = lj.energy(1.0);
  CHECK(v < 0.0);
}

TEST_CASE("LJ93 energy diverges at small distance", "[walls]") {
  LJ93 lj{.sigma = 1.0, .epsilon = 1.0, .density = 1.0, .cutoff = 5.0};
  double v_close = lj.energy(0.1);
  double v_far = lj.energy(2.0);
  CHECK(v_close > v_far);
}

TEST_CASE("LJ93 attachment distance is repulsive minimum", "[walls]") {
  LJ93 lj{.sigma = 1.0, .epsilon = 1.0, .density = 1.0};
  double d = lj.attachment_distance();
  CHECK(d == Catch::Approx(std::pow(2.0 / 5.0, 1.0 / 6.0)).margin(1e-12));
}

TEST_CASE("LJ93 field has correct sign at wall contact", "[walls]") {
  auto grid = make_grid(0.5, {5.0, 5.0, 5.0}, {true, true, false});
  LJ93 lj{.sigma = 1.0, .epsilon = 1.0, .density = 1.0, .cutoff = 4.0};
  auto v = lj.field(grid, 2, true);
  // Near z=0 the potential is repulsive (positive)
  CHECK(v(0) > 0.0);
}

// WallPotential wrapper

TEST_CASE("default WallPotential is inactive", "[walls]") {
  WallPotential wall;
  CHECK_FALSE(wall.is_active());
}

TEST_CASE("WallPotential wrapping LJ93 is active", "[walls]") {
  WallPotential wall(LJ93{.sigma = 1.0, .epsilon = 1.0, .density = 1.0, .cutoff = 5.0}, 2, true);
  CHECK(wall.is_active());
  CHECK(wall.axis == 2);
  CHECK(wall.lower == true);
}

TEST_CASE("WallPotential energy delegates to LJ93", "[walls]") {
  LJ93 lj{.sigma = 1.0, .epsilon = 1.0, .density = 1.0};
  WallPotential wall(lj, 2, true);
  CHECK(wall.energy(1.5) == Catch::Approx(lj.energy(1.5)).margin(1e-14));
}

TEST_CASE("inactive WallPotential energy is zero", "[walls]") {
  WallPotential wall;
  CHECK(wall.energy(1.0) == 0.0);
}

TEST_CASE("WallPotential field has same size as grid", "[walls]") {
  auto grid = make_grid(0.5, {3.0, 3.0, 3.0}, {true, true, false});
  WallPotential wall(LJ93{.sigma = 1.0, .epsilon = 1.0, .density = 1.0, .cutoff = 2.5}, 2, true);
  auto v = wall.field(grid);
  CHECK(static_cast<long>(v.n_elem) == grid.total_points());
}

TEST_CASE("inactive WallPotential field is zero", "[walls]") {
  auto grid = make_grid(0.5, {3.0, 3.0, 3.0});
  WallPotential wall;
  auto v = wall.field(grid);
  CHECK(arma::accu(arma::abs(v)) == 0.0);
}

TEST_CASE("WallPotential reservoir_mask uses far face when active", "[walls]") {
  auto grid = make_grid(1.0, {3.0, 3.0, 3.0}, {true, true, false});
  WallPotential wall(LJ93{.sigma = 1.0, .epsilon = 1.0, .density = 1.0, .cutoff = 2.5}, 2, true);
  auto mask = wall.reservoir_mask(grid);
  // Lower wall → reservoir is upper face (z=2)
  CHECK(mask(static_cast<arma::uword>(grid.flat_index(1, 1, 2))) == 1);
  CHECK(mask(static_cast<arma::uword>(grid.flat_index(1, 1, 0))) == 0);
}

// Gravity

TEST_CASE("gravity with zero strength is inactive", "[walls]") {
  physics::Gravity g{.strength = 0.0};
  CHECK_FALSE(g.is_active());
}

TEST_CASE("gravity field is linear in z", "[walls]") {
  auto grid = make_grid(1.0, {2.0, 2.0, 4.0}, {true, true, false});
  physics::Gravity g{.strength = 0.5};
  WallPotential wall;
  auto v = g.field(grid, wall);
  // z=0 → V=0.5*0.5=0.25, z=1 → V=0.5*1.5=0.75
  CHECK(v(static_cast<arma::uword>(grid.flat_index(0, 0, 0))) == Catch::Approx(0.25).margin(1e-14));
  CHECK(v(static_cast<arma::uword>(grid.flat_index(0, 0, 1))) == Catch::Approx(0.75).margin(1e-14));
}

// WallPotential::suppress_excess

TEST_CASE("suppress_excess replaces density in repulsive region", "[walls]") {
  auto grid = make_grid(0.5, {3.0, 3.0, 3.0}, {true, true, false});
  WallPotential wall(LJ93{.sigma = 1.0, .epsilon = 1.0, .density = 1.0, .cutoff = 2.5}, 2, true);
  auto n = static_cast<arma::uword>(grid.total_points());
  arma::vec rho(n, arma::fill::value(0.8));
  arma::vec background(n, arma::fill::value(0.01));

  auto result = wall.suppress_excess(rho, background, grid);

  auto wf = wall.field(grid);
  arma::uvec repulsive = arma::find(wf > 0.0);
  arma::uvec non_repulsive = arma::find(wf <= 0.0);

  // Repulsive regions should have background values
  for (auto idx : repulsive) {
    CHECK(result(idx) == Catch::Approx(background(idx)));
  }
  // Non-repulsive regions should keep original density
  for (auto idx : non_repulsive) {
    CHECK(result(idx) == Catch::Approx(rho(idx)));
  }
}

TEST_CASE("suppress_excess is no-op for inactive wall", "[walls]") {
  auto grid = make_grid(0.5, {3.0, 3.0, 3.0});
  WallPotential wall;
  auto n = static_cast<arma::uword>(grid.total_points());
  arma::vec rho(n, arma::fill::value(0.8));
  arma::vec background(n, arma::fill::value(0.01));

  auto result = wall.suppress_excess(rho, background, grid);
  CHECK(arma::approx_equal(result, rho, "absdiff", 1e-15));
}
