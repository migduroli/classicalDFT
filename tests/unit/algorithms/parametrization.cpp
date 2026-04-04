#include "dft/algorithms/parametrization.hpp"

#include "dft/grid.hpp"

#include <armadillo>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::parametrization;

// Unbounded parametrization: rho = rho_min + x^2

TEST_CASE("unbounded to_density gives rho_min at zero", "[parametrization]") {
  Unbounded t{.rho_min = 1e-18};
  arma::vec x = {0.0, 0.0, 0.0};
  auto rho = to_density(x, t);
  CHECK(rho(0) == Catch::Approx(1e-18));
  CHECK(rho(1) == Catch::Approx(1e-18));
}

TEST_CASE("unbounded to_density gives correct values", "[parametrization]") {
  Unbounded t{.rho_min = 0.0};
  arma::vec x = {1.0, 2.0, 3.0};
  auto rho = to_density(x, t);
  CHECK(rho(0) == Catch::Approx(1.0));
  CHECK(rho(1) == Catch::Approx(4.0));
  CHECK(rho(2) == Catch::Approx(9.0));
}

TEST_CASE("unbounded roundtrip to_density -> from_density", "[parametrization]") {
  Unbounded t{.rho_min = 1e-18};
  arma::vec x = {0.5, 1.0, 2.5};
  auto rho = to_density(x, t);
  auto x_back = from_density(rho, t);
  CHECK(x_back(0) == Catch::Approx(x(0)).epsilon(1e-12));
  CHECK(x_back(1) == Catch::Approx(x(1)).epsilon(1e-12));
  CHECK(x_back(2) == Catch::Approx(x(2)).epsilon(1e-12));
}

TEST_CASE("unbounded transform_force applies chain rule correctly", "[parametrization]") {
  Unbounded t{.rho_min = 0.0};
  arma::vec x = {1.0, 2.0, 3.0};
  arma::vec f_rho = {1.0, 1.0, 1.0};
  auto f_x = transform_force(f_rho, x, t);
  CHECK(f_x(0) == Catch::Approx(2.0));
  CHECK(f_x(1) == Catch::Approx(4.0));
  CHECK(f_x(2) == Catch::Approx(6.0));
}

TEST_CASE("unbounded parametrization always produces positive density", "[parametrization]") {
  Unbounded t{.rho_min = 1e-18};
  arma::vec x = {-3.0, 0.0, 100.0};
  auto rho = to_density(x, t);
  CHECK(arma::all(rho > 0.0));
}

// Bounded parametrization: rho = rho_min + range * x^2 / (1 + x^2)

TEST_CASE("bounded to_density gives rho_min at zero", "[parametrization]") {
  Bounded t{.rho_min = 1e-18, .rho_max = 1.0};
  arma::vec x = {0.0};
  auto rho = to_density(x, t);
  CHECK(rho(0) == Catch::Approx(1e-18));
}

TEST_CASE("bounded to_density saturates below rho_max", "[parametrization]") {
  Bounded t{.rho_min = 0.0, .rho_max = 1.0};
  arma::vec x = {1e6};
  auto rho = to_density(x, t);
  CHECK(rho(0) == Catch::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("bounded roundtrip to_density -> from_density", "[parametrization]") {
  Bounded t{.rho_min = 1e-18, .rho_max = 0.9};
  arma::vec x = {0.1, 0.5, 1.0, 5.0};
  auto rho = to_density(x, t);
  auto x_back = from_density(rho, t);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    CHECK(x_back(i) == Catch::Approx(x(i)).epsilon(1e-10));
  }
}

// Parametrization variant dispatch

TEST_CASE("variant dispatches to_density correctly", "[parametrization]") {
  Parametrization p = Unbounded{.rho_min = 0.0};
  arma::vec x = {2.0};
  auto rho = to_density(x, p);
  CHECK(rho(0) == Catch::Approx(4.0));
}

TEST_CASE("variant dispatches from_density correctly", "[parametrization]") {
  Parametrization p = Bounded{.rho_min = 0.0, .rho_max = 1.0};
  arma::vec rho = {0.5};
  auto x = from_density(rho, p);
  auto rho_back = to_density(x, p);
  CHECK(rho_back(0) == Catch::Approx(0.5).epsilon(1e-12));
}

TEST_CASE("bounded transform_force applies chain rule correctly", "[parametrization]") {
  Bounded t{.rho_min = 0.0, .rho_max = 1.0};
  arma::vec x = {1.0, 2.0};
  arma::vec f_rho = {1.0, 1.0};
  auto f_x = transform_force(f_rho, x, Parametrization{t});
  // drho/dx = 2 * range * x / (1 + x^2)^2
  double range = 1.0;
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    double x2 = x(i) * x(i);
    double expected = 2.0 * range * x(i) / ((1.0 + x2) * (1.0 + x2));
    CHECK(f_x(i) == Catch::Approx(expected).epsilon(1e-12));
  }
}

// Boundary condition tests

TEST_CASE("boundary_mask identifies face points", "[grid][boundary]") {
  auto grid = dft::make_grid(1.0, {3.0, 3.0, 3.0});
  auto mask = dft::boundary_mask(grid);
  // 3x3x3 = 27 points, interior is 1x1x1 = 1 point
  arma::uword n_boundary = arma::accu(mask);
  CHECK(n_boundary == 26);
}

TEST_CASE("homogeneous_boundary averages face forces", "[grid][boundary]") {
  auto grid = dft::make_grid(1.0, {3.0, 3.0, 3.0});
  auto mask = dft::boundary_mask(grid);
  arma::vec forces(27, arma::fill::zeros);
  // Set boundary forces to different values
  for (arma::uword i = 0; i < 27; ++i) {
    forces(i) = static_cast<double>(i);
  }
  auto result = dft::homogeneous_boundary(forces, mask);
  // Interior point (1,1,1) = index 13 should be unchanged
  CHECK(result(13) == Catch::Approx(13.0));
  // All boundary points should have the same value
  double boundary_val = result(0);
  for (arma::uword i = 0; i < 27; ++i) {
    if (mask(i)) {
      CHECK(result(i) == Catch::Approx(boundary_val).epsilon(1e-12));
    }
  }
}

TEST_CASE("fixed_boundary zeros face forces", "[grid][boundary]") {
  auto grid = dft::make_grid(1.0, {3.0, 3.0, 3.0});
  auto mask = dft::boundary_mask(grid);
  arma::vec forces(27, arma::fill::ones);
  auto result = dft::fixed_boundary(forces, mask);
  // Interior point should be 1
  CHECK(result(13) == Catch::Approx(1.0));
  // Boundary points should be 0
  for (arma::uword i = 0; i < 27; ++i) {
    if (mask(i)) {
      CHECK(result(i) == Catch::Approx(0.0));
    }
  }
}
