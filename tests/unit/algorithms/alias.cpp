#include "dft/algorithms/alias.hpp"

#include <armadillo>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::alias;

// Unbounded alias: rho = rho_min + x^2

TEST_CASE("unbounded density_from_alias gives rho_min at zero", "[alias]") {
  Unbounded t{.rho_min = 1e-18};
  arma::vec x = {0.0, 0.0, 0.0};
  auto rho = density_from_alias(x, t);
  CHECK(rho(0) == Catch::Approx(1e-18));
  CHECK(rho(1) == Catch::Approx(1e-18));
}

TEST_CASE("unbounded density_from_alias gives correct values", "[alias]") {
  Unbounded t{.rho_min = 0.0};
  arma::vec x = {1.0, 2.0, 3.0};
  auto rho = density_from_alias(x, t);
  CHECK(rho(0) == Catch::Approx(1.0));
  CHECK(rho(1) == Catch::Approx(4.0));
  CHECK(rho(2) == Catch::Approx(9.0));
}

TEST_CASE("unbounded roundtrip alias -> density -> alias", "[alias]") {
  Unbounded t{.rho_min = 1e-18};
  arma::vec x = {0.5, 1.0, 2.5};
  auto rho = density_from_alias(x, t);
  auto x_back = alias_from_density(rho, t);
  CHECK(x_back(0) == Catch::Approx(x(0)).epsilon(1e-12));
  CHECK(x_back(1) == Catch::Approx(x(1)).epsilon(1e-12));
  CHECK(x_back(2) == Catch::Approx(x(2)).epsilon(1e-12));
}

TEST_CASE("unbounded alias_force applies chain rule correctly", "[alias]") {
  Unbounded t{.rho_min = 0.0};
  arma::vec x = {1.0, 2.0, 3.0};
  arma::vec f_rho = {1.0, 1.0, 1.0};
  auto f_x = alias_force(f_rho, x, t);
  CHECK(f_x(0) == Catch::Approx(2.0));
  CHECK(f_x(1) == Catch::Approx(4.0));
  CHECK(f_x(2) == Catch::Approx(6.0));
}

TEST_CASE("unbounded alias always produces positive density", "[alias]") {
  Unbounded t{.rho_min = 1e-18};
  arma::vec x = {-3.0, 0.0, 100.0};
  auto rho = density_from_alias(x, t);
  CHECK(arma::all(rho > 0.0));
}

// Bounded alias: rho = rho_min + range * x^2 / (1 + x^2)

TEST_CASE("bounded density_from_alias gives rho_min at zero", "[alias]") {
  Bounded t{.rho_min = 1e-18, .rho_max = 1.0};
  arma::vec x = {0.0};
  auto rho = density_from_alias(x, t);
  CHECK(rho(0) == Catch::Approx(1e-18));
}

TEST_CASE("bounded density_from_alias saturates below rho_max", "[alias]") {
  Bounded t{.rho_min = 0.0, .rho_max = 1.0};
  arma::vec x = {1e6};
  auto rho = density_from_alias(x, t);
  CHECK(rho(0) == Catch::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("bounded roundtrip alias -> density -> alias", "[alias]") {
  Bounded t{.rho_min = 1e-18, .rho_max = 0.9};
  arma::vec x = {0.1, 0.5, 1.0, 5.0};
  auto rho = density_from_alias(x, t);
  auto x_back = alias_from_density(rho, t);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    CHECK(x_back(i) == Catch::Approx(x(i)).epsilon(1e-10));
  }
}

TEST_CASE("bounded alias_force applies chain rule correctly", "[alias]") {
  Bounded t{.rho_min = 0.0, .rho_max = 1.0};
  arma::vec x = {1.0};
  arma::vec f_rho = {1.0};
  auto f_x = alias_force(f_rho, x, t);
  // drho/dx = 2 * range * x / (1 + x^2)^2 = 2 * 1.0 * 1.0 / 4.0 = 0.5
  CHECK(f_x(0) == Catch::Approx(0.5));
}

TEST_CASE("bounded density stays within bounds for any alias value", "[alias]") {
  Bounded t{.rho_min = 0.01, .rho_max = 0.95};
  arma::vec x = {-100.0, -1.0, 0.0, 1.0, 100.0};
  auto rho = density_from_alias(x, t);
  CHECK(arma::all(rho >= t.rho_min));
  CHECK(arma::all(rho <= t.rho_max));
}

// AliasTransform variant dispatch

TEST_CASE("variant dispatch works for unbounded", "[alias]") {
  AliasTransform t = Unbounded{.rho_min = 0.0};
  arma::vec x = {2.0};
  auto rho = density_from_alias(x, t);
  CHECK(rho(0) == Catch::Approx(4.0));
}

TEST_CASE("variant dispatch works for bounded", "[alias]") {
  AliasTransform t = Bounded{.rho_min = 0.0, .rho_max = 1.0};
  arma::vec x = {1.0};
  auto rho = density_from_alias(x, t);
  CHECK(rho(0) == Catch::Approx(0.5));
}
