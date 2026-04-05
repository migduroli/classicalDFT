#include "dft/algorithms/solvers/jacobian.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::solvers;

TEST_CASE("numerical jacobian of linear function is the matrix itself", "[jacobian]") {
  // f(x) = A*x, so J = A everywhere
  arma::mat A = {{2.0, -1.0}, {0.0, 3.0}};

  auto f = [&](const arma::vec& x) -> arma::vec {
    return A * x;
  };

  arma::vec x = {1.0, 2.0};
  auto J = numerical_jacobian(f, x);

  REQUIRE(J.n_rows == 2);
  REQUIRE(J.n_cols == 2);
  CHECK(J(0, 0) == Catch::Approx(2.0).margin(1e-8));
  CHECK(J(0, 1) == Catch::Approx(-1.0).margin(1e-8));
  CHECK(J(1, 0) == Catch::Approx(0.0).margin(1e-8));
  CHECK(J(1, 1) == Catch::Approx(3.0).margin(1e-8));
}

TEST_CASE("numerical jacobian of nonlinear function", "[jacobian]") {
  // f(x) = [x0^2 + x1, x0*x1]
  // J = [[2*x0, 1], [x1, x0]]
  auto f = [](const arma::vec& x) -> arma::vec {
    return arma::vec{x(0) * x(0) + x(1), x(0) * x(1)};
  };

  arma::vec x = {3.0, 4.0};
  auto J = numerical_jacobian(f, x);

  CHECK(J(0, 0) == Catch::Approx(6.0).margin(1e-6));
  CHECK(J(0, 1) == Catch::Approx(1.0).margin(1e-6));
  CHECK(J(1, 0) == Catch::Approx(4.0).margin(1e-6));
  CHECK(J(1, 1) == Catch::Approx(3.0).margin(1e-6));
}

TEST_CASE("numerical jacobian of scalar-valued vector function", "[jacobian]") {
  // f: R^3 -> R^1, f(x) = [x0 + 2*x1 + 3*x2]
  // J = [1, 2, 3]
  auto f = [](const arma::vec& x) -> arma::vec {
    return arma::vec{x(0) + 2.0 * x(1) + 3.0 * x(2)};
  };

  arma::vec x = {0.0, 0.0, 0.0};
  auto J = numerical_jacobian(f, x);

  REQUIRE(J.n_rows == 1);
  REQUIRE(J.n_cols == 3);
  CHECK(J(0, 0) == Catch::Approx(1.0).margin(1e-8));
  CHECK(J(0, 1) == Catch::Approx(2.0).margin(1e-8));
  CHECK(J(0, 2) == Catch::Approx(3.0).margin(1e-8));
}

TEST_CASE("numerical jacobian with custom epsilon", "[jacobian]") {
  auto f = [](const arma::vec& x) -> arma::vec {
    return arma::vec{x(0) * x(0)};
  };

  arma::vec x = {2.0};

  // Coarse epsilon
  auto J_coarse = numerical_jacobian(f, x, 1e-3);
  CHECK(J_coarse(0, 0) == Catch::Approx(4.0).margin(1e-4));

  // Fine epsilon
  auto J_fine = numerical_jacobian(f, x, 1e-8);
  CHECK(J_fine(0, 0) == Catch::Approx(4.0).margin(1e-8));
}
