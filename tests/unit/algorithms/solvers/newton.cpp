#include "dft/algorithms/solvers/newton.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::solvers;

TEST_CASE("newton solves linear system with analytical jacobian", "[newton]") {
  // f(x) = A*x - b, root at x = A^{-1} b
  arma::mat A = {{2.0, 1.0}, {1.0, 3.0}};
  arma::vec b = {5.0, 7.0};

  auto f = [&](const arma::vec& x) -> arma::vec { return A * x - b; };
  auto J = [&](const arma::vec&) -> arma::mat { return A; };

  auto result = Newton{.tolerance = 1e-12}.solve(arma::vec{0.0, 0.0}, f, J);

  REQUIRE(result.converged);
  CHECK(result.iterations <= 2);

  arma::vec expected = arma::solve(A, b);
  CHECK(result.solution(0) == Catch::Approx(expected(0)).margin(1e-12));
  CHECK(result.solution(1) == Catch::Approx(expected(1)).margin(1e-12));
}

TEST_CASE("newton solves nonlinear system with analytical jacobian", "[newton]") {
  // f(x) = [x0^2 - 4, x1^2 - 9], roots at (2, 3) or (-2, -3) etc.
  auto f = [](const arma::vec& x) -> arma::vec { return arma::vec{x(0) * x(0) - 4.0, x(1) * x(1) - 9.0}; };

  auto J = [](const arma::vec& x) -> arma::mat { return arma::mat{{2.0 * x(0), 0.0}, {0.0, 2.0 * x(1)}}; };

  auto result = Newton{.tolerance = 1e-12}.solve(arma::vec{1.0, 2.0}, f, J);

  REQUIRE(result.converged);
  CHECK(result.solution(0) == Catch::Approx(2.0).margin(1e-10));
  CHECK(result.solution(1) == Catch::Approx(3.0).margin(1e-10));
}

TEST_CASE("newton solves with automatic numerical jacobian", "[newton]") {
  // Same nonlinear system, no analytical Jacobian provided
  auto f = [](const arma::vec& x) -> arma::vec { return arma::vec{x(0) * x(0) - 4.0, x(1) * x(1) - 9.0}; };

  auto result = Newton{.tolerance = 1e-10}.solve(arma::vec{1.0, 2.0}, f);

  REQUIRE(result.converged);
  CHECK(result.solution(0) == Catch::Approx(2.0).margin(1e-8));
  CHECK(result.solution(1) == Catch::Approx(3.0).margin(1e-8));
}

TEST_CASE("newton reports non-convergence when max iterations exceeded", "[newton]") {
  // f(x) = [exp(x0) - 1], start far away with very few iterations
  auto f = [](const arma::vec& x) -> arma::vec { return arma::vec{std::exp(x(0)) - 1.0}; };

  auto J = [](const arma::vec& x) -> arma::mat {
    arma::mat m(1, 1);
    m(0, 0) = std::exp(x(0));
    return m;
  };

  auto result = Newton{.max_iterations = 3, .tolerance = 1e-15}.solve(arma::vec{50.0}, f, J);

  CHECK(!result.converged);
  CHECK(result.iterations == 3);
}

TEST_CASE("newton solves scalar equation", "[newton]") {
  // f(x) = x^3 - 8, root at x = 2
  auto f = [](const arma::vec& x) -> arma::vec { return arma::vec{x(0) * x(0) * x(0) - 8.0}; };

  auto J = [](const arma::vec& x) -> arma::mat {
    arma::mat m(1, 1);
    m(0, 0) = 3.0 * x(0) * x(0);
    return m;
  };

  auto result = Newton{.tolerance = 1e-12}.solve(arma::vec{1.0}, f, J);

  REQUIRE(result.converged);
  CHECK(result.solution(0) == Catch::Approx(2.0).margin(1e-10));
}

TEST_CASE("newton converges in one step for linear system", "[newton]") {
  arma::mat A = {{1.0, 0.0}, {0.0, 1.0}};
  arma::vec b = {3.0, 4.0};

  auto f = [&](const arma::vec& x) -> arma::vec { return A * x - b; };
  auto J = [&](const arma::vec&) -> arma::mat { return A; };

  auto result = Newton{.tolerance = 1e-14}.solve(arma::vec{0.0, 0.0}, f, J);

  REQUIRE(result.converged);
  CHECK(result.iterations == 1);
  CHECK(result.solution(0) == Catch::Approx(3.0).margin(1e-14));
  CHECK(result.solution(1) == Catch::Approx(4.0).margin(1e-14));
}

TEST_CASE("newton verbose mode prints iteration info", "[newton]") {
  auto f = [](const arma::vec& x) -> arma::vec { return arma::vec{x(0) - 1.0}; };
  auto J = [](const arma::vec&) -> arma::mat {
    arma::mat m(1, 1);
    m(0, 0) = 1.0;
    return m;
  };

  auto result = Newton{.tolerance = 1e-12, .verbose = true}.solve(arma::vec{0.0}, f, J);

  REQUIRE(result.converged);
}

TEST_CASE("newton returns non-converged for singular jacobian", "[newton]") {
  auto f = [](const arma::vec& x) -> arma::vec { return arma::vec{x(0) * x(0) + 1.0}; };
  auto J = [](const arma::vec&) -> arma::mat {
    arma::mat m(1, 1);
    m(0, 0) = 0.0;
    return m;
  };

  auto result = Newton{.max_iterations = 5, .tolerance = 1e-12}.solve(arma::vec{1.0}, f, J);

  CHECK(!result.converged);
}
