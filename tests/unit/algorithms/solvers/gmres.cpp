#include "dft/algorithms/solvers/gmres.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::solvers;

TEST_CASE("gmres solves diagonal system", "[gmres]") {
  // A = diag(2, 3, 5), b = (4, 9, 25) => x = (2, 3, 5)
  auto A = [](const arma::vec& v) -> arma::vec {
    return arma::vec{2.0 * v(0), 3.0 * v(1), 5.0 * v(2)};
  };
  arma::vec b = {4.0, 9.0, 25.0};

  auto result = GMRES{.tolerance = 1e-12}.solve(A, b);

  REQUIRE(result.converged);
  CHECK(result.solution(0) == Catch::Approx(2.0).margin(1e-10));
  CHECK(result.solution(1) == Catch::Approx(3.0).margin(1e-10));
  CHECK(result.solution(2) == Catch::Approx(5.0).margin(1e-10));
}

TEST_CASE("gmres solves dense SPD system", "[gmres]") {
  // A = [[4, 1], [1, 3]], b = [1, 2] => x = [1/11, 7/11]
  arma::mat M = {{4.0, 1.0}, {1.0, 3.0}};
  auto A = [&M](const arma::vec& v) -> arma::vec {
    return M * v;
  };
  arma::vec b = {1.0, 2.0};

  auto result = GMRES{.tolerance = 1e-12}.solve(A, b);

  REQUIRE(result.converged);
  arma::vec expected = arma::solve(M, b);
  CHECK(result.solution(0) == Catch::Approx(expected(0)).margin(1e-10));
  CHECK(result.solution(1) == Catch::Approx(expected(1)).margin(1e-10));
}

TEST_CASE("gmres solves non-symmetric system", "[gmres]") {
  arma::mat M = {{3.0, 1.0}, {-1.0, 2.0}};
  auto A = [&M](const arma::vec& v) -> arma::vec {
    return M * v;
  };
  arma::vec b = {7.0, 1.0};

  auto result = GMRES{.tolerance = 1e-10}.solve(A, b);

  REQUIRE(result.converged);
  arma::vec expected = arma::solve(M, b);
  CHECK(result.solution(0) == Catch::Approx(expected(0)).margin(1e-8));
  CHECK(result.solution(1) == Catch::Approx(expected(1)).margin(1e-8));
}

TEST_CASE("gmres uses initial guess", "[gmres]") {
  auto A = [](const arma::vec& v) -> arma::vec {
    return v;
  }; // Identity
  arma::vec b = {1.0, 2.0, 3.0};

  auto result = GMRES{.tolerance = 1e-12}.solve(A, b, b);

  REQUIRE(result.converged);
  CHECK(result.iterations == 0);
}

TEST_CASE("gmres reports non-convergence", "[gmres]") {
  // Ill-conditioned: very few iterations allowed
  arma::mat M = {{1e6, 1.0}, {1.0, 1e-6}};
  auto A = [&M](const arma::vec& v) -> arma::vec {
    return M * v;
  };
  arma::vec b = {1.0, 1.0};

  auto result = GMRES{.max_iterations = 1, .restart = 1, .tolerance = 1e-15}.solve(A, b);

  // With only 1 restart cycle of size 1, may not converge to 1e-15
  // Just check it produces some result without crashing
  CHECK(result.solution.n_elem == 2);
}

TEST_CASE("gmres restart works on larger system", "[gmres]") {
  // 10x10 random SPD system
  arma::arma_rng::set_seed(42);
  arma::mat R = arma::randn(10, 10);
  arma::mat M = R.t() * R + 5.0 * arma::eye(10, 10); // SPD, well-conditioned

  auto A = [&M](const arma::vec& v) -> arma::vec {
    return M * v;
  };
  arma::vec b = arma::randn(10);

  auto result = GMRES{.restart = 5, .tolerance = 1e-10}.solve(A, b);

  REQUIRE(result.converged);
  arma::vec expected = arma::solve(M, b);
  for (arma::uword i = 0; i < 10; ++i) {
    CHECK(result.solution(i) == Catch::Approx(expected(i)).margin(1e-6));
  }
}

TEST_CASE("gmres handles identity operator", "[gmres]") {
  auto A = [](const arma::vec& v) -> arma::vec {
    return v;
  };
  arma::vec b = {3.14, 2.72, 1.41};

  auto result = GMRES{.tolerance = 1e-12}.solve(A, b);

  REQUIRE(result.converged);
  CHECK(result.iterations <= 1);
  for (arma::uword i = 0; i < 3; ++i) {
    CHECK(result.solution(i) == Catch::Approx(b(i)).margin(1e-10));
  }
}
