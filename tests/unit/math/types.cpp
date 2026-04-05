#include "dft/math/types.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::math;

TEST_CASE("HessianOperator stores dimension", "[hessian]") {
  HessianOperator op{
    .hessian_dot_v = [](const arma::vec&) -> arma::vec { return {}; },
    .dimension = 10,
  };
  CHECK(op.dimension == 10);
}

TEST_CASE("HessianOperator applies identity", "[hessian]") {
  HessianOperator op{
    .hessian_dot_v = [](const arma::vec& v) -> arma::vec { return v; },
    .dimension = 3,
  };
  arma::vec x = { 1.0, 2.0, 3.0 };
  arma::vec result = op.hessian_dot_v(x);
  CHECK(result(0) == Catch::Approx(1.0));
  CHECK(result(1) == Catch::Approx(2.0));
  CHECK(result(2) == Catch::Approx(3.0));
}

TEST_CASE("HessianOperator applies scaling", "[hessian]") {
  HessianOperator op{
    .hessian_dot_v = [](const arma::vec& v) -> arma::vec { return 2.0 * v; },
    .dimension = 2,
  };
  arma::vec x = { 3.0, 4.0 };
  arma::vec result = op.hessian_dot_v(x);
  CHECK(result(0) == Catch::Approx(6.0));
  CHECK(result(1) == Catch::Approx(8.0));
}

TEST_CASE("HessianOperator applies matrix-vector product", "[hessian]") {
  arma::mat H = { { 2.0, 1.0 }, { 1.0, 3.0 } };
  HessianOperator op{
    .hessian_dot_v = [&H](const arma::vec& v) -> arma::vec { return H * v; },
    .dimension = 2,
  };
  arma::vec x = { 1.0, 1.0 };
  arma::vec result = op.hessian_dot_v(x);
  CHECK(result(0) == Catch::Approx(3.0));
  CHECK(result(1) == Catch::Approx(4.0));
}
