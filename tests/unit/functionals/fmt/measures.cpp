#include "dft/functionals/fmt/measures.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <numbers>

using namespace dft::functionals::fmt;

// Default construction

TEST_CASE("measures default construction is all zero", "[fmt][measures]") {
  Measures m;
  CHECK(m.eta == 0.0);
  CHECK(m.n0 == 0.0);
  CHECK(m.n1 == 0.0);
  CHECK(m.n2 == 0.0);
  CHECK(arma::norm(m.v0) == 0.0);
  CHECK(arma::norm(m.v1) == 0.0);
  CHECK(arma::accu(arma::abs(m.T)) == 0.0);
}

TEST_CASE("inner products default measures is all zero", "[fmt][measures]") {
  Measures m;
  auto p = m.inner_products();
  CHECK(p.dot_v0_v1 == 0.0);
  CHECK(p.dot_v1_v1 == 0.0);
  CHECK(p.quadratic_form == 0.0);
  CHECK(p.trace_T2 == 0.0);
  CHECK(p.trace_T3 == 0.0);
}

// inner_products method

TEST_CASE("inner products with nonzero vectors", "[fmt][measures]") {
  Measures m;
  m.v0 = { 1.0, 2.0, 3.0 };
  m.v1 = { 4.0, 5.0, 6.0 };
  m.T.eye();
  auto p = m.inner_products();

  CHECK(p.dot_v0_v1 == Catch::Approx(1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0));
  CHECK(p.dot_v1_v1 == Catch::Approx(4.0 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0));
  CHECK(p.trace_T2 == Catch::Approx(3.0));
  CHECK(p.trace_T3 == Catch::Approx(3.0));
  CHECK(p.quadratic_form == Catch::Approx(p.dot_v1_v1));
}

// make_uniform_measures factory

TEST_CASE("uniform measures eta is consistent", "[fmt][measures]") {
  double rho = 0.8;
  double d = 1.0;
  auto m = make_uniform_measures(rho, d);
  CHECK(m.eta == Catch::Approx(std::numbers::pi / 6.0 * rho).margin(1e-14));
}

TEST_CASE("uniform measures scalars follow Rosenfeld relations", "[fmt][measures]") {
  double rho = 0.6;
  double d = 1.2;
  auto m = make_uniform_measures(rho, d);

  CHECK(m.n2 == Catch::Approx(std::numbers::pi * rho * d * d).margin(1e-14));
  CHECK(m.n1 == Catch::Approx(0.5 * d * rho).margin(1e-14));
  CHECK(m.n0 == Catch::Approx(rho).margin(1e-14));
  CHECK(m.n0 == Catch::Approx(m.n2 / (std::numbers::pi * d * d)).margin(1e-14));
  CHECK(m.n1 == Catch::Approx(m.n2 / (2.0 * std::numbers::pi * d)).margin(1e-14));
}

TEST_CASE("uniform measures vectors vanish", "[fmt][measures]") {
  auto m = make_uniform_measures(0.7, 1.0);
  CHECK(arma::norm(m.v0) == 0.0);
  CHECK(arma::norm(m.v1) == 0.0);
}

TEST_CASE("uniform measures tensor is diagonal", "[fmt][measures]") {
  double rho = 0.5;
  double d = 1.3;
  auto m = make_uniform_measures(rho, d);

  double t_diag = m.n2 / 3.0;
  CHECK(m.T(0, 0) == Catch::Approx(t_diag).margin(1e-14));
  CHECK(m.T(1, 1) == Catch::Approx(t_diag).margin(1e-14));
  CHECK(m.T(2, 2) == Catch::Approx(t_diag).margin(1e-14));
  CHECK(m.T(0, 1) == 0.0);
  CHECK(m.T(0, 2) == 0.0);
  CHECK(m.T(1, 2) == 0.0);
}

TEST_CASE("uniform measures inner products are pre-computed", "[fmt][measures]") {
  auto m = make_uniform_measures(0.5, 1.0);

  // vectors are zero so dot products should vanish
  CHECK(m.products.dot_v0_v1 == 0.0);
  CHECK(m.products.dot_v1_v1 == 0.0);
  CHECK(m.products.quadratic_form == 0.0);

  // tensor trace products should be non-zero
  double t_diag = m.n2 / 3.0;
  double expected_trace_T2 = 3.0 * t_diag * t_diag;
  CHECK(m.products.trace_T2 == Catch::Approx(expected_trace_T2).margin(1e-14));
}
