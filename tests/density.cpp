#include "dft/types.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace dft;

TEST_CASE("density default-constructs with empty vectors", "[density]") {
  Density d;
  CHECK(d.values.is_empty());
  CHECK(d.external_field.is_empty());
}

TEST_CASE("density values are directly assignable", "[density]") {
  Density d;
  d.values = arma::vec{1.0, 2.0, 3.0};
  CHECK(d.values.n_elem == 3);
  CHECK(d.values(0) == 1.0);
  CHECK(d.values(2) == 3.0);
}

TEST_CASE("density supports designated initializer construction", "[density]") {
  Density d{
      .values = arma::vec(100, arma::fill::ones),
      .external_field = arma::vec(100, arma::fill::zeros),
  };
  CHECK(d.values.n_elem == 100);
  CHECK(d.external_field.n_elem == 100);
}

TEST_CASE("density is copyable", "[density]") {
  Density original{
      .values = arma::vec{1.0, 2.0, 3.0},
      .external_field = arma::vec{0.1, 0.2, 0.3},
  };
  Density copy = original;
  CHECK(copy.values.n_elem == 3);
  CHECK(copy.values(1) == 2.0);

  // Modifying copy does not affect original
  copy.values(0) = 99.0;
  CHECK(original.values(0) == 1.0);
}

TEST_CASE("density is movable", "[density]") {
  Density original{
      .values = arma::vec{1.0, 2.0, 3.0},
      .external_field = arma::vec(3, arma::fill::zeros),
  };
  Density moved = std::move(original);
  CHECK(moved.values.n_elem == 3);
  CHECK(moved.values(2) == 3.0);
}
