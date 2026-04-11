#include "dft/fields.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numbers>

using namespace dft;

// StepProfile

TEST_CASE("StepProfile applies step function to distances", "[fields]") {
  arma::vec distances = {0.5, 1.5, 2.5, 3.5, 4.5};
  auto field = StepProfile{.radius = 2.0, .rho_in = 1.0, .rho_out = 0.1}.apply(distances);

  CHECK(field.n_elem == 5);
  CHECK(field(0) == Catch::Approx(1.0));
  CHECK(field(1) == Catch::Approx(1.0));
  CHECK(field(2) == Catch::Approx(0.1));
  CHECK(field(3) == Catch::Approx(0.1));
  CHECK(field(4) == Catch::Approx(0.1));
}

TEST_CASE("StepProfile with zero radius gives all rho_out", "[fields]") {
  arma::vec distances = {0.1, 1.0, 2.0};
  auto field = StepProfile{.radius = 0.0, .rho_in = 5.0, .rho_out = 0.5}.apply(distances);

  for (arma::uword i = 0; i < field.n_elem; ++i) {
    CHECK(field(i) == Catch::Approx(0.5));
  }
}

// EllipsoidalEnvelope

TEST_CASE("EllipsoidalEnvelope is 1 at center and 0 far away", "[fields]") {
  auto grid = make_grid(0.5, {10.0, 10.0, 10.0});
  std::array<double, 3> center = {5.0, 5.0, 5.0};

  auto envelope = EllipsoidalEnvelope{.radii = {3.0, 3.0, 3.0}, .interface_width = 0.25}.apply(grid, center);

  CHECK(envelope.n_elem == static_cast<arma::uword>(grid.total_points()));

  double center_val = grid.center_value(envelope);
  CHECK(center_val == Catch::Approx(1.0).margin(0.01));

  double corner_val = envelope(0);
  CHECK(corner_val < 0.01);
}

TEST_CASE("EllipsoidalEnvelope respects non-periodic axis", "[fields]") {
  auto grid = make_grid(1.0, {6.0, 6.0, 6.0}, {true, true, false});
  std::array<double, 3> center = {3.0, 3.0, 3.0};

  auto envelope = EllipsoidalEnvelope{.radii = {2.0, 2.0, 2.0}, .interface_width = 0.25}.apply(grid, center);

  CHECK(grid.center_value(envelope) == Catch::Approx(1.0).margin(0.01));
}

TEST_CASE("EllipsoidalEnvelope handles ellipsoidal radii", "[fields]") {
  auto grid = make_grid(0.5, {10.0, 10.0, 10.0});
  std::array<double, 3> center = {5.0, 5.0, 5.0};

  auto envelope = EllipsoidalEnvelope{.radii = {4.0, 2.0, 3.0}, .interface_width = 0.25}.apply(grid, center);

  double center_val = grid.center_value(envelope);
  CHECK(center_val == Catch::Approx(1.0).margin(0.01));
}

// rescale_mass

TEST_CASE("rescale_mass adjusts total integral to target", "[fields]") {
  arma::vec field = {1.0, 2.0, 3.0, 4.0};
  double dv = 0.5;
  double target = 10.0;

  auto result = rescale_mass(field, target, dv);
  CHECK(arma::accu(result) * dv == Catch::Approx(target));
}

TEST_CASE("rescale_mass with zero field returns unchanged", "[fields]") {
  arma::vec field(4, arma::fill::zeros);
  auto result = rescale_mass(field, 10.0, 1.0);
  CHECK(arma::accu(result) == Catch::Approx(0.0));
}

// rescale_excess_mass

TEST_CASE("rescale_excess_mass adjusts excess to target", "[fields]") {
  arma::vec field = {0.5, 1.0, 2.0, 0.5};
  arma::vec background = {0.5, 0.5, 0.5, 0.5};
  double dv = 1.0;
  double target_excess = 3.0;

  auto result = rescale_excess_mass(field, background, target_excess, dv);
  arma::vec excess = arma::clamp(result - background, 0.0, arma::datum::inf);
  CHECK(arma::accu(excess) * dv == Catch::Approx(target_excess));
}

TEST_CASE("rescale_excess_mass with no excess returns background", "[fields]") {
  arma::vec field = {0.3, 0.3};
  arma::vec background = {0.5, 0.5};
  auto result = rescale_excess_mass(field, background, 1.0, 1.0);
  CHECK(result(0) >= 0.5 - 1e-10);
  CHECK(result(1) >= 0.5 - 1e-10);
}

// effective_radius

TEST_CASE("effective_radius computes correct sphere radius", "[fields]") {
  double rho_in = 1.0;
  double rho_out = 0.0;
  double R = 2.0;
  double dv = 0.01;

  double volume = (4.0 / 3.0) * std::numbers::pi * R * R * R;
  arma::uword n_in = static_cast<arma::uword>(std::round(volume / dv));
  arma::uword n_out = 1000;
  arma::vec field(n_in + n_out);
  field.head(n_in).fill(rho_in);
  field.tail(n_out).fill(rho_out);

  double delta_rho = rho_in - rho_out;
  double R_eff = effective_radius(field, rho_out, delta_rho, dv);
  CHECK(R_eff == Catch::Approx(R).margin(0.1));
}

TEST_CASE("effective_radius returns 0 for negative excess", "[fields]") {
  arma::vec field = {0.1, 0.1, 0.1};
  CHECK(effective_radius(field, 0.5, 1.0, 1.0) == 0.0);
}
