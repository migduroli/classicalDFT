#include "dft/algorithms/solvers/continuation.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::continuation;
using namespace dft::algorithms::solvers;

// Unit circle: x^2 + lambda^2 - 1 = 0
// The curve is the unit circle parametrized by arclength.
static auto circle_residual(const arma::vec& x, double lambda) -> arma::vec {
  return arma::vec{x(0) * x(0) + lambda * lambda - 1.0};
}

TEST_CASE("continuation step advances along unit circle", "[continuation]") {
  // Start at (x=0, lambda=1) with tangent along +x direction
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  ContinuationConfig config{
      .initial_step = 0.1,
      .newton = {.max_iterations = 50, .tolerance = 1e-12},
  };

  auto next = step(start, circle_residual, 0.1, config);

  REQUIRE(next.has_value());

  // New point should still be on the circle
  double r2 = next->x(0) * next->x(0) + next->lambda * next->lambda;
  CHECK(r2 == Catch::Approx(1.0).margin(1e-10));

  // x should have increased, lambda decreased
  CHECK(next->x(0) > 0.0);
  CHECK(next->lambda < 1.0);
}

TEST_CASE("continuation traces quarter circle", "[continuation]") {
  // Start at (0, 1), trace to (1, 0) — a quarter of the unit circle.
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  ContinuationConfig config{
      .initial_step = 0.05,
      .max_step = 0.2,
      .min_step = 1e-4,
      .growth_factor = 1.2,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 50, .tolerance = 1e-10},
  };

  auto curve = trace(start, circle_residual, config, [](const CurvePoint& p) { return p.lambda < 0.05; });

  REQUIRE(curve.size() > 2);

  // Every point should lie on the circle
  for (const auto& p : curve) {
    double r2 = p.x(0) * p.x(0) + p.lambda * p.lambda;
    CHECK(r2 == Catch::Approx(1.0).margin(1e-8));
  }

  // Last point should be near (1, 0)
  CHECK(curve.back().x(0) == Catch::Approx(1.0).margin(0.1));
  CHECK(curve.back().lambda == Catch::Approx(0.0).margin(0.1));
}

TEST_CASE("continuation handles turning point on folded cubic", "[continuation]") {
  // Folded curve: lambda - x^3 + x = 0, which has a turning point.
  // Rewritten: R(x, lambda) = lambda - x^3 + x = 0
  auto cubic_residual = [](const arma::vec& x, double lambda) -> arma::vec {
    return arma::vec{lambda - x(0) * x(0) * x(0) + x(0)};
  };

  // Start at (x=0, lambda=0) with tangent dx/ds = 0, dlambda/ds = 1
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 0.0,
      .dx_ds = arma::vec{0.0},
      .dlambda_ds = 1.0,
  };

  ContinuationConfig config{
      .initial_step = 0.05,
      .max_step = 0.1,
      .min_step = 1e-4,
      .growth_factor = 1.1,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 50, .tolerance = 1e-10},
  };

  // Trace until lambda starts decreasing (past the turning point)
  double prev_lambda = start.lambda;
  bool passed_turning = false;
  int steps_taken = 0;
  auto curve = trace(start, cubic_residual, config, [&](const CurvePoint& p) {
    ++steps_taken;
    if (steps_taken > 5 && p.lambda < prev_lambda) {
      passed_turning = true;
    }
    prev_lambda = p.lambda;
    return passed_turning;
  });

  // Should have traced multiple points
  REQUIRE(curve.size() > 3);

  // All points should satisfy the residual
  for (const auto& p : curve) {
    double res = p.lambda - p.x(0) * p.x(0) * p.x(0) + p.x(0);
    CHECK(std::abs(res) < 1e-8);
  }
}

TEST_CASE("continuation step returns nullopt for impossible step", "[continuation]") {
  // Trivial residual that has no solution for lambda != 0
  auto bad_residual = [](const arma::vec& x, double lambda) -> arma::vec {
    return arma::vec{x(0) * x(0) + lambda * lambda + 1.0};
  };

  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 0.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  ContinuationConfig config{
      .initial_step = 0.1,
      .newton = {.max_iterations = 10, .tolerance = 1e-12},
  };

  auto next = step(start, bad_residual, 0.1, config);
  CHECK(!next.has_value());
}

TEST_CASE("trace shrinks step on failed steps and continues", "[continuation]") {
  // A residual that fails for large steps but succeeds for small ones.
  // R(x, lambda) = x^2 + lambda^2 - 1 (unit circle) with a very strict
  // newton tolerance so that large steps fail, forcing shrinkage.
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  ContinuationConfig config{
      .initial_step = 0.5,
      .max_step = 0.5,
      .min_step = 0.01,
      .growth_factor = 1.1,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 3, .tolerance = 1e-14},
  };

  auto curve = trace(start, circle_residual, config, [](const CurvePoint& p) { return p.x(0) > 0.5; });

  REQUIRE(curve.size() > 2);
  for (const auto& p : curve) {
    double r2 = p.x(0) * p.x(0) + p.lambda * p.lambda;
    CHECK(r2 == Catch::Approx(1.0).margin(1e-6));
  }
}

TEST_CASE("trace returns start point when all steps fail", "[continuation]") {
  auto bad_residual = [](const arma::vec& x, double lambda) -> arma::vec {
    return arma::vec{x(0) * x(0) + lambda * lambda + 1.0};
  };

  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 0.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  ContinuationConfig config{
      .initial_step = 0.1,
      .max_step = 0.1,
      .min_step = 0.01,
      .growth_factor = 1.1,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 5, .tolerance = 1e-12},
  };

  auto curve = trace(start, bad_residual, config);

  // Only the starting point should be in the curve
  CHECK(curve.size() == 1);
}

TEST_CASE("trace catches exceptions in step and returns curve so far", "[continuation]") {
  int call_count = 0;
  auto throwing_residual = [&](const arma::vec& x, double lambda) -> arma::vec {
    call_count++;
    if (call_count > 3) {
      throw std::runtime_error("deliberate failure");
    }
    return arma::vec{x(0) * x(0) + lambda * lambda - 1.0};
  };

  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  ContinuationConfig config{
      .initial_step = 0.05,
      .max_step = 0.1,
      .min_step = 0.01,
      .newton = {.max_iterations = 50, .tolerance = 1e-10},
  };

  auto curve = trace(start, throwing_residual, config);

  // Should have at least the start point and stop gracefully
  CHECK(curve.size() >= 1);
}

TEST_CASE("trace grows step after successful steps", "[continuation]") {
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  ContinuationConfig config{
      .initial_step = 0.01,
      .max_step = 0.5,
      .min_step = 1e-4,
      .growth_factor = 2.0,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 50, .tolerance = 1e-10},
  };

  auto curve = trace(start, circle_residual, config, [](const CurvePoint& p) { return p.x(0) > 0.3; });

  // With growth_factor=2.0, the step grows quickly so we need fewer steps
  // to reach x > 0.3 than we would with growth_factor=1.0
  CHECK(curve.size() >= 2);
  CHECK(curve.back().x(0) > 0.3);
}
