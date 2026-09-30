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

  Continuation config{
      .initial_step = 0.1,
      .newton = {.max_iterations = 50, .tolerance = 1e-12},
  };

  auto next = config.step(start, circle_residual, 0.1);

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

  Continuation config{
      .initial_step = 0.05,
      .max_step = 0.2,
      .min_step = 1e-4,
      .growth_factor = 1.2,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 50, .tolerance = 1e-10},
  };

  auto curve = config.trace(start, circle_residual, [](const CurvePoint& p) { return p.lambda < 0.05; });

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

  Continuation config{
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
  auto curve = config.trace(start, cubic_residual, [&](const CurvePoint& p) {
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

  Continuation config{
      .initial_step = 0.1,
      .newton = {.max_iterations = 10, .tolerance = 1e-12},
  };

  auto next = config.step(start, bad_residual, 0.1);
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

  Continuation config{
      .initial_step = 0.5,
      .max_step = 0.5,
      .min_step = 0.01,
      .growth_factor = 1.1,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 3, .tolerance = 1e-14},
  };

  auto curve = config.trace(start, circle_residual, [](const CurvePoint& p) { return p.x(0) > 0.5; });

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

  Continuation config{
      .initial_step = 0.1,
      .max_step = 0.1,
      .min_step = 0.01,
      .growth_factor = 1.1,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 5, .tolerance = 1e-12},
  };

  auto curve = config.trace(start, bad_residual);

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

  Continuation config{
      .initial_step = 0.05,
      .max_step = 0.1,
      .min_step = 0.01,
      .newton = {.max_iterations = 50, .tolerance = 1e-10},
  };

  auto curve = config.trace(start, throwing_residual);

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

  Continuation config{
      .initial_step = 0.01,
      .max_step = 0.5,
      .min_step = 1e-4,
      .growth_factor = 2.0,
      .shrink_factor = 0.5,
      .newton = {.max_iterations = 50, .tolerance = 1e-10},
  };

  auto curve = config.trace(start, circle_residual, [](const CurvePoint& p) { return p.x(0) > 0.3; });

  // With growth_factor=2.0, the step grows quickly so we need fewer steps
  // to reach x > 0.3 than we would with growth_factor=1.0
  CHECK(curve.size() >= 2);
  CHECK(curve.back().x(0) > 0.3);
}

TEST_CASE("matrix-free continuation step advances along unit circle", "[continuation]") {
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  MatrixFreeContinuation config{
      .initial_step = 0.1,
      .newton = {.max_iterations = 50, .tolerance = 1e-10, .gmres = {.tolerance = 1e-12}},
  };

  auto next = config.step(start, circle_residual, 0.1);

  REQUIRE(next.has_value());

  double r2 = next->x(0) * next->x(0) + next->lambda * next->lambda;
  CHECK(r2 == Catch::Approx(1.0).margin(1e-6));
  CHECK(next->x(0) > 0.0);
  CHECK(next->lambda < 1.0);
}

TEST_CASE("matrix-free continuation traces quarter circle", "[continuation]") {
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };

  MatrixFreeContinuation config{
      .initial_step = 0.05,
      .max_step = 0.2,
      .min_step = 1e-4,
      .newton = {.max_iterations = 50, .tolerance = 1e-8, .gmres = {.tolerance = 1e-10}},
  };

  auto curve = config.trace(start, circle_residual, [](const CurvePoint& p) { return p.lambda < 0.05; });

  REQUIRE(curve.size() > 2);

  for (const auto& p : curve) {
    double r2 = p.x(0) * p.x(0) + p.lambda * p.lambda;
    CHECK(r2 == Catch::Approx(1.0).margin(1e-5));
  }

  CHECK(curve.back().x(0) == Catch::Approx(1.0).margin(0.1));
}

TEST_CASE("matrix-free continuation handles turning point", "[continuation]") {
  auto cubic_residual = [](const arma::vec& x, double lambda) -> arma::vec {
    return arma::vec{lambda - x(0) * x(0) * x(0) + x(0)};
  };

  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 0.0,
      .dx_ds = arma::vec{0.0},
      .dlambda_ds = 1.0,
  };

  MatrixFreeContinuation config{
      .initial_step = 0.05,
      .max_step = 0.1,
      .min_step = 1e-4,
      .newton = {.max_iterations = 50, .tolerance = 1e-8, .gmres = {.tolerance = 1e-10}},
  };

  double prev_lambda = start.lambda;
  bool passed_turning = false;
  int steps_taken = 0;
  auto curve = config.trace(start, cubic_residual, [&](const CurvePoint& p) {
    ++steps_taken;
    if (steps_taken > 5 && p.lambda < prev_lambda) {
      passed_turning = true;
    }
    prev_lambda = p.lambda;
    return passed_turning;
  });

  REQUIRE(curve.size() > 3);

  for (const auto& p : curve) {
    double res = p.lambda - p.x(0) * p.x(0) * p.x(0) + p.x(0);
    CHECK(std::abs(res) < 1e-5);
  }
}

// Event location and branch switching.

static auto cubic_fold_residual(const arma::vec& x, double lambda) -> arma::vec {
  return arma::vec{lambda - x(0) * x(0) * x(0) + x(0)};
}

// Curve lambda = x^3 - x from x = -1.5 to x = 1.5, through both folds.
static auto trace_cubic(const Continuation& config) -> std::vector<CurvePoint> {
  const double x0 = -1.5;
  const double norm = std::sqrt(1.0 + std::pow(3.0 * x0 * x0 - 1.0, 2));
  CurvePoint start{
      .x = arma::vec{x0},
      .lambda = x0 * x0 * x0 - x0,
      .dx_ds = arma::vec{1.0 / norm},
      .dlambda_ds = (3.0 * x0 * x0 - 1.0) / norm,
  };
  return config.trace(start, cubic_fold_residual, [](const CurvePoint& p) { return p.x(0) > 1.5; });
}

static const Continuation event_config{
    .initial_step = 0.05,
    .max_step = 0.2,
    .min_step = 1e-6,
    .newton = {.max_iterations = 50, .tolerance = 1e-12},
};

TEST_CASE("arclength recovers the step length", "[continuation]") {
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };
  auto next = event_config.step(start, circle_residual, 0.1);
  REQUIRE(next.has_value());
  CHECK(arclength(start, *next) == Catch::Approx(0.1).margin(1e-12));
}

TEST_CASE("locate finds a root of a test function between two points", "[continuation]") {
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };
  auto next = event_config.step(start, circle_residual, 0.8);
  REQUIRE(next.has_value());
  REQUIRE(next->x(0) > 0.5);

  auto root = event_config.locate(start, *next, circle_residual, [](const CurvePoint& p) { return p.x(0) - 0.5; });

  CHECK(root.x(0) == Catch::Approx(0.5).margin(1e-10));
  CHECK(root.lambda == Catch::Approx(std::sqrt(0.75)).margin(1e-10));
}

TEST_CASE("locate works with the matrix-free stepper", "[continuation]") {
  MatrixFreeContinuation config{
      .newton = {.max_iterations = 50, .tolerance = 1e-9, .gmres = {.tolerance = 1e-12}},
  };
  CurvePoint start{
      .x = arma::vec{0.0},
      .lambda = 1.0,
      .dx_ds = arma::vec{1.0},
      .dlambda_ds = 0.0,
  };
  auto next = config.step(start, circle_residual, 0.6);
  REQUIRE(next.has_value());
  REQUIRE(next->x(0) > 0.5);

  auto root = config.locate(start, *next, circle_residual, [](const CurvePoint& p) { return p.x(0) - 0.5; });

  CHECK(root.x(0) == Catch::Approx(0.5).margin(1e-7));
}

TEST_CASE("sign_changes reports intervals and respects the floor", "[continuation]") {
  std::vector<CurvePoint> curve;
  for (double g : {1.0, 0.5, -0.5, -1e-12, 1e-12, 2.0}) {
    curve.push_back(CurvePoint{.x = arma::vec{g}, .lambda = 0.0, .dx_ds = arma::vec{1.0}, .dlambda_ds = 0.0});
  }
  auto g = [](const CurvePoint& p) {
    return p.x(0);
  };

  auto all = sign_changes(curve, g);
  REQUIRE(all.size() == 2);
  CHECK(all[0] == 1);
  CHECK(all[1] == 3);

  auto resolved = sign_changes(curve, g, 1e-8);
  REQUIRE(resolved.size() == 1);
  CHECK(resolved[0] == 1);
}

TEST_CASE("folds locates both turning points of the cubic", "[continuation]") {
  auto curve = trace_cubic(event_config);
  auto found = event_config.folds(curve, cubic_fold_residual);

  REQUIRE(found.size() == 2);
  const double xf = 1.0 / std::sqrt(3.0);
  const double lf = 2.0 / (3.0 * std::sqrt(3.0));
  CHECK(found[0].x(0) == Catch::Approx(-xf).margin(1e-9));
  CHECK(found[0].lambda == Catch::Approx(lf).margin(1e-12));
  CHECK(found[1].x(0) == Catch::Approx(xf).margin(1e-9));
  CHECK(found[1].lambda == Catch::Approx(-lf).margin(1e-12));
  CHECK(std::abs(found[0].dlambda_ds) < 1e-8);
}

TEST_CASE("crossings locates eigenvalue zeros and their positions in the spectrum", "[continuation]") {
  // Two decoupled pitchforks on the trivial branch x = 0:
  //   R_i(x, lambda) = (lambda - c_i) x_i - x_i^3,  c = (1, 2).
  // The Jacobian at x = 0 is diag(c_i - lambda) after a sign change of R,
  // so its eigenvalues go negative at lambda = 1 and lambda = 2.
  auto residual = [](const arma::vec& x, double lambda) -> arma::vec {
    arma::vec c{1.0, 2.0};
    return (c - lambda) % x + arma::pow(x, 3);
  };
  auto spectrum = [](const CurvePoint& p) -> arma::vec {
    arma::vec c{1.0, 2.0};
    return arma::sort(c - p.lambda + 3.0 * arma::square(p.x));
  };

  CurvePoint start{
      .x = arma::vec{0.0, 0.0},
      .lambda = 0.0,
      .dx_ds = arma::vec{0.0, 0.0},
      .dlambda_ds = 1.0,
  };
  auto curve = event_config.trace(start, residual, [](const CurvePoint& p) { return p.lambda > 3.0; });
  auto found = event_config.crossings(curve, residual, spectrum);

  REQUIRE(found.size() == 2);
  CHECK(found[0].point.lambda == Catch::Approx(1.0).margin(1e-10));
  CHECK(found[0].eigenvalue == 0);
  CHECK(found[1].point.lambda == Catch::Approx(2.0).margin(1e-10));
  CHECK(found[1].eigenvalue == 1);
}

TEST_CASE("crossings returns nothing for an empty or stable curve", "[continuation]") {
  auto spectrum = [](const CurvePoint&) -> arma::vec {
    return arma::vec{1.0};
  };
  CHECK(event_config.crossings({}, circle_residual, spectrum).empty());

  CurvePoint start{.x = arma::vec{0.0}, .lambda = 1.0, .dx_ds = arma::vec{1.0}, .dlambda_ds = 0.0};
  auto curve = event_config.trace(start, circle_residual, [](const CurvePoint& p) { return p.x(0) > 0.5; });
  CHECK(event_config.crossings(curve, circle_residual, spectrum).empty());
}

TEST_CASE("switch_branch steps onto both arms of a pitchfork", "[continuation]") {
  // R(x, lambda) = lambda x - x^3: trivial branch x = 0 and the parabola
  // lambda = x^2, meeting at the pitchfork (0, 0).
  auto residual = [](const arma::vec& x, double lambda) -> arma::vec {
    return arma::vec{lambda * x(0) - x(0) * x(0) * x(0)};
  };
  CurvePoint bifurcation{.x = arma::vec{0.0}, .lambda = 0.0, .dx_ds = arma::vec{0.0}, .dlambda_ds = 1.0};

  auto plus = event_config.switch_branch(bifurcation, residual, arma::vec{2.0}, 0.1);
  auto minus = event_config.switch_branch(bifurcation, residual, arma::vec{-1.0}, 0.1);

  REQUIRE(plus.has_value());
  REQUIRE(minus.has_value());
  CHECK(plus->x(0) == Catch::Approx(0.1).margin(1e-12));
  CHECK(plus->lambda == Catch::Approx(0.01).margin(1e-10));
  CHECK(minus->x(0) == Catch::Approx(-0.1).margin(1e-12));
  CHECK(minus->lambda == Catch::Approx(0.01).margin(1e-10));
}

TEST_CASE("switch_branch normalises a direction with a lambda component", "[continuation]") {
  // Transcritical R(x, lambda) = x (lambda - x): branches x = 0 and x = lambda.
  auto residual = [](const arma::vec& x, double lambda) -> arma::vec {
    return arma::vec{x(0) * (lambda - x(0))};
  };
  CurvePoint bifurcation{.x = arma::vec{0.0}, .lambda = 0.0, .dx_ds = arma::vec{0.0}, .dlambda_ds = 1.0};

  auto next = event_config.switch_branch(bifurcation, residual, arma::vec{1.0}, 0.2, 1.0);

  REQUIRE(next.has_value());
  CHECK(next->x(0) == Catch::Approx(next->lambda).margin(1e-10));
  CHECK(std::hypot(next->x(0), next->lambda) == Catch::Approx(0.2).margin(1e-10));
}
