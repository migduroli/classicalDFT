#include "dft/algorithms/fire.hpp"

#include <armadillo>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::fire;

// Test with a simple quadratic: E = 0.5 * x^T x, f = -x.

static auto quadratic_force(const std::vector<arma::vec>& x) -> std::pair<double, std::vector<arma::vec>> {
  double energy = 0.0;
  std::vector<arma::vec> forces(x.size());
  for (std::size_t s = 0; s < x.size(); ++s) {
    energy += 0.5 * arma::dot(x[s], x[s]);
    forces[s] = -x[s];
  }
  return {energy, forces};
}

TEST_CASE("fire initialize produces valid initial state", "[fire]") {
  std::vector<arma::vec> x0 = {arma::vec{1.0, 2.0, 3.0}};
  Fire config;
  auto state = config.initialize(x0, quadratic_force);

  CHECK(state.energy == Catch::Approx(7.0));
  CHECK(state.rms_force > 0.0);
  CHECK(state.iteration == 0);
  CHECK_FALSE(state.converged);
  CHECK(state.v.size() == 1);
  CHECK(arma::norm(state.v[0]) == Catch::Approx(0.0));
}

TEST_CASE("fire step reduces energy on quadratic", "[fire]") {
  std::vector<arma::vec> x0 = {arma::vec{1.0, 2.0, 3.0}};
  Fire config{.dt = 0.01, .force_tolerance = 1e-6};
  auto state = config.initialize(x0, quadratic_force);
  auto [_, forces] = quadratic_force(state.x);

  auto [new_state, new_forces] = config.step(state, forces, quadratic_force);

  CHECK(new_state.energy < state.energy);
  CHECK(new_state.iteration == 1);
}

TEST_CASE("fire minimize converges on 1D quadratic", "[fire]") {
  std::vector<arma::vec> x0 = {arma::vec{5.0}};
  Fire config{.dt = 0.01, .force_tolerance = 1e-6, .max_steps = 5000};

  auto result = config.minimize(x0, quadratic_force);

  CHECK(result.converged);
  CHECK(std::abs(result.x[0](0)) < 1e-3);
  CHECK(result.energy < 1e-6);
}

TEST_CASE("fire minimize converges on 3D quadratic", "[fire]") {
  std::vector<arma::vec> x0 = {arma::vec{3.0, -2.0, 1.0}};
  Fire config{.dt = 0.01, .force_tolerance = 1e-6, .max_steps = 5000};

  auto result = config.minimize(x0, quadratic_force);

  CHECK(result.converged);
  CHECK(arma::norm(result.x[0]) < 1e-3);
}

TEST_CASE("fire minimize handles multi-species", "[fire]") {
  std::vector<arma::vec> x0 = {arma::vec{2.0, 1.0}, arma::vec{-1.0, 3.0}};
  Fire config{.dt = 0.01, .force_tolerance = 1e-6, .max_steps = 5000};

  auto result = config.minimize(x0, quadratic_force);

  CHECK(result.converged);
  CHECK(arma::norm(result.x[0]) < 1e-3);
  CHECK(arma::norm(result.x[1]) < 1e-3);
}

// Rosenbrock-like: E = (1-x)^2 + 100*(y-x^2)^2

static auto rosenbrock_force(const std::vector<arma::vec>& x) -> std::pair<double, std::vector<arma::vec>> {
  double xi = x[0](0);
  double yi = x[0](1);
  double energy = (1.0 - xi) * (1.0 - xi) + 100.0 * (yi - xi * xi) * (yi - xi * xi);
  double dfdx = -2.0 * (1.0 - xi) - 400.0 * xi * (yi - xi * xi);
  double dfdy = 200.0 * (yi - xi * xi);
  return {energy, {arma::vec{-dfdx, -dfdy}}};
}

TEST_CASE("fire minimize makes progress on Rosenbrock", "[fire]") {
  std::vector<arma::vec> x0 = {arma::vec{-1.0, 1.0}};
  Fire config{
      .dt = 1e-4,
      .dt_max = 1e-3,
      .force_tolerance = 0.1,
      .max_steps = 50000,
  };

  auto result = config.minimize(x0, rosenbrock_force);

  // Should at least reduce energy significantly from the starting point
  auto [e0, _] = rosenbrock_force(x0);
  CHECK(result.energy < e0 * 0.1);
}

TEST_CASE("fire at minimum reports converged", "[fire]") {
  std::vector<arma::vec> x0 = {arma::vec{0.0, 0.0, 0.0}};
  Fire config{.force_tolerance = 1e-6};

  auto state = config.initialize(x0, quadratic_force);
  CHECK(state.converged);
}

TEST_CASE("fire step backs off after transient compute failure", "[fire]") {
  int eval_count = 0;
  auto transient_failure = [&](const std::vector<arma::vec>& x) -> std::pair<double, std::vector<arma::vec>> {
    eval_count++;
    if (eval_count == 2) {
      throw std::runtime_error("transient failure");
    }
    return quadratic_force(x);
  };

  std::vector<arma::vec> x0 = {arma::vec{1.0}};
  Fire config{.dt = 0.1, .force_tolerance = 1e-6};
  auto state = config.initialize(x0, transient_failure);
  auto [_, forces] = quadratic_force(state.x);

  auto [backed_off_state, backed_off_forces] = config.step(state, forces, transient_failure);

  CHECK(backed_off_state.energy == Catch::Approx(state.energy));
  CHECK(backed_off_state.dt == Catch::Approx(0.05));
  CHECK(arma::approx_equal(backed_off_state.x[0], state.x[0], "absdiff", 0.0));
  CHECK(arma::approx_equal(backed_off_forces[0], forces[0], "absdiff", 0.0));

  auto [recovered_state, recovered_forces] = config.step(backed_off_state, backed_off_forces, transient_failure);
  CHECK(recovered_state.energy < state.energy);
  CHECK(recovered_state.iteration == 1);
}

// Uphill function: force that changes sign rapidly to cause persistent negative power

TEST_CASE("fire throws when max uphill steps exceeded", "[fire]") {
  // Force function where the gradient changes direction every step.
  // This causes P = v.f < 0 persistently after the first step.
  int eval_count = 0;
  auto alternating_force = [&](const std::vector<arma::vec>& x) -> std::pair<double, std::vector<arma::vec>> {
    eval_count++;
    double sign = (eval_count % 2 == 0) ? -1.0 : 1.0;
    double energy = sign * arma::dot(x[0], x[0]);
    return {energy, {sign * (-2.0 * x[0])}};
  };

  std::vector<arma::vec> x0 = {arma::vec{1.0, 1.0}};
  Fire config{
      .dt = 0.5,
      .dt_max = 0.5,
      .n_delay = 0,
      .max_uphill = 3,
      .force_tolerance = 1e-10,
      .max_steps = 5000,
  };

  REQUIRE_THROWS_AS(config.minimize(x0, alternating_force), std::runtime_error);
}
