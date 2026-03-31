#include "dft/math/integration.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

using namespace dft::math;

TEST_CASE("integrate constant function", "[integration]") {
  Integrator ig([](double) { return 2.0; });
  auto r = ig.integrate(0.0, 5.0);
  CHECK(r.value == Catch::Approx(10.0).epsilon(1e-8));
}

TEST_CASE("integrate linear function", "[integration]") {
  Integrator ig([](double x) { return x; });
  auto r = ig.integrate(0.0, 4.0);
  CHECK(r.value == Catch::Approx(8.0).epsilon(1e-8));
}

TEST_CASE("integrate sin over full period", "[integration]") {
  Integrator ig([](double x) { return std::sin(x); });
  auto r = ig.integrate(0.0, 2.0 * M_PI);
  CHECK(r.value == Catch::Approx(0.0).margin(1e-10));
}

TEST_CASE("integrate sin over half period", "[integration]") {
  Integrator ig([](double x) { return std::sin(x); });
  auto r = ig.integrate(0.0, M_PI);
  CHECK(r.value == Catch::Approx(2.0).epsilon(1e-8));
}

TEST_CASE("integrate_fast on smooth function", "[integration]") {
  Integrator ig([](double x) { return x * x; });
  auto r = ig.integrate_fast(0.0, 3.0);
  CHECK(r.value == Catch::Approx(9.0).epsilon(1e-8));
}

TEST_CASE("integrate_upper_infinite on exp(-x)", "[integration]") {
  Integrator ig([](double x) { return std::exp(-x); });
  auto r = ig.integrate_upper_infinite(0.0);
  CHECK(r.value == Catch::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("integrate_lower_infinite on exp(x)", "[integration]") {
  Integrator ig([](double x) { return std::exp(x); });
  auto r = ig.integrate_lower_infinite(0.0);
  CHECK(r.value == Catch::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("integrate_infinite on Gaussian", "[integration]") {
  // integral of exp(-x^2) from -inf to +inf = sqrt(pi)
  Integrator ig([](double x) { return std::exp(-x * x); });
  auto r = ig.integrate_infinite();
  CHECK(r.value == Catch::Approx(std::sqrt(M_PI)).epsilon(1e-6));
}

TEST_CASE("custom integration config tolerances", "[integration]") {
  IntegrationConfig cfg{.absolute_tolerance = 1e-12, .relative_tolerance = 1e-12, .workspace_size = 2000};
  Integrator ig([](double x) { return std::exp(-x * x); }, cfg);
  auto r = ig.integrate(-5.0, 5.0);
  CHECK(r.value == Catch::Approx(std::sqrt(M_PI)).epsilon(1e-10));
  CHECK(r.error < 1e-10);
}
