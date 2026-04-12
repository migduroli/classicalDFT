// check.cpp — Cross-validation of string method against Jim's
// String_Master.cpp / String_Slave.cpp implementation (translated
// verbatim in legacy/algorithms.hpp).
//
// Compares:
//   1. Arc-length reparametrization (our reparametrize vs Jim's string_interpolate)
//   2. Full string method run on the same quadratic problem

#include "legacy/algorithms.hpp"

#include <cmath>
#include <dftlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

using namespace dft;

static int g_failures = 0;
static int g_checks = 0;

static void check(std::string_view label, double ours, double jim, double tol = 1e-10) {
  ++g_checks;
  double diff = std::abs(ours - jim);
  bool ok = diff <= tol;
  if (!ok) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": ours=" << ours << " jim=" << jim << " diff=" << diff << "\n";
  }
}

static void section(std::string_view title) {
  std::cout << "\n" << title << "\n";
  std::cout << std::string(title.size(), '-') << "\n";
}

// Quadratic energy: F(x) = 0.5 * ||x||^2 with force = x.

static auto quadratic_energy_ours(const std::vector<arma::vec>& x) -> std::pair<double, std::vector<arma::vec>> {
  double energy = 0.0;
  std::vector<arma::vec> forces(x.size());
  for (std::size_t s = 0; s < x.size(); ++s) {
    energy += 0.5 * arma::dot(x[s], x[s]);
    forces[s] = x[s];
  }
  return {energy, forces};
}

static auto quadratic_energy_jim(const arma::vec& x) -> std::pair<double, arma::vec> {
  return {0.5 * arma::dot(x, x), x};
}

// Damped relaxation: x_new = (1 - alpha)^steps * x.

static auto relax_ours(const std::vector<arma::vec>& x) -> std::pair<std::vector<arma::vec>, double> {
  std::vector<arma::vec> result(x.size());
  double energy = 0.0;
  for (std::size_t s = 0; s < x.size(); ++s) {
    result[s] = 0.99 * x[s];
    energy += 0.5 * arma::dot(result[s], result[s]);
  }
  return {result, energy};
}

static auto relax_jim(const arma::vec& x) -> std::pair<arma::vec, double> {
  arma::vec result = 0.99 * x;
  return {result, 0.5 * arma::dot(result, result)};
}

int main() {
  std::cout << std::setprecision(15);

  // ------------------------------------------------------------------
  // Step 1: Arc-length reparametrization
  //
  // Build a non-uniform 1D chain, reparametrize with both
  // implementations, compare pointwise.
  // ------------------------------------------------------------------

  section("Step 1: Reparametrize vs legacy string_interpolate");

  constexpr int N = 32;
  constexpr int num_images = 7;

  arma::vec endpoint_a = arma::linspace(-1.0, 1.0, N);
  arma::vec endpoint_b = arma::linspace(2.0, 4.0, N);

  // Build images with non-uniform spacing (t^2 distortion).
  std::vector<algorithms::string_method::Image> our_images(num_images);
  std::vector<arma::vec> jim_images(num_images);

  for (int j = 0; j < num_images; ++j) {
    double t = static_cast<double>(j) / (num_images - 1);
    double f = t * t; // Quadratic distortion.
    arma::vec img = (1.0 - f) * endpoint_a + f * endpoint_b;
    our_images[j].x = {img};
    jim_images[j] = img;
  }

  // Reparametrize both.
  algorithms::string_method::reparametrize(our_images, 4);
  legacy::algorithms::string_interpolate(jim_images, 4);

  int step1_ok = 0;
  for (int j = 0; j < num_images; ++j) {
    for (int i = 0; i < N; ++i) {
      check(
          "reparam[" + std::to_string(j) + "][" + std::to_string(i) + "]",
          our_images[j].x[0](i),
          jim_images[j](i),
          1e-10
      );
      if (std::abs(our_images[j].x[0](i) - jim_images[j](i)) <= 1e-10) {
        ++step1_ok;
      }
    }
  }
  std::cout << "  Matched " << step1_ok << "/" << num_images * N << " pointwise values\n";

  // ------------------------------------------------------------------
  // Step 2: Full string method on quadratic problem
  //
  // Both use identical initial conditions, same relaxation, same
  // tolerance. Results must match: iteration count, final energies,
  // final densities.
  // ------------------------------------------------------------------

  section("Step 2: Full string method vs legacy");

  arma::vec state_a = arma::ones(N) * (-2.0);
  arma::vec state_b = arma::ones(N) * 3.0;
  int n_interior = 9;
  int n_total = n_interior + 2;

  // Run our string method.
  algorithms::string_method::StringMethod sm{
      .tolerance = 1e-6,
      .max_iterations = 100,
      .reparametrize_passes = 4,
      .log_interval = 0, // Silent.
  };

  auto our_result = sm.find_pathway({state_a}, {state_b}, n_total, quadratic_energy_ours, relax_ours);

  // Run Jim's string method.
  legacy::algorithms::StringConfig jim_config{
      .tol = 1e-6,
      .max_iterations = 100,
      .interpolation_passes = 4,
  };

  auto jim_result =
      legacy::algorithms::string_method(state_a, state_b, n_interior, quadratic_energy_jim, relax_jim, jim_config);

  std::cout << "  Our iterations:  " << our_result.iterations << "\n";
  std::cout << "  Jim iterations:  " << jim_result.iterations << "\n";
  std::cout << "  Our final_error: " << our_result.final_error << "\n";
  std::cout << "  Jim final_error: " << jim_result.final_error << "\n";
  std::cout << "  Our converged:   " << our_result.converged << "\n";
  std::cout << "  Jim converged:   " << jim_result.converged << "\n";

  check("iterations", static_cast<double>(our_result.iterations), static_cast<double>(jim_result.iterations), 0.0);
  check("converged", static_cast<double>(our_result.converged), static_cast<double>(jim_result.converged), 0.0);
  check("final_error", our_result.final_error, jim_result.final_error, 1e-12);

  // Compare energies.
  for (int j = 0; j < n_total; ++j) {
    check("energy[" + std::to_string(j) + "]", our_result.images[j].energy, jim_result.energies[j], 1e-10);
  }

  // Compare densities pointwise.
  int step2_ok = 0;
  for (int j = 0; j < n_total; ++j) {
    for (int i = 0; i < N; ++i) {
      check(
          "x[" + std::to_string(j) + "][" + std::to_string(i) + "]",
          our_result.images[j].x[0](i),
          jim_result.images[j](i),
          1e-10
      );
      if (std::abs(our_result.images[j].x[0](i) - jim_result.images[j](i)) <= 1e-10) {
        ++step2_ok;
      }
    }
  }
  std::cout << "  Matched " << step2_ok << "/" << n_total * N << " pointwise values\n";

  // ------------------------------------------------------------------
  // Summary
  // ------------------------------------------------------------------

  std::cout << "\n========================================\n";
  std::cout << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures > 0 ? 1 : 0;
}
