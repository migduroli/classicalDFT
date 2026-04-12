// Integration test for the string method algorithm.
// Cross-validates our implementation against Jim's legacy code.

#include "dft/algorithms/string_method.hpp"

#include "legacy/algorithms.hpp"

#include <armadillo>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::string_method;
using Catch::Approx;

namespace {

  constexpr int N = 64;

  // Quadratic energy: F(x) = 0.5 * ||x||^2, force = x.

  auto quadratic_energy(const std::vector<arma::vec>& x) -> std::pair<double, std::vector<arma::vec>> {
    double energy = 0.0;
    std::vector<arma::vec> forces(x.size());
    for (std::size_t s = 0; s < x.size(); ++s) {
      energy += 0.5 * arma::dot(x[s], x[s]);
      forces[s] = x[s];
    }
    return {energy, forces};
  }

  // Single-species version for legacy.

  auto quadratic_energy_single(const arma::vec& x) -> std::pair<double, arma::vec> {
    return {0.5 * arma::dot(x, x), x};
  }

  // Damped relaxation: x_new = alpha * x.

  auto damped_relax(const std::vector<arma::vec>& x) -> std::pair<std::vector<arma::vec>, double> {
    std::vector<arma::vec> relaxed(x.size());
    for (std::size_t s = 0; s < x.size(); ++s) {
      relaxed[s] = 0.98 * x[s];
    }
    auto [energy, forces] = quadratic_energy(relaxed);
    return {relaxed, energy};
  }

  auto damped_relax_single(const arma::vec& x) -> std::pair<arma::vec, double> {
    arma::vec relaxed = 0.98 * x;
    double energy = 0.5 * arma::dot(relaxed, relaxed);
    return {relaxed, energy};
  }

}  // namespace

// --- End-to-end tests ---

TEST_CASE("string method converges for quadratic pathway", "[string_method][integration]") {
  std::vector<arma::vec> a = {0.4 * arma::ones(N)};
  std::vector<arma::vec> b = {0.6 * arma::ones(N)};

  StringMethod sm{
      .tolerance = 1e-3,
      .max_iterations = 15,
      .log_interval = 0,
  };

  auto result = sm.find_pathway(a, b, 7, quadratic_energy, damped_relax);

  CHECK(result.images.size() == 7);
  CHECK(result.iterations > 0);
  CHECK(std::isfinite(result.final_error));

  CHECK(arma::approx_equal(result.images.front().x[0], a[0], "absdiff", 1e-12));
  CHECK(arma::approx_equal(result.images.back().x[0], b[0], "absdiff", 1e-12));
}

TEST_CASE("string method pathway has monotonic arc length", "[string_method][integration]") {
  std::vector<arma::vec> a = {0.4 * arma::ones(N)};
  std::vector<arma::vec> b = {0.6 * arma::ones(N)};

  StringMethod sm{
      .tolerance = 1e-3,
      .max_iterations = 10,
      .log_interval = 0,
  };

  auto result = sm.find_pathway(a, b, 5, quadratic_energy, damped_relax);

  auto alpha = arc_lengths(result.images);
  for (std::size_t j = 1; j < alpha.size(); ++j) {
    CHECK(alpha[j] > alpha[j - 1]);
  }
}

TEST_CASE("string method pathway has finite energies", "[string_method][integration]") {
  std::vector<arma::vec> a = {0.4 * arma::ones(N)};
  std::vector<arma::vec> b = {0.6 * arma::ones(N)};

  StringMethod sm{
      .tolerance = 1e-3,
      .max_iterations = 10,
      .log_interval = 0,
  };

  auto result = sm.find_pathway(a, b, 5, quadratic_energy, damped_relax);

  for (const auto& img : result.images) {
    CHECK(std::isfinite(img.energy));
    CHECK(img.energy >= 0.0);
  }
}

TEST_CASE("string method quadratic pathway stays nearly linear", "[string_method][integration]") {
  std::vector<arma::vec> a = {0.4 * arma::ones(N)};
  std::vector<arma::vec> b = {0.6 * arma::ones(N)};

  StringMethod sm{
      .tolerance = 1e-3,
      .max_iterations = 10,
      .log_interval = 0,
  };

  auto result = sm.find_pathway(a, b, 5, quadratic_energy, damped_relax);

  arma::vec midpoint = 0.5 * (a[0] + b[0]);
  double dev = arma::norm(result.images[2].x[0] - midpoint) / arma::norm(midpoint);

  CHECK(dev < 0.1);
}

// --- Cross-validation against legacy ---

TEST_CASE("reparametrize matches legacy string_interpolate", "[string_method][integration][legacy]") {
  arma::vec a = 0.3 * arma::ones(N);
  arma::vec b = 0.7 * arma::ones(N);

  // Create unequally spaced images.
  int num = 7;
  std::vector<Image> our_images(num);
  std::vector<arma::vec> ref_images(num);

  our_images[0].x = {a};
  our_images[num - 1].x = {b};
  ref_images[0] = a;
  ref_images[num - 1] = b;

  our_images[1].x = {0.32 * arma::ones(N)};
  our_images[2].x = {0.55 * arma::ones(N)};
  our_images[3].x = {0.56 * arma::ones(N)};
  our_images[4].x = {0.57 * arma::ones(N)};
  our_images[5].x = {0.68 * arma::ones(N)};

  ref_images[1] = our_images[1].x[0];
  ref_images[2] = our_images[2].x[0];
  ref_images[3] = our_images[3].x[0];
  ref_images[4] = our_images[4].x[0];
  ref_images[5] = our_images[5].x[0];

  reparametrize(our_images, 4);
  legacy::algorithms::string_interpolate(ref_images, 4);

  for (int j = 0; j < num; ++j) {
    CHECK(arma::approx_equal(our_images[j].x[0], ref_images[j], "absdiff", 1e-12));
  }
}

TEST_CASE("string method matches legacy on same problem", "[string_method][integration][legacy]") {
  arma::vec a = 0.4 * arma::ones(N);
  arma::vec b = 0.6 * arma::ones(N);

  int num_images = 7;
  int max_iter = 5;
  double tol = 1e-3;

  // Our implementation.
  StringMethod sm{
      .tolerance = tol,
      .max_iterations = max_iter,
      .reparametrize_passes = 4,
      .log_interval = 0,
  };

  auto our_result = sm.find_pathway({a}, {b}, num_images, quadratic_energy, damped_relax);

  // Legacy implementation.
  legacy::algorithms::StringConfig ref_config{
      .tol = tol,
      .max_iterations = max_iter,
      .interpolation_passes = 4,
  };

  auto ref_result =
      legacy::algorithms::string_method(a, b, num_images - 2, quadratic_energy_single, damped_relax_single, ref_config);

  // Both should produce the same number of images and iterations.
  CHECK(our_result.images.size() == ref_result.images.size());
  CHECK(our_result.iterations == ref_result.iterations);

  // Energies should match closely after the same number of iterations.
  for (std::size_t j = 0; j < our_result.images.size(); ++j) {
    CHECK(our_result.images[j].energy == Approx(ref_result.energies[j]).epsilon(1e-8));
  }

  // Densities should match.
  for (std::size_t j = 0; j < our_result.images.size(); ++j) {
    CHECK(arma::approx_equal(our_result.images[j].x[0], ref_result.images[j], "absdiff", 1e-10));
  }
}
