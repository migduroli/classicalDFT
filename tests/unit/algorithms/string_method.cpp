#include "dft/algorithms/string_method.hpp"

#include <armadillo>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::string_method;

static constexpr int N = 64;

// Quadratic energy: F(x) = 0.5 * ||x||^2, force = x.

static auto quadratic_energy(const std::vector<arma::vec>& x) -> std::pair<double, std::vector<arma::vec>> {
  double energy = 0.0;
  std::vector<arma::vec> forces(x.size());
  for (std::size_t s = 0; s < x.size(); ++s) {
    energy += 0.5 * arma::dot(x[s], x[s]);
    forces[s] = x[s];
  }
  return {energy, forces};
}

// Trivial relaxation: move x toward zero by a small fraction.

static auto trivial_relax(const std::vector<arma::vec>& x) -> std::pair<std::vector<arma::vec>, double> {
  std::vector<arma::vec> relaxed(x.size());
  for (std::size_t s = 0; s < x.size(); ++s) {
    relaxed[s] = 0.99 * x[s];
  }
  auto [energy, forces] = quadratic_energy(relaxed);
  return {relaxed, energy};
}

// --- linear_interpolation ---

TEST_CASE("linear_interpolation returns correct number of images", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 5);

  CHECK(images.size() == 5);
}

TEST_CASE("linear_interpolation preserves endpoints", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 5);

  CHECK(arma::approx_equal(images.front().x[0], a[0], "absdiff", 1e-14));
  CHECK(arma::approx_equal(images.back().x[0], b[0], "absdiff", 1e-14));
}

TEST_CASE("linear_interpolation midpoint is average of endpoints", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 5);

  arma::vec expected = 0.5 * (a[0] + b[0]);
  CHECK(arma::approx_equal(images[2].x[0], expected, "absdiff", 1e-14));
}

TEST_CASE("linear_interpolation with two images gives only endpoints", "[string_method]") {
  std::vector<arma::vec> a = {0.2 * arma::ones(N)};
  std::vector<arma::vec> b = {0.8 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 2);

  CHECK(images.size() == 2);
  CHECK(arma::approx_equal(images[0].x[0], a[0], "absdiff", 1e-14));
  CHECK(arma::approx_equal(images[1].x[0], b[0], "absdiff", 1e-14));
}

TEST_CASE("linear_interpolation handles multi-component x", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N), 1.0 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N), 2.0 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 3);

  CHECK(images[1].x.size() == 2);
  arma::vec mid0 = 0.5 * (a[0] + b[0]);
  arma::vec mid1 = 0.5 * (a[1] + b[1]);
  CHECK(arma::approx_equal(images[1].x[0], mid0, "absdiff", 1e-14));
  CHECK(arma::approx_equal(images[1].x[1], mid1, "absdiff", 1e-14));
}

// --- arc_lengths ---

TEST_CASE("arc_lengths starts at zero", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 5);
  auto alpha = arc_lengths(images);

  CHECK(alpha.front() == 0.0);
}

TEST_CASE("arc_lengths is monotonically increasing", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 5);
  auto alpha = arc_lengths(images);

  for (std::size_t j = 1; j < alpha.size(); ++j) {
    CHECK(alpha[j] > alpha[j - 1]);
  }
}

TEST_CASE("arc_lengths is zero for identical images", "[string_method]") {
  std::vector<arma::vec> a = {0.5 * arma::ones(N)};

  std::vector<Image> images(4);
  for (auto& img : images) {
    img.x = a;
  }
  auto alpha = arc_lengths(images);

  for (double val : alpha) {
    CHECK(val == 0.0);
  }
}

TEST_CASE("arc_lengths has equal segments for linear interpolation", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 5);
  auto alpha = arc_lengths(images);

  double segment = alpha[1] - alpha[0];
  for (std::size_t j = 2; j < alpha.size(); ++j) {
    CHECK((alpha[j] - alpha[j - 1]) == Catch::Approx(segment).epsilon(1e-12));
  }
}

// --- reparametrize ---

TEST_CASE("reparametrize preserves endpoints", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 5);

  // Perturb interior images to create unequal spacing.
  images[1].x = {0.35 * arma::ones(N)};
  images[2].x = {0.60 * arma::ones(N)};
  images[3].x = {0.62 * arma::ones(N)};

  auto a_copy = images.front().x;
  auto b_copy = images.back().x;

  reparametrize(images);

  CHECK(arma::approx_equal(images.front().x[0], a_copy[0], "absdiff", 1e-14));
  CHECK(arma::approx_equal(images.back().x[0], b_copy[0], "absdiff", 1e-14));
}

TEST_CASE("reparametrize yields nearly equal arc-length spacing", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 7);

  images[1].x = {0.32 * arma::ones(N)};
  images[2].x = {0.55 * arma::ones(N)};
  images[3].x = {0.56 * arma::ones(N)};
  images[4].x = {0.57 * arma::ones(N)};
  images[5].x = {0.68 * arma::ones(N)};

  reparametrize(images, 8);

  auto alpha = arc_lengths(images);
  double ideal_dl = alpha.back() / 6.0;

  for (std::size_t j = 1; j < alpha.size(); ++j) {
    double segment = alpha[j] - alpha[j - 1];
    CHECK(segment == Catch::Approx(ideal_dl).epsilon(1e-4));
  }
}

TEST_CASE("reparametrize is a no-op for equally spaced images", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 5);

  std::vector<std::vector<arma::vec>> originals;
  for (const auto& img : images) {
    originals.push_back(img.x);
  }

  reparametrize(images);

  for (std::size_t j = 0; j < images.size(); ++j) {
    CHECK(arma::approx_equal(images[j].x[0], originals[j][0], "absdiff", 1e-12));
  }
}

TEST_CASE("reparametrize handles two images gracefully", "[string_method]") {
  std::vector<arma::vec> a = {0.3 * arma::ones(N)};
  std::vector<arma::vec> b = {0.7 * arma::ones(N)};

  auto images = linear_interpolation(a, b, 2);

  reparametrize(images);

  CHECK(arma::approx_equal(images[0].x[0], a[0], "absdiff", 1e-14));
  CHECK(arma::approx_equal(images[1].x[0], b[0], "absdiff", 1e-14));
}

// --- StringMethod::find_pathway ---

TEST_CASE("find_pathway produces correct number of images", "[string_method]") {
  std::vector<arma::vec> a = {0.4 * arma::ones(N)};
  std::vector<arma::vec> b = {0.6 * arma::ones(N)};

  StringMethod sm{
      .tolerance = 1e-3,
      .max_iterations = 10,
      .log_interval = 0,
  };

  auto result = sm.find_pathway(a, b, 5, quadratic_energy, trivial_relax);

  CHECK(result.images.size() == 5);
  CHECK(result.iterations > 0);
  CHECK(std::isfinite(result.final_error));
}

TEST_CASE("find_pathway preserves endpoint states", "[string_method]") {
  std::vector<arma::vec> a = {0.4 * arma::ones(N)};
  std::vector<arma::vec> b = {0.6 * arma::ones(N)};

  StringMethod sm{
      .tolerance = 1e-3,
      .max_iterations = 5,
      .log_interval = 0,
  };

  auto result = sm.find_pathway(a, b, 5, quadratic_energy, trivial_relax);

  CHECK(arma::approx_equal(result.images.front().x[0], a[0], "absdiff", 1e-12));
  CHECK(arma::approx_equal(result.images.back().x[0], b[0], "absdiff", 1e-12));
}

TEST_CASE("find_pathway invokes iteration callback", "[string_method]") {
  std::vector<arma::vec> a = {0.4 * arma::ones(N)};
  std::vector<arma::vec> b = {0.6 * arma::ones(N)};

  int callback_count = 0;

  StringMethod sm{
      .tolerance = 1e-3,
      .max_iterations = 3,
      .log_interval = 0,
      .on_iteration = [&](int, double, const std::vector<Image>&) {
        ++callback_count;
        return false;
      },
  };

  auto result = sm.find_pathway(a, b, 4, quadratic_energy, trivial_relax);

  CHECK(callback_count == result.iterations);
}

TEST_CASE("find_pathway early stop via callback", "[string_method]") {
  std::vector<arma::vec> a = {0.4 * arma::ones(N)};
  std::vector<arma::vec> b = {0.6 * arma::ones(N)};

  StringMethod sm{
      .tolerance = 1e-12,
      .max_iterations = 100,
      .log_interval = 0,
      .on_iteration = [](int iteration, double, const std::vector<Image>&) {
        return iteration >= 1;
      },
  };

  auto result = sm.find_pathway(a, b, 4, quadratic_energy, trivial_relax);

  CHECK(result.iterations <= 2);
}
