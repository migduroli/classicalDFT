#include "plot.hpp"

#include <dftlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>

using namespace dft;

// Two-dimensional potential with a curved valley connecting two minima.
//
//   V(x, y) = (x^2 - 1)^2 + 10 (y - x^2)^2
//
// Minima at (-1, 1) and (+1, 1) with V = 0.
// The valley follows the parabola y = x^2.
// The linear interpolation between the minima stays at y = 1, passing
// over a high barrier at (0, 1) with V = 11.
// The true MEP follows the parabola and has a barrier of only 1.0
// at the saddle point (0, 0).
//
// State: {arma::vec({x}), arma::vec({y})} — two scalar components.

static auto curved_valley(const std::vector<arma::vec>& state) -> std::pair<double, std::vector<arma::vec>> {
  double x = state[0](0);
  double y = state[1](0);
  double r = x * x - 1.0;
  double s = y - x * x;
  double energy = r * r + 10.0 * s * s;
  double fx = 4.0 * x * r - 40.0 * x * s;
  double fy = 20.0 * s;
  return {energy, {arma::vec({fx}), arma::vec({fy})}};
}

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  console::info("String method: MEP in a curved valley");

  // Minima at (-1, 1) and (+1, 1).

  std::vector<arma::vec> state_a = {arma::vec({-1.0}), arma::vec({1.0})};
  std::vector<arma::vec> state_b = {arma::vec({+1.0}), arma::vec({1.0})};

  auto [energy_a, _a] = curved_valley(state_a);
  auto [energy_b, _b] = curved_valley(state_b);

  std::println(std::cout, "  Endpoint A (-1, 1)  energy: {:.6f}", energy_a);
  std::println(std::cout, "  Endpoint B (+1, 1)  energy: {:.6f}", energy_b);
  std::println(std::cout, "");

  int num_images = 41;

  // Capture the initial (linear) path for comparison.

  auto initial_images = algorithms::string_method::linear_interpolation(state_a, state_b, num_images);
  for (auto& img : initial_images) {
    auto [e, f] = curved_valley(img.x);
    img.energy = e;
  }

  std::vector<double> path_x_init, path_y_init;
  for (std::size_t j = 0; j < initial_images.size(); ++j) {
    path_x_init.push_back(initial_images[j].x[0](0));
    path_y_init.push_back(initial_images[j].x[1](0));
  }

  // --- Simplified string method (full-gradient relaxation) ---

  console::info("Running simplified string method (full-gradient relaxation)");

  // Gradient descent relaxation used by the simplified variant.

  auto gradient_descent = [](const std::vector<arma::vec>& state) -> std::pair<std::vector<arma::vec>, double> {
    auto x = state;
    constexpr double alpha = 0.01;
    constexpr int steps = 20;
    for (int step = 0; step < steps; ++step) {
      auto [e, forces] = curved_valley(x);
      for (std::size_t s = 0; s < x.size(); ++s) {
        x[s] -= alpha * forces[s];
      }
    }
    auto [e_final, f_final] = curved_valley(x);
    return {x, e_final};
  };

  algorithms::string_method::StringMethod sm_simplified{
      .tolerance = 1e-8,
      .max_iterations = 200,
      .reparametrize_passes = 4,
      .log_interval = 10,
  };

  auto result_simplified = sm_simplified.find_pathway(state_a, state_b, num_images, curved_valley, gradient_descent);

  std::println(std::cout, "\n  Converged:    {}", result_simplified.converged);
  std::println(std::cout, "  Iterations:   {}", result_simplified.iterations);
  std::println(std::cout, "  Final error:  {:.6e} (RMS energy change)\n", result_simplified.final_error);

  // --- Full string method (perpendicular-force evolution) ---

  std::vector<double> iter_history;
  std::vector<double> error_history;

  algorithms::string_method::StringMethod sm_full{
      .tolerance = 1e-6,
      .max_iterations = 200,
      .reparametrize_passes = 4,
      .log_interval = 10,
      .evolution_steps = 20,
      .evolution_alpha = 0.01,
      .on_iteration = [&](int iteration, double error, const auto& /*images*/) {
        iter_history.push_back(static_cast<double>(iteration));
        error_history.push_back(error);
        return false;
      },
  };

  console::info("Running full string method (perpendicular-force evolution)");

  auto result_full = sm_full.find_pathway(state_a, state_b, num_images, curved_valley);

  std::println(std::cout, "\n  Converged:    {}", result_full.converged);
  std::println(std::cout, "  Iterations:   {}", result_full.iterations);
  std::println(std::cout, "  Final error:  {:.6e} (max |F_perp|)\n", result_full.final_error);

  // Report the full string method pathway.

  auto alpha = algorithms::string_method::arc_lengths(result_full.images);

  std::println(std::cout, "  {:>5s}  {:>10s}  {:>10s}  {:>12s}  {:>14s}", "image", "x", "y", "arc_length", "energy");
  std::println(std::cout, "  {}", std::string(57, '-'));
  for (std::size_t j = 0; j < result_full.images.size(); ++j) {
    double x = result_full.images[j].x[0](0);
    double y = result_full.images[j].x[1](0);
    std::println(
        std::cout,
        "  {:>5d}  {:>10.6f}  {:>10.6f}  {:>12.6f}  {:>14.6f}",
        static_cast<int>(j),
        x,
        y,
        alpha[j],
        result_full.images[j].energy
    );
  }

  // Find the barrier.

  double barrier = result_full.images[0].energy;
  int barrier_image = 0;
  for (std::size_t j = 1; j < result_full.images.size(); ++j) {
    if (result_full.images[j].energy > barrier) {
      barrier = result_full.images[j].energy;
      barrier_image = static_cast<int>(j);
    }
  }

  std::println(std::cout, "\n  Barrier:          {:.6f} (image {})", barrier, barrier_image);
  std::println(std::cout, "  Straight-line:    11.000000  (linear interpolation at y=1)");
  std::println(std::cout, "  True saddle:      1.000000   (parabolic valley at y=x^2)");

  // Save both pathways to CSV.

  {
    std::ofstream csv("exports/pathway.csv");
    csv << "image,x,y,arc_length,energy\n";
    for (std::size_t j = 0; j < result_full.images.size(); ++j) {
      double x = result_full.images[j].x[0](0);
      double y = result_full.images[j].x[1](0);
      csv << std::format("{},{:.8e},{:.8e},{:.8e},{:.8e}\n", j, x, y, alpha[j], result_full.images[j].energy);
    }
  }
  {
    auto alpha_s = algorithms::string_method::arc_lengths(result_simplified.images);
    std::ofstream csv("exports/pathway_simplified.csv");
    csv << "image,x,y,arc_length,energy\n";
    for (std::size_t j = 0; j < result_simplified.images.size(); ++j) {
      double x = result_simplified.images[j].x[0](0);
      double y = result_simplified.images[j].x[1](0);
      csv << std::format("{},{:.8e},{:.8e},{:.8e},{:.8e}\n", j, x, y, alpha_s[j], result_simplified.images[j].energy);
    }
  }
  std::println(std::cout, "\n  Saved pathways to exports/pathway.csv and exports/pathway_simplified.csv");

  // Compute theoretical MEP by RK4 steepest descent from the saddle point.
  // The MEP is the integral curve of -∇V starting at (ε, 0) toward (+1, 1),
  // then mirrored by the symmetry V(x, y) = V(-x, y).

  std::vector<double> theo_mep_x;
  std::vector<double> theo_mep_y;

  {
    auto steepest_rhs = [](double xx, double yy) -> std::pair<double, double> {
      double rr = xx * xx - 1.0;
      double ss = yy - xx * xx;
      return {-(4.0 * xx * rr - 40.0 * xx * ss), -20.0 * ss};
    };

    constexpr double dt = 1e-5;
    constexpr double dx_sample = 1.0 / 200;
    double x = 1e-8;
    double y = 0.0;
    double next_x = 0.0;

    std::vector<double> half_x = {0.0};
    std::vector<double> half_y = {0.0};

    while (x < 0.999) {
      auto [k1x, k1y] = steepest_rhs(x, y);
      auto [k2x, k2y] = steepest_rhs(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
      auto [k3x, k3y] = steepest_rhs(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
      auto [k4x, k4y] = steepest_rhs(x + dt * k3x, y + dt * k3y);
      x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
      y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);

      if (x >= next_x + dx_sample) {
        half_x.push_back(x);
        half_y.push_back(y);
        next_x = x;
      }
    }

    half_x.push_back(1.0);
    half_y.push_back(1.0);

    // Full path: mirror -x branch, then +x branch.
    for (int i = static_cast<int>(half_x.size()) - 1; i >= 0; --i) {
      theo_mep_x.push_back(-half_x[static_cast<std::size_t>(i)]);
      theo_mep_y.push_back(half_y[static_cast<std::size_t>(i)]);
    }
    for (std::size_t i = 1; i < half_x.size(); ++i) {
      theo_mep_x.push_back(half_x[i]);
      theo_mep_y.push_back(half_y[i]);
    }
  }

  std::println(std::cout, "\n  Theoretical MEP: {} points (steepest descent from saddle)", theo_mep_x.size());

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  {
    // Simplified path data.
    std::vector<double> path_x_simp, path_y_simp, alpha_simp_vec, energy_simp_vec;
    auto alpha_simp = algorithms::string_method::arc_lengths(result_simplified.images);
    for (std::size_t j = 0; j < result_simplified.images.size(); ++j) {
      path_x_simp.push_back(result_simplified.images[j].x[0](0));
      path_y_simp.push_back(result_simplified.images[j].x[1](0));
      alpha_simp_vec.push_back(alpha_simp[j]);
      energy_simp_vec.push_back(result_simplified.images[j].energy);
    }

    // Full path data.
    std::vector<double> path_x_full, path_y_full, alpha_full_vec, energy_full_vec;
    for (std::size_t j = 0; j < result_full.images.size(); ++j) {
      path_x_full.push_back(result_full.images[j].x[0](0));
      path_y_full.push_back(result_full.images[j].x[1](0));
      alpha_full_vec.push_back(alpha[j]);
      energy_full_vec.push_back(result_full.images[j].energy);
    }

    plot::make_plots({
        .path_x_init = path_x_init,
        .path_y_init = path_y_init,
        .path_x_simplified = path_x_simp,
        .path_y_simplified = path_y_simp,
        .alpha_simplified = alpha_simp_vec,
        .energy_simplified = energy_simp_vec,
        .path_x_full = path_x_full,
        .path_y_full = path_y_full,
        .alpha_full = alpha_full_vec,
        .energy_full = energy_full_vec,
        .iter_history = iter_history,
        .error_history = error_history,
        .theo_mep_x = theo_mep_x,
        .theo_mep_y = theo_mep_y,
    });
  }
#endif

  std::println(std::cout, "\nDone.");
}
