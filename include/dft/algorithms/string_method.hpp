#ifndef DFT_ALGORITHMS_STRING_METHOD_HPP
#define DFT_ALGORITHMS_STRING_METHOD_HPP

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <vector>

namespace dft::algorithms::string_method {

  // A single image along the minimum free-energy path.

  struct Image {
    std::vector<arma::vec> x;
    double energy{0.0};
  };

  // Result of the string method computation.

  struct StringResult {
    std::vector<Image> images;
    int iterations{0};
    double final_error{0.0};
    bool converged{false};
  };

  // Energy function: given x, returns (energy, forces).

  using ForceFunction = std::function<std::pair<double, std::vector<arma::vec>>(const std::vector<arma::vec>&)>;

  // Relaxation function: given x, returns (relaxed_x, energy).

  using RelaxFunction =
      std::function<std::pair<std::vector<arma::vec>, double>(const std::vector<arma::vec>&)>;

  // Optional callback invoked after each outer iteration.
  // Return true to stop early.

  using IterationCallback = std::function<bool(int iteration, double error, const std::vector<Image>&)>;

  // Initialize images via linear interpolation between two endpoints.

  [[nodiscard]] inline auto
  linear_interpolation(const std::vector<arma::vec>& initial, const std::vector<arma::vec>& final_state, int num_images)
      -> std::vector<Image> {
    std::vector<Image> images(num_images);
    images.front().x = initial;
    images.back().x = final_state;

    for (int j = 1; j < num_images - 1; ++j) {
      double f = static_cast<double>(j) / static_cast<double>(num_images - 1);
      images[j].x.resize(initial.size());
      for (std::size_t s = 0; s < initial.size(); ++s) {
        images[j].x[s] = (1.0 - f) * initial[s] + f * final_state[s];
      }
    }

    return images;
  }

  // Euclidean arc-length parameter along the path.

  [[nodiscard]] inline auto arc_lengths(const std::vector<Image>& images) -> std::vector<double> {
    std::vector<double> alpha(images.size(), 0.0);
    for (std::size_t j = 1; j < images.size(); ++j) {
      double dist_sq = 0.0;
      for (std::size_t s = 0; s < images[j].x.size(); ++s) {
        arma::vec diff = images[j].x[s] - images[j - 1].x[s];
        dist_sq += arma::dot(diff, diff);
      }
      alpha[j] = alpha[j - 1] + std::sqrt(dist_sq);
    }
    return alpha;
  }

  // Redistribute images at equal arc-length spacing via linear interpolation.
  // Endpoints are held fixed.

  inline void reparametrize(std::vector<Image>& images, int passes = 4) {
    int n = static_cast<int>(images.size());
    if (n < 3) {
      return;
    }

    for (int pass = 0; pass < passes; ++pass) {
      auto alpha = arc_lengths(images);

      if (alpha.back() <= 0.0) {
        return;
      }

      double dl = alpha.back() / static_cast<double>(n - 1);

      // Find bracketing interval for each interior image.
      std::vector<int> intervals(n, 0);
      for (int j = 1; j < n - 1; ++j) {
        double target = j * dl;
        for (int k = 0; k < n - 1; ++k) {
          if (target >= alpha[k] && target < alpha[k + 1]) {
            intervals[j] = k;
            break;
          }
        }
      }

      // Interpolate pointwise for each component at each position.
      for (std::size_t s = 0; s < images.front().x.size(); ++s) {
        long n_points = static_cast<long>(images.front().x[s].n_elem);

        for (long pos = 0; pos < n_points; ++pos) {
          std::vector<double> y(n);
          for (int j = 0; j < n; ++j) {
            y[j] = images[j].x[s](pos);
          }

          for (int j = 1; j < n - 1; ++j) {
            double target = j * dl;
            int k = intervals[j];
            double h = alpha[k + 1] - alpha[k];
            if (h <= 0.0) {
              continue;
            }
            double t = (target - alpha[k]) / h;
            images[j].x[s](pos) = (1.0 - t) * y[k] + t * y[k + 1];
          }
        }
      }
    }
  }

  // The simplified string method (E, Ren, Vanden-Eijnden, J. Chem. Phys.
  // 126, 164103, 2007) for finding minimum free-energy paths.
  //
  // The algorithm alternates:
  //   1. Relaxation of each interior image (via user-provided relax_fn)
  //   2. Arc-length reparametrization to maintain equal spacing
  //
  // The user injects the relaxation method (DDFT, FIRE, Picard, etc.)
  // through the RelaxFunction callback.
  //
  // When the full string method is desired (E, Ren, Vanden-Eijnden,
  // Phys. Rev. B 66, 052301, 2002), use the 4-argument overload of
  // find_pathway (without relax_fn).  This evolves images using only
  // the perpendicular component of the force, which converges the path
  // to zero perpendicular force (the MEP condition) by construction.

  struct StringMethod {
    double tolerance{1e-4};
    int max_iterations{20};
    int reparametrize_passes{4};
    int log_interval{1};
    int evolution_steps{10};
    double evolution_alpha{0.01};
    IterationCallback on_iteration{};

    // Full string method: perpendicular-force evolution (no relax_fn).

    [[nodiscard]] auto find_pathway(
        const std::vector<arma::vec>& initial,
        const std::vector<arma::vec>& final_state,
        int num_images,
        const ForceFunction& energy_fn
    ) const -> StringResult;

    // Simplified string method: user-provided relaxation + reparametrize.

    [[nodiscard]] auto find_pathway(
        const std::vector<arma::vec>& initial,
        const std::vector<arma::vec>& final_state,
        int num_images,
        const ForceFunction& energy_fn,
        const RelaxFunction& relax_fn
    ) const -> StringResult;
  };

  // Full string method (E, Ren, Vanden-Eijnden, Phys. Rev. B 66,
  // 052301, 2002).  Each iteration:
  //   1. Compute the gradient at each interior image
  //   2. Project out the tangential component → keep only F_perp
  //   3. Evolve x_j -= alpha * F_perp for evolution_steps
  //   4. Reparametrize to equal arc-length spacing
  // Convergence: max |F_perp| across all interior images < tolerance.

  [[nodiscard]] inline auto StringMethod::find_pathway(
      const std::vector<arma::vec>& initial,
      const std::vector<arma::vec>& final_state,
      int num_images,
      const ForceFunction& energy_fn
  ) const -> StringResult {
    auto images = linear_interpolation(initial, final_state, num_images);

    // Evaluate initial energies.

    for (auto& img : images) {
      auto [energy, forces] = energy_fn(img.x);
      img.energy = energy;
    }

    if (log_interval > 0) {
      std::println(std::cout, "  {:>6s}  {:>14s}  {:>14s}  {:>14s}", "iter", "max|F_perp|", "F_min", "F_max");
      std::println(std::cout, "  {}", std::string(54, '-'));
    }

    double err = 1.0 + tolerance;
    int iteration = 0;

    for (; iteration < max_iterations && err > tolerance; ++iteration) {
      // Evolve interior images using only the perpendicular force.

      for (int step = 0; step < evolution_steps; ++step) {
        for (int j = 1; j < num_images - 1; ++j) {
          auto [energy, forces] = energy_fn(images[j].x);

          // Path tangent from finite differences of neighbours.

          double tangent_norm_sq = 0.0;
          std::vector<arma::vec> tangent(images[j].x.size());
          for (std::size_t s = 0; s < images[j].x.size(); ++s) {
            tangent[s] = images[j + 1].x[s] - images[j - 1].x[s];
            tangent_norm_sq += arma::dot(tangent[s], tangent[s]);
          }

          // Tangential projection: F_parallel = (F . t) / |t|^2.

          double f_dot_t = 0.0;
          for (std::size_t s = 0; s < forces.size(); ++s) {
            f_dot_t += arma::dot(forces[s], tangent[s]);
          }

          // Perpendicular force: F_perp = F - (F.t / |t|^2) t.

          for (std::size_t s = 0; s < images[j].x.size(); ++s) {
            arma::vec f_perp = forces[s] - (f_dot_t / tangent_norm_sq) * tangent[s];
            images[j].x[s] -= evolution_alpha * f_perp;
          }
        }
      }

      // Reparametrize to equal arc-length spacing.

      reparametrize(images, reparametrize_passes);

      // Recompute energies after reparametrization.

      for (auto& img : images) {
        auto [energy, forces] = energy_fn(img.x);
        img.energy = energy;
      }

      // Convergence: max |F_perp| across interior images.

      err = 0.0;
      for (int j = 1; j < num_images - 1; ++j) {
        auto [energy, forces] = energy_fn(images[j].x);

        double tangent_norm_sq = 0.0;
        std::vector<arma::vec> tangent(images[j].x.size());
        for (std::size_t s = 0; s < images[j].x.size(); ++s) {
          tangent[s] = images[j + 1].x[s] - images[j - 1].x[s];
          tangent_norm_sq += arma::dot(tangent[s], tangent[s]);
        }

        double f_dot_t = 0.0;
        for (std::size_t s = 0; s < forces.size(); ++s) {
          f_dot_t += arma::dot(forces[s], tangent[s]);
        }

        double f_perp_sq = 0.0;
        for (std::size_t s = 0; s < forces.size(); ++s) {
          arma::vec f_perp = forces[s] - (f_dot_t / tangent_norm_sq) * tangent[s];
          f_perp_sq += arma::dot(f_perp, f_perp);
        }

        err = std::max(err, std::sqrt(f_perp_sq));
      }

      if (log_interval > 0 && (iteration % log_interval == 0 || err <= tolerance)) {
        double f_min = images.front().energy;
        double f_max = images.front().energy;
        for (const auto& img : images) {
          f_min = std::min(f_min, img.energy);
          f_max = std::max(f_max, img.energy);
        }

        std::println(std::cout, "  {:>6d}  {:>14.6e}  {:>14.6f}  {:>14.6f}", iteration, err, f_min, f_max);
      }

      if (on_iteration) {
        if (on_iteration(iteration, err, images)) {
          break;
        }
      }
    }

    return StringResult{
        .images = std::move(images),
        .iterations = iteration,
        .final_error = err,
        .converged = err <= tolerance,
    };
  }

  // Simplified string method (E, Ren, Vanden-Eijnden, J. Chem. Phys.
  // 126, 164103, 2007).  Uses full-gradient relaxation via user callback
  // followed by reparametrization.  Convergence: RMS energy change.

  [[nodiscard]] inline auto StringMethod::find_pathway(
      const std::vector<arma::vec>& initial,
      const std::vector<arma::vec>& final_state,
      int num_images,
      const ForceFunction& energy_fn,
      const RelaxFunction& relax_fn
  ) const -> StringResult {
    auto images = linear_interpolation(initial, final_state, num_images);

    // Evaluate initial energies.
    for (auto& img : images) {
      auto [energy, forces] = energy_fn(img.x);
      img.energy = energy;
    }

    std::vector<double> f_old(num_images);
    for (int j = 0; j < num_images; ++j) {
      f_old[j] = images[j].energy;
    }

    if (log_interval > 0) {
      std::println(std::cout, "  {:>6s}  {:>14s}  {:>14s}  {:>14s}", "iter", "error", "F_min", "F_max");
      std::println(std::cout, "  {}", std::string(54, '-'));
    }

    double err = 1.0 + tolerance;
    int iteration = 0;

    for (; iteration < max_iterations && err > tolerance; ++iteration) {
      // Relax interior images.
      for (int j = 1; j < num_images - 1; ++j) {
        auto [relaxed_x, relaxed_energy] = relax_fn(images[j].x);
        images[j].x = std::move(relaxed_x);
        images[j].energy = relaxed_energy;
      }

      // Reparametrize to equal arc-length spacing.
      reparametrize(images, reparametrize_passes);

      // Recompute energies after reparametrization.
      for (auto& img : images) {
        auto [energy, forces] = energy_fn(img.x);
        img.energy = energy;
      }

      // Convergence: RMS change in free energies (interior images only).
      double sum_sq = 0.0;
      for (int j = 1; j < num_images - 1; ++j) {
        double df = images[j].energy - f_old[j];
        sum_sq += df * df;
      }
      err = std::sqrt(sum_sq / static_cast<double>(num_images - 2));

      for (int j = 0; j < num_images; ++j) {
        f_old[j] = images[j].energy;
      }

      if (log_interval > 0 && (iteration % log_interval == 0 || err <= tolerance)) {
        double f_min = images.front().energy;
        double f_max = images.front().energy;
        for (const auto& img : images) {
          f_min = std::min(f_min, img.energy);
          f_max = std::max(f_max, img.energy);
        }

        std::println(std::cout, "  {:>6d}  {:>14.6e}  {:>14.6f}  {:>14.6f}", iteration, err, f_min, f_max);
      }

      if (on_iteration) {
        if (on_iteration(iteration, err, images)) {
          break;
        }
      }
    }

    return StringResult{
        .images = std::move(images),
        .iterations = iteration,
        .final_error = err,
        .converged = err <= tolerance,
    };
  }

}  // namespace dft::algorithms::string_method

#endif  // DFT_ALGORITHMS_STRING_METHOD_HPP
