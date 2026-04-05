#ifndef DFT_ALGORITHMS_PICARD_HPP
#define DFT_ALGORITHMS_PICARD_HPP

#include <armadillo>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <vector>

namespace dft::algorithms::picard {

  // Force function: given densities, returns (grand potential, forces).
  // Same signature as fire::ForceFunction and ddft::ForceCallback.

  using ForceFunction = std::function<std::pair<double, std::vector<arma::vec>>(const std::vector<arma::vec>&)>;

  // Optional projection applied after each density update.
  // Returns a copy with the constraint enforced (e.g. fixed total mass).

  using Constraint = std::function<std::vector<arma::vec>(const std::vector<arma::vec>&)>;

  struct PicardResult {
    std::vector<arma::vec> densities;
    double grand_potential;
    double residual;
    int iterations;
    bool converged;
  };

  struct Picard {
    double mixing{0.01};
    double min_density{1e-30};
    double tolerance{1e-6};
    int max_iterations{5000};
    int log_interval{500};

    [[nodiscard]] auto solve(
        std::vector<arma::vec> densities,
        const ForceFunction& compute,
        double cell_volume,
        const Constraint& constraint = {}
    ) const -> PicardResult;
  };

  // Picard (self-consistent field) iteration for finding stationary
  // points of the grand potential Omega[rho].
  //
  // The force from functionals::total satisfies:
  //   force(r) = [ln(rho) + delta_F_ex/(kT delta_rho) - mu/kT] * dV
  //
  // The log-space Picard update is:
  //   rho_{n+1} = rho_n * exp(-alpha * force_n / dV)
  //
  // This converges to the nearest minimum of Omega. To find saddle
  // points (e.g. the critical nucleus), pass a constraint that fixes
  // the total mass (see minimization::fixed_mass_constraint).
  //
  // When a constraint is active, the residual may plateau at a
  // nonzero value (the Lagrange multiplier). Convergence is then
  // detected when Omega stops changing between iterations.

  [[nodiscard]] inline auto Picard::solve(
      std::vector<arma::vec> densities,
      const ForceFunction& compute,
      double cell_volume,
      const Constraint& constraint
  ) const -> PicardResult {
    double residual = 0.0;
    double omega = 0.0;
    double omega_prev = std::numeric_limits<double>::max();
    int iter = 0;

    if (log_interval > 0) {
      std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}\n", "iter", "Omega", "||force||");
      std::cout << "  " << std::string(38, '-') << "\n";
    }

    for (iter = 0; iter < max_iterations; ++iter) {
      auto [energy, forces] = compute(densities);
      omega = energy;

      double total_dof = 0.0;
      double sum_f2 = 0.0;
      for (const auto& f : forces) {
        sum_f2 += arma::dot(f, f);
        total_dof += static_cast<double>(f.n_elem);
      }
      residual = std::sqrt(sum_f2 / total_dof);

      if (log_interval > 0 && (iter % log_interval == 0)) {
        std::cout << std::format("  {:>6d}  {:>14.6f}  {:>14.6e}\n", iter, omega, residual);
      }

      // Converged when forces vanish (unconstrained) or when Omega
      // stops changing (constrained, where residual plateaus).
      bool force_converged = residual < tolerance;
      bool omega_converged = (iter > 0) && std::abs(omega - omega_prev) < tolerance;
      if (force_converged || omega_converged) {
        if (log_interval > 0) {
          std::cout << std::format("  {:>6d}  {:>14.6f}  {:>14.6e}\n", iter, omega, residual);
        }
        break;
      }
      omega_prev = omega;

      for (std::size_t s = 0; s < densities.size(); ++s) {
        densities[s] %= arma::exp(-mixing * forces[s] / cell_volume);
        densities[s] = arma::clamp(densities[s], min_density, arma::datum::inf);
      }

      if (constraint) {
        densities = constraint(densities);
      }
    }

    return PicardResult{
        .densities = std::move(densities),
        .grand_potential = omega,
        .residual = residual,
        .iterations = iter,
        .converged = residual < tolerance || (iter > 0 && std::abs(omega - omega_prev) < tolerance),
    };
  }

} // namespace dft::algorithms::picard

#endif // DFT_ALGORITHMS_PICARD_HPP
