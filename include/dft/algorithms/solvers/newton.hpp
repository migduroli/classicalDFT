#ifndef DFT_ALGORITHMS_SOLVERS_NEWTON_HPP
#define DFT_ALGORITHMS_SOLVERS_NEWTON_HPP

#include "dft/algorithms/solvers/jacobian.hpp"

#include <armadillo>
#include <concepts>
#include <print>

namespace dft::algorithms::solvers {

  template <typename J>
  concept JacobianFunction = requires(J jac, const arma::vec& x) {
    { jac(x) } -> std::convertible_to<arma::mat>;
  };

  struct SolverResult {
    arma::vec solution;
    int iterations;
    double final_norm;
    bool converged;
  };

  struct Newton {
    int max_iterations{100};
    double tolerance{1e-6};
    bool verbose{false};

    // Newton-Raphson with analytical Jacobian.
    template <VectorFunction Func, JacobianFunction JacFunc>
    [[nodiscard]] auto solve(arma::vec x, Func&& f, JacFunc&& J) const -> SolverResult {
      for (int k = 0; k < max_iterations; ++k) {
        const arma::vec fk = f(x);
        const double norm_fk = arma::norm(fk);

        if (verbose) {
          std::println("newton: iter={} ||f||={}", k, norm_fk);
        }

        if (norm_fk < tolerance) {
          return SolverResult{.solution = std::move(x), .iterations = k, .final_norm = norm_fk, .converged = true};
        }

        const arma::mat Jk = J(x);
        arma::vec delta;
        if (!arma::solve(delta, Jk, fk, arma::solve_opts::no_approx)) {
          return SolverResult{.solution = std::move(x), .iterations = k, .final_norm = norm_fk, .converged = false};
        }
        x -= delta;
      }

      const double final_norm = arma::norm(f(x));
      return SolverResult{
          .solution = std::move(x),
          .iterations = max_iterations,
          .final_norm = final_norm,
          .converged = final_norm < tolerance,
      };
    }

    // Newton-Raphson with automatic numerical Jacobian.
    template <VectorFunction Func>
    [[nodiscard]] auto solve(arma::vec x, Func&& f) const -> SolverResult {
      auto J = [&f](const arma::vec& x_) { return numerical_jacobian(f, x_); };
      return solve(std::move(x), std::forward<Func>(f), J);
    }
  };

}  // namespace dft::algorithms::solvers

#endif  // DFT_ALGORITHMS_SOLVERS_NEWTON_HPP
