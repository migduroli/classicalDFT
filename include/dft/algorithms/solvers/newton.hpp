#ifndef DFT_ALGORITHMS_SOLVERS_NEWTON_HPP
#define DFT_ALGORITHMS_SOLVERS_NEWTON_HPP

#include "dft/algorithms/solvers/gmres.hpp"
#include "dft/algorithms/solvers/jacobian.hpp"

#include <armadillo>
#include <concepts>
#include <print>

namespace dft::algorithms::solvers {

  template <typename J>
  concept JacobianFunction = requires(J jac, const arma::vec& x) {
    { jac(x) } -> std::convertible_to<arma::mat>;
  };

  // A Jacobian-vector product factory: given x, returns a LinearOperator v -> J(x)*v.

  template <typename F>
  concept JvpFactory = requires(F factory, const arma::vec& x) {
    { factory(x) } -> LinearOperator;
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
    GMRES gmres{};

    // Newton-Raphson with analytical Jacobian (dense direct solve).
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

    // Newton-Raphson with automatic numerical Jacobian (dense direct solve).
    template <VectorFunction Func> [[nodiscard]] auto solve(arma::vec x, Func&& f) const -> SolverResult {
      auto J = [&f](const arma::vec& x_) {
        return numerical_jacobian(f, x_);
      };
      return solve(std::move(x), std::forward<Func>(f), J);
    }

    // Newton-GMRES: matrix-free Newton using Jacobian-vector products.
    // jvp_factory(x) returns a callable Jv: vec -> vec (the action of J(x) on v).
    template <VectorFunction Func, JvpFactory JvpFac>
    [[nodiscard]] auto solve_matrix_free(arma::vec x, Func&& f, JvpFac&& jvp_factory) const -> SolverResult {
      for (int k = 0; k < max_iterations; ++k) {
        const arma::vec fk = f(x);
        const double norm_fk = arma::norm(fk);

        if (verbose) {
          std::println("newton-gmres: iter={} ||f||={}", k, norm_fk);
        }

        if (norm_fk < tolerance) {
          return SolverResult{.solution = std::move(x), .iterations = k, .final_norm = norm_fk, .converged = true};
        }

        // Inexact Newton: adaptive GMRES tolerance (forcing term η * ||f||)
        // so the linear solve accuracy scales with the nonlinear residual.
        GMRES linear_solver = gmres;
        linear_solver.tolerance = gmres.tolerance * norm_fk;

        auto Jv = jvp_factory(x);
        auto result = linear_solver.solve(std::move(Jv), fk);
        x -= result.solution;
      }

      const double final_norm = arma::norm(f(x));
      return SolverResult{
          .solution = std::move(x),
          .iterations = max_iterations,
          .final_norm = final_norm,
          .converged = final_norm < tolerance,
      };
    }

    // Newton-GMRES with automatic finite-difference Jacobian-vector products:
    //   J(x)*v ≈ (f(x + eps*v) - f(x)) / eps
    template <VectorFunction Func>
    [[nodiscard]] auto solve_matrix_free(arma::vec x, Func&& f, double jvp_epsilon = 1e-7) const -> SolverResult {
      auto jvp_factory = [&f, jvp_epsilon](const arma::vec& x_) {
        const arma::vec fx = f(x_);
        return [&f, x_, fx, jvp_epsilon](const arma::vec& v) -> arma::vec {
          return (f(x_ + jvp_epsilon * v) - fx) / jvp_epsilon;
        };
      };
      return solve_matrix_free(std::move(x), std::forward<Func>(f), jvp_factory);
    }
  };

} // namespace dft::algorithms::solvers

#endif // DFT_ALGORITHMS_SOLVERS_NEWTON_HPP
