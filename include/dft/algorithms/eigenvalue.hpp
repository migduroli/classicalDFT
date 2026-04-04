#ifndef DFT_ALGORITHMS_EIGENVALUE_HPP
#define DFT_ALGORITHMS_EIGENVALUE_HPP

#include "dft/algorithms/hessian.hpp"

#include <armadillo>
#include <cmath>
#include <format>
#include <iostream>

namespace dft::algorithms {

  struct EigenvalueConfig {
    double tolerance{1e-4};
    int max_iterations{300};
    double hessian_eps{1e-6};
    int log_interval{0};
  };

  struct EigenvalueResult {
    arma::vec eigenvector;
    double eigenvalue{0.0};
    int iterations{0};
    bool converged{false};
  };

  // Find the smallest eigenvalue and its eigenvector of the Hessian
  // d²Omega/(drho_i drho_j) using the LOBPCG algorithm.
  //
  // At each iteration we build an orthonormal basis {v, w, p} via
  // Gram-Schmidt, project the Hessian into this subspace, and solve
  // the small standard symmetric eigenvalue problem.
  //
  // Each iteration requires one Hessian-vector product (one extra
  // force evaluation via finite differences).

  [[nodiscard]] inline auto smallest_eigenvalue(
      const GrandPotentialForce& force_fn,
      const arma::vec& rho,
      const EigenvalueConfig& config = {},
      const arma::vec& initial_guess = {}
  ) -> EigenvalueResult {
    auto [omega, forces] = force_fn(rho);
    arma::uword n = rho.n_elem;

    arma::vec v = initial_guess.n_elem == n
        ? initial_guess
        : arma::randn(n);
    v /= arma::norm(v);

    if (config.log_interval > 0) {
      std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}\n",
                               "iter", "eigenvalue", "residual");
      std::cout << "  " << std::string(40, '-') << "\n";
    }

    arma::vec hv = hessian_times_vector(force_fn, rho, forces, v, config.hessian_eps);
    double lambda = arma::dot(v, hv);
    arma::vec p_raw;  // Previous search direction (un-orthogonalized).
    arma::vec hp_raw; // Hessian times p_raw.

    bool converged = false;
    int iter = 0;

    for (; iter < config.max_iterations; ++iter) {
      arma::vec residual = hv - lambda * v;
      double res_norm = arma::norm(residual);

      if (config.log_interval > 0 && (iter % config.log_interval == 0 || res_norm < config.tolerance)) {
        std::cout << std::format("  {:>6d}  {:>14.6e}  {:>14.6e}\n", iter, lambda, res_norm);
      }

      if (res_norm < config.tolerance) {
        converged = true;
        break;
      }

      // Orthonormal basis via Gram-Schmidt.
      // q0 = v (already unit norm).
      arma::vec q0 = v;

      // q1 = residual orthogonalized against q0.
      arma::vec q1 = residual - arma::dot(residual, q0) * q0;
      double q1n = arma::norm(q1);
      if (q1n < 1e-30) break;
      q1 /= q1n;

      arma::vec hq0 = hv;
      arma::vec hq1 = hessian_times_vector(force_fn, rho, forces, q1, config.hessian_eps);

      int dim = 2;
      arma::vec q2;
      arma::vec hq2;

      if (p_raw.n_elem == n) {
        // Orthogonalize p against q0 and q1.
        q2 = p_raw - arma::dot(p_raw, q0) * q0 - arma::dot(p_raw, q1) * q1;
        double q2n = arma::norm(q2);
        if (q2n > 1e-10) {
          q2 /= q2n;
          hq2 = hessian_times_vector(force_fn, rho, forces, q2, config.hessian_eps);
          dim = 3;
        }
      }

      // Build the projected Hessian: H_proj = Q^T H Q (symmetric, since basis is orthonormal).
      arma::mat H_proj(dim, dim);
      H_proj(0, 0) = arma::dot(q0, hq0);
      H_proj(0, 1) = arma::dot(q0, hq1);
      H_proj(1, 0) = H_proj(0, 1);
      H_proj(1, 1) = arma::dot(q1, hq1);

      if (dim == 3) {
        H_proj(0, 2) = arma::dot(q0, hq2);
        H_proj(1, 2) = arma::dot(q1, hq2);
        H_proj(2, 0) = H_proj(0, 2);
        H_proj(2, 1) = H_proj(1, 2);
        H_proj(2, 2) = arma::dot(q2, hq2);
      }

      // Solve standard symmetric eigenvalue problem.
      arma::vec eigvals;
      arma::mat eigvecs;
      arma::eig_sym(eigvals, eigvecs, H_proj);
      arma::vec c = eigvecs.col(0);  // Smallest eigenvalue.

      // Reconstruct the new eigenvector in full space.
      arma::vec v_new = c(0) * q0 + c(1) * q1;
      arma::vec hv_new = c(0) * hq0 + c(1) * hq1;
      if (dim == 3) {
        v_new += c(2) * q2;
        hv_new += c(2) * hq2;
      }

      // Store conjugate direction for next iteration (before normalizing v_new).
      p_raw = v_new - v;
      hp_raw = hv_new - hv;

      v = v_new / arma::norm(v_new);
      // Recompute Hv exactly for numerical stability.
      hv = hessian_times_vector(force_fn, rho, forces, v, config.hessian_eps);
      lambda = arma::dot(v, hv);
    }

    return EigenvalueResult{
        .eigenvector = v,
        .eigenvalue = lambda,
        .iterations = iter,
        .converged = converged,
    };
  }

}  // namespace dft::algorithms

#endif  // DFT_ALGORITHMS_EIGENVALUE_HPP
