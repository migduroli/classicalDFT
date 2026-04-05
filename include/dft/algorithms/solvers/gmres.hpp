#ifndef DFT_ALGORITHMS_SOLVERS_GMRES_HPP
#define DFT_ALGORITHMS_SOLVERS_GMRES_HPP

#include <armadillo>
#include <concepts>
#include <cmath>

namespace dft::algorithms::solvers {

  // A linear operator is any callable that maps vec -> vec.

  template <typename A>
  concept LinearOperator = requires(A a, const arma::vec& v) {
    { a(v) } -> std::convertible_to<arma::vec>;
  };

  struct GMRESResult {
    arma::vec solution;
    int iterations;
    double final_residual;
    bool converged;
  };

  // Restarted GMRES(m) for solving A*x = b using only matrix-vector products.
  // A is a LinearOperator (callable), not an explicit matrix.

  struct GMRES {
    int max_iterations{100};
    int restart{30};
    double tolerance{1e-6};

    template <LinearOperator Op>
    [[nodiscard]] auto solve(Op&& A, const arma::vec& b, arma::vec x0 = {}) const -> GMRESResult {
      const arma::uword n = b.n_elem;

      if (x0.empty()) {
        x0.zeros(n);
      }

      int total_iters = 0;

      for (int cycle = 0; cycle < max_iterations; ++cycle) {
        arma::vec r = b - A(x0);
        double beta = arma::norm(r);

        if (beta < tolerance) {
          return {.solution = std::move(x0), .iterations = total_iters, .final_residual = beta, .converged = true};
        }

        int m = std::min(restart, static_cast<int>(n));

        // Arnoldi basis V (columns) and upper Hessenberg H.
        arma::mat V(n, static_cast<arma::uword>(m + 1), arma::fill::zeros);
        arma::mat H(static_cast<arma::uword>(m + 1), static_cast<arma::uword>(m), arma::fill::zeros);
        V.col(0) = r / beta;

        // Givens rotation components.
        arma::vec cs(static_cast<arma::uword>(m), arma::fill::zeros);
        arma::vec sn(static_cast<arma::uword>(m), arma::fill::zeros);
        arma::vec g(static_cast<arma::uword>(m + 1), arma::fill::zeros);
        g(0) = beta;

        int j = 0;
        for (; j < m; ++j) {
          auto ju = static_cast<arma::uword>(j);
          ++total_iters;

          // Arnoldi step.
          arma::vec w = A(V.col(ju));
          for (arma::uword i = 0; i <= ju; ++i) {
            H(i, ju) = arma::dot(w, V.col(i));
            w -= H(i, ju) * V.col(i);
          }
          H(ju + 1, ju) = arma::norm(w);

          bool breakdown = H(ju + 1, ju) < 1e-30;
          if (!breakdown) {
            V.col(ju + 1) = w / H(ju + 1, ju);
          }

          // Apply previous Givens rotations to column j of H.
          for (arma::uword i = 0; i < ju; ++i) {
            double h_i = H(i, ju);
            double h_i1 = H(i + 1, ju);
            H(i, ju) = cs(i) * h_i + sn(i) * h_i1;
            H(i + 1, ju) = -sn(i) * h_i + cs(i) * h_i1;
          }

          // Compute Givens rotation for row j.
          double a = H(ju, ju);
          double b_val = H(ju + 1, ju);
          double r_val = std::hypot(a, b_val);
          cs(ju) = a / r_val;
          sn(ju) = b_val / r_val;

          H(ju, ju) = r_val;
          H(ju + 1, ju) = 0.0;

          // Update RHS.
          double g_j = g(ju);
          g(ju) = cs(ju) * g_j;
          g(ju + 1) = -sn(ju) * g_j;

          double residual = std::abs(g(ju + 1));
          if (residual < tolerance || breakdown) {
            ++j;
            break;
          }
        }

        // Back-substitution: solve H(0:j-1, 0:j-1) * y = g(0:j-1).
        auto jj = static_cast<arma::uword>(j);
        if (jj == 0) {
          return {.solution = std::move(x0), .iterations = total_iters, .final_residual = beta, .converged = false};
        }
        arma::vec y = arma::solve(
            arma::trimatu(H.submat(0, 0, jj - 1, jj - 1)),
            g.head(jj));

        // Update solution.
        x0 += V.head_cols(jj) * y;

        double res = arma::norm(b - A(x0));
        if (res < tolerance) {
          return {.solution = std::move(x0), .iterations = total_iters, .final_residual = res, .converged = true};
        }
      }

      double final_res = arma::norm(b - A(x0));
      return {
          .solution = std::move(x0),
          .iterations = total_iters,
          .final_residual = final_res,
          .converged = final_res < tolerance,
      };
    }
  };

}  // namespace dft::algorithms::solvers

#endif  // DFT_ALGORITHMS_SOLVERS_GMRES_HPP
