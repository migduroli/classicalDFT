#ifndef DFT_ALGORITHMS_SOLVERS_JACOBIAN_HPP
#define DFT_ALGORITHMS_SOLVERS_JACOBIAN_HPP

#include <armadillo>
#include <cmath>
#include <concepts>

namespace dft::algorithms::solvers {

  template <typename F>
  concept VectorFunction = requires(F f, const arma::vec& x) {
    { f(x) } -> std::convertible_to<arma::vec>;
  };

  // Central-difference O(h^2) numerical Jacobian.
  // Column j of J = (f(x + eps*e_j) - f(x - eps*e_j)) / (2*eps).
  template <VectorFunction Func>
  [[nodiscard]] auto numerical_jacobian(Func&& f, const arma::vec& x, double epsilon = 1e-7) -> arma::mat {
    const arma::vec f0 = f(x);
    const arma::uword m = f0.n_elem;
    const arma::uword n = x.n_elem;

    arma::mat J(m, n);
    arma::vec x_plus = x;
    arma::vec x_minus = x;

    for (arma::uword j = 0; j < n; ++j) {
      x_plus(j) = x(j) + epsilon;
      x_minus(j) = x(j) - epsilon;

      const arma::vec fp = f(x_plus);
      const arma::vec fm = f(x_minus);
      J.col(j) = (fp - fm) / (2.0 * epsilon);

      x_plus(j) = x(j);
      x_minus(j) = x(j);
    }

    return J;
  }

} // namespace dft::algorithms::solvers

#endif // DFT_ALGORITHMS_SOLVERS_JACOBIAN_HPP
