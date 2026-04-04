#ifndef DFT_ALGORITHMS_HESSIAN_HPP
#define DFT_ALGORITHMS_HESSIAN_HPP

#include "dft/functionals/functionals.hpp"
#include "dft/grid.hpp"
#include "dft/init.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <cmath>
#include <functional>
#include <vector>

namespace dft::algorithms {

  // Force function signature: rho -> (Omega, dOmega/drho * dV).
  // Same convention as functionals::total().forces.

  using GrandPotentialForce = std::function<std::pair<double, arma::vec>(const arma::vec&)>;

  // Finite-difference Hessian-vector product:
  //   H * v ≈ (F(rho + eps*v) - F(rho)) / eps
  // where F(rho) = dOmega/drho * dV (the force vector).
  //
  // Requires one extra force evaluation per call.

  [[nodiscard]] inline auto hessian_times_vector(
      const GrandPotentialForce& force_fn,
      const arma::vec& rho,
      const arma::vec& forces_at_rho,
      const arma::vec& v,
      double eps = 1e-6
  ) -> arma::vec {
    auto [_, forces_shifted] = force_fn(rho + eps * v);
    return (forces_shifted - forces_at_rho) / eps;
  }

}  // namespace dft::algorithms

#endif  // DFT_ALGORITHMS_HESSIAN_HPP
