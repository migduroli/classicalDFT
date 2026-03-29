#ifndef DFT_MATH_HESSIAN_H
#define DFT_MATH_HESSIAN_H

#include <armadillo>

namespace dft::math {

  /**
   * @brief Abstract interface for Hessian-vector products $H \cdot v$
   *
   * Provides the operator interface needed by eigenvalue solvers and
   * saddle-point methods. The DFT Solver implements this interface
   * by computing $\sum_j (\partial^2 F / \partial \rho_i \partial \rho_j) v_j$.
   */
  class HessianOperator {
   public:
    virtual ~HessianOperator() = default;

    /**
     * @brief Total number of degrees of freedom
     */
    [[nodiscard]] virtual arma::uword dimension() const noexcept = 0;

    /**
     * @brief Compute the Hessian-vector product $H \cdot v$
     * @param v input vector (length = dimension())
     * @param result output vector, overwritten with $H \cdot v$
     */
    virtual void hessian_dot_v(const arma::vec& v, arma::vec& result) const = 0;
  };

}  // namespace dft::math

#endif  // DFT_MATH_HESSIAN_H
