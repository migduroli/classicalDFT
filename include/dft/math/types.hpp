#ifndef DFT_MATH_TYPES_HPP
#define DFT_MATH_TYPES_HPP

#include <armadillo>
#include <functional>

namespace dft::math {

  // Type-erased Hessian-vector product operator for eigenvalue solvers
  // and saddle-point methods.
  struct HessianOperator {
    std::function<void(const arma::vec&, arma::vec&)> hessian_dot_v;
    arma::uword dimension;
  };

}  // namespace dft::math

#endif  // DFT_MATH_TYPES_HPP
