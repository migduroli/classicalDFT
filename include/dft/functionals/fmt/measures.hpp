#ifndef DFT_FUNCTIONALS_FMT_MEASURES_HPP
#define DFT_FUNCTIONALS_FMT_MEASURES_HPP

#include <armadillo>
#include <numbers>

namespace dft::functionals::fmt {

  // Pre-computed contractions of the vector and tensor weighted densities.
  struct InnerProducts {
    double dot_v0_v1{0.0};
    double dot_v1_v1{0.0};
    double trace_T2{0.0};
    double trace_T3{0.0};
    double quadratic_form{0.0};
  };

  // The 19-component set of weighted densities in Fundamental Measure Theory.
  // Scalars: eta, n0, n1, n2. Vectors: v0, v1. Tensor: T (symmetric 3x3).
  struct Measures {
    double eta{0.0};
    double n0{0.0};
    double n1{0.0};
    double n2{0.0};

    arma::rowvec3 v0 = arma::zeros<arma::rowvec>(3);
    arma::rowvec3 v1 = arma::zeros<arma::rowvec>(3);

    arma::mat33 T = arma::zeros<arma::mat>(3, 3);

    InnerProducts products;
  };

  // Computes all inner products from the primary fields and returns the result.
  [[nodiscard]] inline auto inner_products(const Measures& m) -> InnerProducts {
    InnerProducts p;
    p.dot_v0_v1 = arma::dot(m.v0, m.v1);
    p.dot_v1_v1 = arma::dot(m.v1, m.v1);
    arma::mat33 T2 = m.T * m.T;
    p.trace_T2 = arma::trace(T2);
    p.trace_T3 = arma::trace(T2 * m.T);
    p.quadratic_form = arma::as_scalar(m.v1 * m.T * m.v1.t());
    return p;
  }

  // Factory for a uniform (homogeneous) fluid with given density and diameter.
  // eta = (pi/6) rho d^3, n2 = pi rho d^2, n1 = rho d/2, n0 = rho.
  // Vectors vanish. T_ij = (n2/3) delta_ij.
  [[nodiscard]] inline auto make_uniform_measures(double density, double diameter) -> Measures {
    double d = diameter;
    double r = 0.5 * d;
    double n2 = std::numbers::pi * density * d * d;
    double t_diag = n2 / 3.0;

    Measures m;
    m.eta = (std::numbers::pi / 6.0) * density * d * d * d;
    m.n2 = n2;
    m.n1 = r * density;
    m.n0 = density;
    m.v0.zeros();
    m.v1.zeros();
    m.T = arma::diagmat(arma::rowvec3{t_diag, t_diag, t_diag});
    m.products = inner_products(m);
    return m;
  }

  // Partial derivatives of the free energy density Phi with respect to
  // each weighted density. Same layout as Measures but every field
  // holds dPhi/d(field) instead of the weighted density itself.

  struct MeasureDerivatives {
    double d_eta{0.0};
    double d_n0{0.0};
    double d_n1{0.0};
    double d_n2{0.0};

    arma::rowvec3 d_v0 = arma::zeros<arma::rowvec>(3);
    arma::rowvec3 d_v1 = arma::zeros<arma::rowvec>(3);

    arma::mat33 d_T = arma::zeros<arma::mat>(3, 3);
  };

}  // namespace dft::functionals::fmt

#endif  // DFT_FUNCTIONALS_FMT_MEASURES_HPP
