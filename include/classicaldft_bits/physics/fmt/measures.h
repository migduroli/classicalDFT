#ifndef CLASSICALDFT_PHYSICS_FMT_MEASURES_H
#define CLASSICALDFT_PHYSICS_FMT_MEASURES_H

#include <armadillo>
#include <numbers>

namespace dft_core::physics::fmt {

  /**
   * @brief The 19-component set of weighted densities used in Fundamental Measure Theory.
   *
   * Primary fields: $\eta$, $n_0$, $n_1$, $n_2$ (scalars), $\mathbf{v}_1$,
   * $\mathbf{v}_2$ (vectors), $\mathbf{T}$ (symmetric tensor). Derived
   * quantities are computed on demand by `compute_derived()`.
   */
  struct Measures {
    // ── Primary fields ────────────────────────────────────────────────────

    double eta = 0.0;
    double n0 = 0.0;
    double n1 = 0.0;
    double n2 = 0.0;

    arma::rowvec3 v1 = arma::zeros<arma::rowvec>(3);
    arma::rowvec3 v2 = arma::zeros<arma::rowvec>(3);

    arma::mat33 T = arma::zeros<arma::mat>(3, 3);

    // ── Derived quantities (set by compute_derived) ───────────────────────

    double v1_dot_v2 = 0.0;
    double v2_dot_v2 = 0.0;
    double v_T_v = 0.0;
    double trace_T2 = 0.0;
    double trace_T3 = 0.0;

    /**
     * @brief Computes all derived quantities from the primary fields.
     *
     * Must be called after the primary fields have been populated.
     */
    void compute_derived() {
      v1_dot_v2 = arma::dot(v1, v2);
      v2_dot_v2 = arma::dot(v2, v2);
      arma::mat33 T2 = T * T;
      trace_T2 = arma::trace(T2);
      trace_T3 = arma::trace(T2 * T);
      v_T_v = arma::as_scalar(v2 * T * v2.t());
    }

    /**
     * @brief Factory for a uniform (homogeneous) fluid.
     *
     * In a uniform fluid:
     *   $\eta = \frac{\pi}{6} \rho d^3$, $n_2 = \pi \rho d^2$,
     *   $n_1 = \rho d / 2$, $n_0 = \rho$,
     *   all vector components vanish, and $T_{ij} = (n_2 / 3) \delta_{ij}$.
     *
     * @param density Number density $\rho$.
     * @param diameter Hard-sphere diameter $d$.
     */
    [[nodiscard]] static Measures uniform(double density, double diameter) {
      Measures m;
      double d = diameter;
      double R = 0.5 * d;

      m.eta = (std::numbers::pi / 6.0) * density * d * d * d;
      m.n2 = std::numbers::pi * density * d * d;
      m.n1 = R * density;
      m.n0 = density;

      m.v1.zeros();
      m.v2.zeros();

      double T_diag = m.n2 / 3.0;
      m.T.zeros();
      m.T(0, 0) = T_diag;
      m.T(1, 1) = T_diag;
      m.T(2, 2) = T_diag;

      m.compute_derived();
      return m;
    }
  };

}  // namespace dft_core::physics::fmt

#endif  // CLASSICALDFT_PHYSICS_FMT_MEASURES_H
