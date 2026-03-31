#ifndef DFT_FUNCTIONAL_FMT_MEASURES_H
#define DFT_FUNCTIONAL_FMT_MEASURES_H

#include <armadillo>
#include <numbers>

namespace dft::functional::fmt {

  /**
   * @brief Scalar and vector inner products derived from the fundamental measures.
   *
   * These are computed from the primary fields and cached for reuse across
   * the free energy density and its derivatives.
   */
  struct InnerProducts {
    double dot_v0_v1 = 0.0;       ///< $\mathbf{v}_0 \cdot \mathbf{v}_1$
    double dot_v1_v1 = 0.0;       ///< $|\mathbf{v}_1|^2$
    double trace_T2 = 0.0;        ///< $\mathrm{Tr}(\mathbf{T}^2)$
    double trace_T3 = 0.0;        ///< $\mathrm{Tr}(\mathbf{T}^3)$
    double quadratic_form = 0.0;  ///< $\mathbf{v}_1^T \mathbf{T} \mathbf{v}_1^T$
  };

  /**
   * @brief The 19-component set of weighted densities used in Fundamental Measure Theory.
   *
   * Primary fields: $\eta$, $n_0$, $n_1$, $n_2$ (scalars), $\mathbf{v}_0$,
   * $\mathbf{v}_1$ (vectors), $\mathbf{T}$ (symmetric tensor). Inner products
   * are computed on demand by `compute_inner_products()`.
   *
   * Vector naming: $\mathbf{v}_0$ and $\mathbf{v}_1$ correspond to the literature's
   * $\mathbf{v}_1$ and $\mathbf{v}_2$ (the vectorial weighted densities associated
   * with the first and second scalar weighted densities).
   */
  struct Measures {
    // ── Primary fields ────────────────────────────────────────────────────

    double eta = 0.0;
    double n0 = 0.0;
    double n1 = 0.0;
    double n2 = 0.0;

    arma::rowvec3 v0 = arma::zeros<arma::rowvec>(3);
    arma::rowvec3 v1 = arma::zeros<arma::rowvec>(3);

    arma::mat33 T = arma::zeros<arma::mat>(3, 3);

    // ── Inner products (set by compute_inner_products) ────────────────────

    InnerProducts products;

    /**
     * @brief Computes all inner products from the primary fields.
     *
     * Must be called after the primary fields have been populated.
     */
    void compute_inner_products() {
      products.dot_v0_v1 = arma::dot(v0, v1);
      products.dot_v1_v1 = arma::dot(v1, v1);
      arma::mat33 T2 = T * T;
      products.trace_T2 = arma::trace(T2);
      products.trace_T3 = arma::trace(T2 * T);
      products.quadratic_form = arma::as_scalar(v1 * T * v1.t());
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
      double r = 0.5 * d;

      m.eta = (std::numbers::pi / 6.0) * density * d * d * d;
      m.n2 = std::numbers::pi * density * d * d;
      m.n1 = r * density;
      m.n0 = density;

      m.v0.zeros();
      m.v1.zeros();

      double t_diag = m.n2 / 3.0;
      m.T.zeros();
      m.T(0, 0) = t_diag;
      m.T(1, 1) = t_diag;
      m.T(2, 2) = t_diag;

      m.compute_inner_products();
      return m;
    }
  };

}  // namespace dft::functional::fmt

#endif  // DFT_FUNCTIONAL_FMT_MEASURES_H
