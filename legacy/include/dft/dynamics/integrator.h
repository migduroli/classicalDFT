#ifndef DFT_DYNAMICS_INTEGRATOR_H
#define DFT_DYNAMICS_INTEGRATOR_H

#include "dft/dynamics/minimizer.h"
#include "dft/math/fourier.h"

#include <armadillo>
#include <vector>

namespace dft::dynamics {

  /**
   * @brief Integration scheme for density dynamics.
   */
  enum class IntegrationScheme {
    SplitOperator,   ///< Exact diffusion + explicit Euler for excess
    CrankNicholson,  ///< Exact diffusion + Crank-Nicholson implicit solve for excess
  };

  /**
   * @brief Configuration for the density dynamics integrator.
   *
   * Supports designated initializers:
   *   `IntegratorConfig{.scheme = IntegrationScheme::CrankNicholson, .dt = 1e-4}`
   */
  struct IntegratorConfig {
    IntegrationScheme scheme = IntegrationScheme::SplitOperator;
    double dt = 1e-4;                    ///< Fixed timestep
    double diffusion_coefficient = 1.0;  ///< Diffusion coefficient $D$
    double force_limit = 0.1;            ///< Force convergence threshold
    double min_density = 1e-30;          ///< Minimum average density
    int crank_nicholson_iterations = 5;  ///< Fixed-point iterations (CrankNicholson only)
    double cn_tolerance = 1e-10;         ///< Convergence tolerance (CrankNicholson only)
  };

  /**
   * @brief Density dynamics integrator for the DDFT equation.
   *
   * Integrates the dynamical density functional theory equation in time:
   *
   *   $\frac{\partial \rho}{\partial t} = D \nabla \cdot \left[ \rho \nabla
   *   \frac{\delta \beta F}{\delta \rho} \right]$
   *
   * The ideal part ($D \nabla^2 \rho$, free diffusion) is always treated
   * exactly via a Fourier-space propagator $e^{-D k^2 \Delta t}$.
   *
   * Two schemes are available for the excess (nonlinear) part:
   *
   *   - **SplitOperator**: explicit forward Euler. Stable for small $\Delta t$.
   *
   *   - **CrankNicholson**: implicit Crank-Nicholson iteration with the
   *     integrating-factor approach. Second-order accurate; unconditionally
   *     stable for the linear part.
   */
  class Integrator final : public Minimizer {
   public:
    explicit Integrator(Solver& solver, IntegratorConfig config = {});

    // ── Inspectors ──────────────────────────────────────────────────────

    [[nodiscard]] double dt() const noexcept { return config_.dt; }
    [[nodiscard]] double diffusion_coefficient() const noexcept { return config_.diffusion_coefficient; }
    [[nodiscard]] IntegrationScheme scheme() const noexcept { return config_.scheme; }
    [[nodiscard]] const IntegratorConfig& config() const noexcept { return config_; }

   private:
    [[nodiscard]] double do_step() override;
    void do_reset() override;

    // ── Split-operator step ─────────────────────────────────────────────

    [[nodiscard]] double do_split_operator_step();

    // ── Crank-Nicholson step ────────────────────────────────────────────

    [[nodiscard]] double do_crank_nicholson_step();

    /**
     * @brief Compute the nonlinear excess flux term $N[\rho]$ in Fourier space.
     *
     * Returns $-D k^2 \hat{(\rho \cdot c_{\text{ex}}^{(1)})}$ where
     * $c_{\text{ex}}^{(1)}$ is the excess one-body direct correlation function.
     *
     * @param species_index Species to compute for.
     * @return Complex Fourier coefficients of the excess flux divergence.
     */
    [[nodiscard]] arma::cx_vec compute_nonlinear_term(int species_index);

    // ── Shared Fourier-space helpers ────────────────────────────────────

    /**
     * @brief Compute $k^2$ for each Fourier mode.
     *
     * For a periodic box of length $L$ with $N$ grid points, the wavevectors
     * are $k_j = 2\pi j / L$ for $j = 0, \ldots, N/2$.
     */
    [[nodiscard]] arma::vec compute_k_squared() const;

    /**
     * @brief Apply the free diffusion propagator in Fourier space.
     *
     * Multiplies each Fourier mode by $e^{-D k^2 \Delta t}$.
     */
    void apply_diffusion_propagator(int species_index, double dt_propagate);

    IntegratorConfig config_;
    arma::vec k_squared_;                       ///< Precomputed $k^2$
    math::fourier::FourierTransform work_fft_;  ///< Working FFT buffer
    math::fourier::FourierTransform cn_fft_;    ///< Working FFT for Crank-Nicholson
  };

}  // namespace dft::dynamics

#endif  // DFT_DYNAMICS_INTEGRATOR_H
