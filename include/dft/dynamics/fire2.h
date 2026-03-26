#ifndef DFT_DYNAMICS_FIRE2_H
#define DFT_DYNAMICS_FIRE2_H

#include "dft/dynamics/minimizer.h"

#include <armadillo>
#include <vector>

namespace dft::dynamics {

  /**
   * @brief Configuration for the FIRE2 minimizer.
   *
   * Supports designated initializers:
   *   `Fire2Config{.dt = 1e-3, .dt_max = 0.01}`
   */
  struct Fire2Config {
    double dt = 1e-3;            ///< Initial timestep
    double dt_max = 1e-2;        ///< Maximum timestep
    double dt_min = 1e-8;        ///< Minimum timestep
    double alpha_start = 0.1;    ///< Initial FIRE mixing parameter
    double f_inc = 1.1;          ///< Timestep increase factor when moving downhill
    double f_dec = 0.5;          ///< Timestep decrease factor when moving uphill
    double f_alf = 0.99;         ///< Alpha damping factor per delay cycle
    int n_delay = 5;             ///< Number of downhill steps before increasing dt
    int max_uphill_steps = 20;   ///< Max consecutive uphill steps before exception
    double force_limit = 0.1;    ///< Force convergence threshold
    double min_density = 1e-30;  ///< Minimum average density before stopping
  };

  /**
   * @brief FIRE2 minimizer: Fast Inertial Relaxation Engine, version 2.
   *
   * Velocity-Verlet with adaptive timestep and velocity damping that steers
   * the trajectory toward the nearest energy minimum. The algorithm:
   *
   *  1. Compute power $P = \mathbf{v} \cdot (-\nabla F)$
   *  2. If $P > 0$ for `n_delay` consecutive steps, increase $\Delta t$ and
   *     decrease $\alpha$
   *  3. If $P < 0$, halve the velocity, cut $\Delta t$, reset $\alpha$
   *  4. Semi-implicit Euler: update velocities then mix with force direction
   *  5. Backtrack if packing fraction exceeds physical limits
   *
   * Reference: Bitzek et al., PRL 97, 170201 (2006), with improvements from
   * Guénolé et al., Comp. Mat. Sci. 175, 109584 (2020).
   */
  class Fire2Minimizer final : public Minimizer {
   public:
    explicit Fire2Minimizer(Solver& solver, Fire2Config config = {});

    // ── Inspectors ──────────────────────────────────────────────────────

    [[nodiscard]] double dt() const noexcept { return dt_; }
    [[nodiscard]] double alpha() const noexcept { return alpha_; }
    [[nodiscard]] double rms_force() const noexcept { return rms_force_; }
    [[nodiscard]] const Fire2Config& config() const noexcept { return config_; }

   private:
    [[nodiscard]] double do_step() override;
    void do_reset() override;

    /**
     * @brief Semi-implicit Euler integration step.
     *
     * Updates velocities with forces, applies FIRE mixing, and advances
     * positions. Recomputes forces at the new position.
     *
     * @param begin First species index to update.
     * @param end One past the last species index to update.
     * @return Free energy at the new position.
     */
    [[nodiscard]] double semi_implicit_euler(int begin, int end);

    Fire2Config config_;
    std::vector<arma::vec> velocities_;
    double dt_;
    double alpha_;
    int n_positive_ = 0;
    int n_negative_ = 0;
    int backtracks_ = 0;
    double dt_best_ = 0.0;
    double rms_force_ = 0.0;
    double f_max_ = 0.0;
    long iteration_ = 0;
  };

}  // namespace dft::dynamics

#endif  // DFT_DYNAMICS_FIRE2_H
