#ifndef DFT_DYNAMICS_MINIMIZER_H
#define DFT_DYNAMICS_MINIMIZER_H

#include "dft/solver.h"

#include <armadillo>
#include <functional>
#include <optional>
#include <vector>

namespace dft::dynamics {

  /**
   * @brief Abstract base class for DFT minimizers and time-steppers.
   *
   * Manages the iteration loop, alias coordinate system, convergence
   * monitoring, and optional fixed-direction projection (for saddle-point
   * searches). Concrete subclasses implement `do_step()` with the specific
   * algorithm (FIRE2, DDFT, etc.).
   *
   * Uses the Non-Virtual Interface (NVI) pattern: public `run()`/`resume()`
   * call the protected virtual hook `do_step()`.
   */
  class Minimizer {
   public:
    /**
     * @brief Callback invoked after each step.
     *
     * Arguments: step number, energy, max force.
     * Return true to continue, false to stop.
     */
    using StepCallback = std::function<bool(long, double, double)>;

    explicit Minimizer(Solver& solver, double force_limit = 0.1, double min_density = 1e-30);

    virtual ~Minimizer() = default;

    Minimizer(const Minimizer&) = delete;
    Minimizer& operator=(const Minimizer&) = delete;
    Minimizer(Minimizer&&) = delete;
    Minimizer& operator=(Minimizer&&) = delete;

    // ── Public interface (NVI) ──────────────────────────────────────────

    /**
     * @brief Run the minimizer from the current density state.
     *
     * Resets step counter and aliases, then iterates.
     *
     * @param max_steps Maximum iterations (0 = unlimited).
     * @return true if converged (max force < force_limit).
     */
    [[nodiscard]] bool run(long max_steps = 0);

    /**
     * @brief Continue iterating from the current minimizer state.
     *
     * Does not reset step counter or velocities.
     *
     * @param max_steps Maximum additional iterations (0 = unlimited).
     * @return true if converged.
     */
    [[nodiscard]] bool resume(long max_steps = 0);

    /**
     * @brief Reset aliases from current density profiles.
     */
    void reset();

    // ── Inspectors ──────────────────────────────────────────────────────

    [[nodiscard]] double energy() const noexcept { return energy_; }
    [[nodiscard]] double max_force() const noexcept { return max_force_; }
    [[nodiscard]] long step_count() const noexcept { return step_count_; }
    [[nodiscard]] double force_limit() const noexcept { return force_limit_; }
    [[nodiscard]] const Solver& solver() const noexcept { return solver_; }
    [[nodiscard]] Solver& solver() noexcept { return solver_; }

    // ── Configuration ───────────────────────────────────────────────────

    void set_force_limit(double limit) noexcept { force_limit_ = limit; }
    void set_min_density(double rho_min) noexcept { min_density_ = rho_min; }
    void set_step_callback(StepCallback cb) { step_callback_ = std::move(cb); }

    // ── Fixed direction (saddle-point search) ───────────────────────────

    void set_fixed_direction(const arma::vec& direction);
    void clear_fixed_direction() noexcept { fixed_direction_.reset(); }
    [[nodiscard]] bool has_fixed_direction() const noexcept { return fixed_direction_.has_value(); }

   protected:
    // ── NVI hooks ───────────────────────────────────────────────────────

    /**
     * @brief Perform one algorithm step.
     * @return The total free energy after the step.
     */
    [[nodiscard]] virtual double do_step() = 0;

    /**
     * @brief Reset algorithm-specific state (velocities, timers, etc.).
     */
    virtual void do_reset() {}

    // ── Helpers for subclasses ──────────────────────────────────────────

    /**
     * @brief Set densities from aliases, compute free energy and forces,
     *        project forces if a fixed direction is set, convert to alias
     *        derivatives.
     * @return Total free energy.
     */
    [[nodiscard]] double compute_energy_and_forces();

    /**
     * @brief Compute energy and forces in density space (no alias conversion).
     *
     * Sets densities from current aliases, computes forces, and applies
     * fixed-direction projection if set. Forces remain as dΩ/dρ * dV.
     * Used by DDFT methods that operate directly on density.
     *
     * @return Total free energy.
     */
    [[nodiscard]] double compute_density_forces();

    [[nodiscard]] std::vector<arma::vec>& aliases() noexcept { return aliases_; }
    [[nodiscard]] const std::vector<arma::vec>& aliases() const noexcept { return aliases_; }

    [[nodiscard]] Solver& mutable_solver() noexcept { return solver_; }
    [[nodiscard]] const Solver& mutable_solver() const noexcept { return solver_; }

   private:
    void initialize_aliases();
    [[nodiscard]] double convergence_monitor() const;

    Solver& solver_;
    double force_limit_;
    double min_density_;
    std::vector<arma::vec> aliases_;
    std::optional<arma::vec> fixed_direction_;
    StepCallback step_callback_;
    double energy_ = 0.0;
    double max_force_ = 0.0;
    long step_count_ = 0;
  };

}  // namespace dft::dynamics

#endif  // DFT_DYNAMICS_MINIMIZER_H
