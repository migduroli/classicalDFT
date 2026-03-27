#ifndef CLASSICALDFT_PHYSICS_SPECIES_SPECIES_H
#define CLASSICALDFT_PHYSICS_SPECIES_SPECIES_H

#include "classicaldft_bits/physics/density/density.h"

#include <armadillo>
#include <optional>

namespace dft_core::physics::species {

  /**
   * @brief A single-component species in a DFT calculation.
   *
   * Owns a `Density` (the density profile $\rho(\mathbf{r})$), a force vector
   * $\delta F / \delta \rho$ of the same size, a chemical potential $\mu$,
   * and an optional fixed-mass constraint.
   *
   * The alias coordinate system maps between an unconstrained variable $x$
   * and the physical density via $\rho = \rho_{\min} + x^2$, guaranteeing
   * positivity. This is virtual so that `FMTSpecies` (Phase 6) can override
   * with a bounded alias that also enforces $\eta < 1$.
   */
  class Species {
   public:
    /** @brief Minimum density floor used in alias coordinates. */
    static constexpr double rho_min = 1e-18;

    /**
     * @brief Constructs a species owning the given density profile.
     * @param density Density object (moved into the species).
     * @param chemical_potential Initial $\mu / k_BT$.
     */
    explicit Species(density::Density density, double chemical_potential = 0.0);

    virtual ~Species() = default;

    // ── Density access ──────────────────────────────────────────────────────

    [[nodiscard]] const density::Density& density() const noexcept { return density_; }
    [[nodiscard]] density::Density& density() noexcept { return density_; }

    // ── Force management ────────────────────────────────────────────────────

    [[nodiscard]] const arma::vec& force() const noexcept { return force_; }

    void zero_force() noexcept { force_.zeros(); }

    void add_to_force(const arma::vec& f) { force_ += f; }
    void add_to_force(arma::uword index, double value) { force_(index) += value; }

    // ── Chemical potential ──────────────────────────────────────────────────

    [[nodiscard]] double chemical_potential() const noexcept { return chemical_potential_; }
    void set_chemical_potential(double mu) noexcept { chemical_potential_ = mu; }

    // ── Fixed-mass constraint ───────────────────────────────────────────────

    [[nodiscard]] bool has_fixed_mass() const noexcept { return fixed_mass_.has_value(); }
    [[nodiscard]] std::optional<double> fixed_mass() const noexcept { return fixed_mass_; }

    /**
     * @brief Enables the fixed-mass constraint.
     * @param mass Target particle number $N$.
     */
    void set_fixed_mass(double mass);

    /**
     * @brief Disables the fixed-mass constraint.
     */
    void clear_fixed_mass() noexcept;

    /**
     * @brief Call before accumulating forces.
     *
     * If the fixed-mass constraint is active, rescales $\rho$ so that
     * $\sum \rho_i \, dV = m_{\text{fixed}}$.
     */
    void begin_force_calculation();

    /**
     * @brief Call after all force contributions have been accumulated.
     *
     * If the fixed-mass constraint is active, computes the Lagrange multiplier
     * $\mu = \frac{\sum_i dF_i \, \rho_i}{m_{\text{fixed}}}$
     * and projects the force: $dF_i \leftarrow dF_i - \mu \, dV$.
     */
    void end_force_calculation();

    // ── Convergence ─────────────────────────────────────────────────────────

    /**
     * @brief Convergence monitor: $\lVert dF \rVert_\infty / dV$.
     */
    [[nodiscard]] double convergence_monitor() const;

    // ── Alias coordinates ───────────────────────────────────────────────────

    /**
     * @brief Sets density from alias variable: $\rho_i = \rho_{\min} + x_i^2$.
     */
    virtual void set_density_from_alias(const arma::vec& x);

    /**
     * @brief Computes alias variable from current density: $x_i = \sqrt{\max(0, \rho_i - \rho_{\min})}$.
     */
    [[nodiscard]] virtual arma::vec density_alias() const;

    /**
     * @brief Chain-rule transform of force to alias coordinates: $dF/dx_i = 2 x_i \cdot dF/d\rho_i$.
     */
    [[nodiscard]] virtual arma::vec alias_force(const arma::vec& x) const;

    // ── External field energy ───────────────────────────────────────────────

    /**
     * @brief Computes $E = \sum_i (\rho_i V_{\text{ext},i} - \mu) \, dV$.
     *
     * If @p accumulate_force is true, adds $(V_{\text{ext},i} - \mu) \, dV$ to the force vector.
     */
    double external_field_energy(bool accumulate_force = false);

   private:
    density::Density density_;
    arma::vec force_;
    double chemical_potential_ = 0.0;
    std::optional<double> fixed_mass_;
  };

}  // namespace dft_core::physics::species

#endif  // CLASSICALDFT_PHYSICS_SPECIES_SPECIES_H
