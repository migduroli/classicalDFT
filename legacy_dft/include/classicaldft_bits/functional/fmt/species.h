#ifndef CLASSICALDFT_FUNCTIONAL_FMT_SPECIES_H
#define CLASSICALDFT_FUNCTIONAL_FMT_SPECIES_H

#include "classicaldft_bits/functional/fmt/functional.h"
#include "classicaldft_bits/functional/fmt/weights.h"
#include "classicaldft_bits/functional/data_structures.h"

namespace dft::functional::fmt {

  /**
   * @brief A hard-sphere species with FMT convolution machinery.
   *
   * Inherits `Species` and overrides the alias system to enforce
   * $\eta < 1$ (bounded alias). Owns a `WeightedDensitySet` whose channels compute
   * the fundamental measures via FFT convolution.
   *
   * The pipeline is split into two single-responsibility steps:
   *  - `compute_free_energy(functional)` runs the full forward/backward pipeline
   *    and returns the excess free energy $F_{\text{ex}}$.
   *  - `compute_forces(functional)` runs the pipeline and accumulates forces
   *    into the species' force vector.
   *
   * Each step can be called independently.
   */
  class FMTSpecies : public Species {
   public:
    /**
     * @brief Constructs an FMT species.
     *
     * Generates the weight functions for a hard sphere of diameter $d$.
     *
     * @param density The density object (moved in).
     * @param diameter Hard-sphere diameter $d$.
     * @param chemical_potential Initial $\mu / k_B T$.
     */
    FMTSpecies(density::Density density, double diameter, double chemical_potential = 0.0);

    [[nodiscard]] double diameter() const noexcept { return diameter_; }

    // ── FMT pipeline (single responsibility) ────────────────────────────────

    /**
     * @brief Computes the excess free energy $F_{\text{ex}} = \sum_i \Phi_i \, dV$.
     *
     * Runs the forward convolution, evaluates $\Phi$ at each lattice point,
     * and returns the total excess free energy. Does NOT touch forces.
     */
    [[nodiscard]] double compute_free_energy(const FundamentalMeasureTheory& functional);

    /**
     * @brief Computes forces and accumulates them into the force vector.
     *
     * Runs the full pipeline (forward convolution, local $\Phi$ and $d\Phi$,
     * back-convolution) and adds the result to the force vector.
     *
     * @return The excess free energy (computed as a byproduct).
     */
    double compute_forces(const FundamentalMeasureTheory& functional);

    // ── Weighted density access ─────────────────────────────────────────────

    /**
     * @brief Forward-convolve the density with all weight functions.
     *
     * @param tensor If false, skip the 6 tensor components.
     */
    void convolve_density(bool tensor);

    /**
     * @brief Extract the 19 fundamental measures at lattice point @p pos.
     */
    [[nodiscard]] FundamentalMeasures measures_at(arma::uword pos) const;

    /**
     * @brief Collapse 19 derivatives back to the 11 stored derivative arrays.
     */
    void set_derivatives(const FundamentalMeasures& dm, arma::uword pos, bool tensor);

    /**
     * @brief Back-convolve the derivative fields and accumulate into forces.
     */
    void accumulate_forces(bool tensor);

    // ── Alias override (bounded: enforces eta < 1) ──────────────────────────

    void set_density_from_alias(const arma::vec& x) override;
    [[nodiscard]] arma::vec density_alias() const override;
    [[nodiscard]] arma::vec alias_force(const arma::vec& x) const override;

   private:
    double diameter_;
    WeightedDensitySet weights_;
    double density_range_;  // rho_max - rho_min (bounded alias parameter)
  };

}  // namespace dft::functional::fmt

#endif  // CLASSICALDFT_FUNCTIONAL_FMT_SPECIES_H
