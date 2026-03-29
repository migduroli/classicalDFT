#ifndef CLASSICALDFT_FUNCTIONAL_MEAN_FIELD_INTERACTION_H
#define CLASSICALDFT_FUNCTIONAL_MEAN_FIELD_INTERACTION_H

#include "classicaldft_bits/functional/fmt/weighted_density.h"
#include "classicaldft_bits/potential/potential.h"
#include "classicaldft_bits/functional/data_structures.h"

#include <armadillo>

namespace dft::functional::mean_field {

  /**
   * @brief Integration scheme for computing interaction weights on the lattice.
   *
   * Controls how the attractive potential is sampled when building the
   * discrete weight array $w(\mathbf{R})$ from the continuum potential.
   */
  enum class WeightScheme {
    InterpolationZero,     ///< Value at lattice point
    InterpolationLinearE,  ///< Linear interpolation, energy route
    InterpolationLinearF,  ///< Linear interpolation, force route
    GaussE,                ///< Gauss-Legendre, energy route
    GaussF                 ///< Gauss-Legendre, force route
  };

  /**
   * @brief Mean-field interaction between two species via FFT convolution.
   *
   * Computes the mean-field contribution to the free energy:
   * $$F_{\text{mf}} = \frac{1}{2} \sum_{\mathbf{R}, \mathbf{R}'}
   *   \rho_1(\mathbf{R})\, w(\mathbf{R} - \mathbf{R}')\, \rho_2(\mathbf{R}') \, dV$$
   *
   * where $w(\mathbf{R})$ is a discrete weight array computed from the
   * attractive part of the interatomic potential, divided by $k_B T$.
   *
   * Energy and forces are computed via FFT convolution in $O(N \log N)$.
   */
  class Interaction {
   public:
    // ── Construction ──────────────────────────────────────────────────────

    /**
     * @brief Constructs a mean-field interaction.
     *
     * @param s1 First species (non-owning reference).
     * @param s2 Second species (non-owning reference).
     * @param potential The pair potential whose attractive tail provides $w$.
     * @param kT Temperature in energy units ($k_B T$).
     * @param scheme Integration scheme for weight generation.
     * @param gauss_order Number of Gauss-Legendre points (only for GaussE/GaussF).
     *
     * @throws std::invalid_argument if kT <= 0, or species grids are incompatible.
     */
    Interaction(
        Species& s1,
        Species& s2,
        const potentials::Potential& potential,
        double kT,
        WeightScheme scheme = WeightScheme::InterpolationLinearF,
        int gauss_order = 5
    );

    Interaction(const Interaction&) = delete;
    Interaction& operator=(const Interaction&) = delete;
    Interaction(Interaction&&) noexcept = default;  // NOLINT(bugprone-exception-escape)
    Interaction& operator=(Interaction&&) = delete;

    // ── Energy and forces ─────────────────────────────────────────────────

    /**
     * @brief Computes the mean-field free energy.
     *
     * $F = \frac{1}{2} \sum_i \rho_1(i) \, (\hat w * \rho_2)(i) \, dV$
     */
    [[nodiscard]] double compute_free_energy();

    /**
     * @brief Computes forces and accumulates them into the species' force vectors.
     *
     * Force on species 1: $f_1(i) = -(\hat w * \rho_2)(i) \, dV / k_B T$
     * Force on species 2: $f_2(j) = -(\hat w * \rho_1)(j) \, dV / k_B T$
     *
     * @return The mean-field free energy (computed as a byproduct).
     */
    double compute_forces();

    // ── Bulk thermodynamics ───────────────────────────────────────────────

    /**
     * @brief The van der Waals parameter $a = \sum_R w(R) \, dV$.
     */
    [[nodiscard]] double vdw_parameter() const noexcept;

    /**
     * @brief Bulk mean-field free energy density for uniform densities.
     *
     * $f = \frac{1}{2}\, a_{\text{vdw}}\, \rho_1\, \rho_2$
     */
    [[nodiscard]] double bulk_free_energy_density(double rho1, double rho2) const;

    /**
     * @brief Bulk chemical potential contribution from species 2 onto species 1.
     *
     * $\mu_1^{\text{mf}} = a_{\text{vdw}} \cdot \rho_2$
     */
    [[nodiscard]] double bulk_chemical_potential(double rho_other) const;

    // ── Inspectors ────────────────────────────────────────────────────────

    [[nodiscard]] const Species& species_1() const noexcept;
    [[nodiscard]] const Species& species_2() const noexcept;
    [[nodiscard]] double temperature() const noexcept;
    [[nodiscard]] WeightScheme scheme() const noexcept;

   private:
    void generate_weights();
    [[nodiscard]] double compute_cell_weight_at_point(double r2) const;
    [[nodiscard]] double compute_cell_weight_interpolation_zero(long sx, long sy, long sz) const;
    [[nodiscard]] double compute_cell_weight_interpolation_linear(long sx, long sy, long sz) const;
    [[nodiscard]] double compute_cell_weight_gauss(long sx, long sy, long sz) const;

    Species& s1_;
    Species& s2_;
    const potentials::Potential& potential_;
    double kT_;
    WeightScheme scheme_;
    int gauss_order_;
    double a_vdw_ = 0.0;
    double dx_;
    double dv_;
    std::vector<long> shape_;
    fmt::WeightedDensity convolution_;
  };

}  // namespace dft::functional::mean_field

#endif  // CLASSICALDFT_FUNCTIONAL_MEAN_FIELD_INTERACTION_H
