#ifndef DFT_SPECIES_H
#define DFT_SPECIES_H

#include "dft/density.h"

#include <armadillo>
#include <optional>

namespace dft::species {

  /**
   * @brief A single-component species in a DFT calculation.
   *
   * Owns a `Density` (the density profile $\rho(\mathbf{r})$), a force vector
   * $\delta F / \delta \rho$ of the same size, a chemical potential $\mu$,
   * and an optional fixed-mass constraint.
   *
   * The alias coordinate system maps between an unconstrained variable $x$
   * and the physical density via $\rho = \rho_{\min} + x^2$, guaranteeing
   * positivity. Subclasses may override the alias to impose additional bounds
   * (e.g. $\eta < 1$ for hard spheres).
   */
  class Species {
   public:
    static constexpr double RHO_MIN = 1e-18;

    explicit Species(density::Density density, double chemical_potential = 0.0);

    virtual ~Species() = default;

    // ── Density access ──────────────────────────────────────────────────────

    [[nodiscard]] const density::Density& density() const noexcept { return density_; }
    [[nodiscard]] density::Density& density() noexcept { return density_; }

    // ── Force management ────────────────────────────────────────────────────

    [[nodiscard]] const arma::vec& force() const noexcept { return force_; }
    [[nodiscard]] arma::vec& force() noexcept { return force_; }

    void zero_force() noexcept { force_.zeros(); }

    void add_to_force(const arma::vec& f) { force_ += f; }
    void add_to_force(arma::uword index, double value) { force_(index) += value; }

    // ── Chemical potential ──────────────────────────────────────────────────

    [[nodiscard]] double chemical_potential() const noexcept { return chemical_potential_; }
    void set_chemical_potential(double mu) noexcept { chemical_potential_ = mu; }

    // ── Fixed-mass constraint ───────────────────────────────────────────────

    [[nodiscard]] bool has_fixed_mass() const noexcept { return fixed_mass_.has_value(); }
    [[nodiscard]] std::optional<double> fixed_mass() const noexcept { return fixed_mass_; }

    void set_fixed_mass(double mass);
    void clear_fixed_mass() noexcept;

    void begin_force_calculation();
    void end_force_calculation();

    // ── Convergence ─────────────────────────────────────────────────────────

    [[nodiscard]] double convergence_monitor() const;

    // ── Alias coordinates ───────────────────────────────────────────────────

    virtual void set_density_from_alias(const arma::vec& x);
    [[nodiscard]] virtual arma::vec density_alias() const;
    [[nodiscard]] virtual arma::vec alias_force(const arma::vec& x) const;

    // ── External field energy ───────────────────────────────────────────────

    double external_field_energy(bool accumulate_force = false);

   private:
    density::Density density_;
    arma::vec force_;
    double chemical_potential_ = 0.0;
    std::optional<double> fixed_mass_;
  };

}  // namespace dft::species

#endif  // DFT_SPECIES_H
