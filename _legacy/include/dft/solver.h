#ifndef DFT_SOLVER_H
#define DFT_SOLVER_H

#include "dft/density.h"
#include "dft/functional/fmt/functional.h"
#include "dft/functional/fmt/species.h"
#include "dft/functional/interaction.h"
#include "dft/math/hessian.h"
#include "dft/species.h"

#include <memory>
#include <string>
#include <vector>

namespace dft {

  /**
   * @brief Core DFT orchestrator
   *
   * Owns the species, interactions, and hard-sphere functional that together
   * define the grand-potential functional
   *
   *   $\Omega[\rho] = F_{\text{id}} + F_{\text{hs}} + F_{\text{mf}} + F_{\text{ext}}$
   *
   * and its functional derivatives.
   */
  class Solver : public math::HessianOperator {
   public:
    Solver() = default;
    ~Solver() override = default;

    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) noexcept = default;
    Solver& operator=(Solver&&) noexcept = default;

    // ── Components ──────────────────────────────────────────────────────

    void add_species(std::unique_ptr<species::Species> s);
    void add_interaction(std::unique_ptr<functional::interaction::Interaction> i);
    void set_fmt(std::unique_ptr<functional::fmt::FMT> fmt);

    // ── Inspectors ──────────────────────────────────────────────────────

    [[nodiscard]] int num_species() const noexcept;
    [[nodiscard]] const species::Species& species(int i) const;
    [[nodiscard]] species::Species& species(int i);
    [[nodiscard]] const density::Density& density(int i) const;
    [[nodiscard]] double convergence_monitor() const;
    [[nodiscard]] std::string fmt_name() const;

    // ── Core computation ────────────────────────────────────────────────

    /**
     * @brief Compute total free energy and accumulate functional derivatives
     * @param excess_only if true, skip ideal gas and external contributions
     * @return total free energy $\Omega$
     */
    [[nodiscard]] double compute_free_energy_and_forces(bool excess_only = false);

    // ── Free energy decomposition ───────────────────────────────────────

    [[nodiscard]] double ideal_free_energy() const noexcept;
    [[nodiscard]] double hard_sphere_free_energy() const noexcept;
    [[nodiscard]] double mean_field_free_energy() const noexcept;
    [[nodiscard]] double external_free_energy() const noexcept;

    // ── Bulk thermodynamics ─────────────────────────────────────────────

    [[nodiscard]] double chemical_potential(const std::vector<double>& densities, int species_index) const;
    [[nodiscard]] double grand_potential_density(const std::vector<double>& densities) const;
    [[nodiscard]] double helmholtz_free_energy_density(const std::vector<double>& densities) const;

    [[nodiscard]] double chemical_potential(double density) const;
    [[nodiscard]] double grand_potential_density(double density) const;
    [[nodiscard]] double helmholtz_free_energy_density(double density) const;

    // ── Coexistence utilities ───────────────────────────────────────────

    void find_spinodal(double max_density, double step, double& rho_s1, double& rho_s2, double tol) const;
    [[nodiscard]] double find_density_from_chemical_potential(double mu, double rho_min, double rho_max, double tol)
        const;
    void find_coexistence(double max_density, double step, double& rho_v, double& rho_l, double tol) const;

    // ── Structural properties ───────────────────────────────────────────

    [[nodiscard]] double real_space_dcf(double r, double density) const;
    [[nodiscard]] double fourier_space_dcf(double k, double density) const;

    // ── Alias support ───────────────────────────────────────────────────

    void set_densities_from_aliases(std::vector<arma::vec>& aliases);
    void convert_forces_to_alias_derivatives(std::vector<arma::vec>& aliases);

    // ── HessianOperator interface ───────────────────────────────────────

    [[nodiscard]] arma::uword dimension() const noexcept override;
    void hessian_dot_v(const arma::vec& v, arma::vec& result) const override;

   private:
    std::vector<std::unique_ptr<species::Species>> species_;
    std::vector<std::unique_ptr<functional::interaction::Interaction>> interactions_;
    std::unique_ptr<functional::fmt::FMT> fmt_;

    double f_ideal_ = 0.0;
    double f_hard_sphere_ = 0.0;
    double f_mean_field_ = 0.0;
    double f_external_ = 0.0;
  };

}  // namespace dft

#endif  // DFT_SOLVER_H
