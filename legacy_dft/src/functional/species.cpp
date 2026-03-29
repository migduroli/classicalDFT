#include "classicaldft_bits/functional/data_structures.h"

#include <stdexcept>

namespace dft::functional {

  Species::Species(density::Density density, double chemical_potential)
      : density_(std::move(density)),
        force_(arma::vec(density_.size(), arma::fill::zeros)),
        chemical_potential_(chemical_potential) {}

  // ── Fixed-mass constraint ───────────────────────────────────────────────────

  void Species::set_fixed_mass(double mass) {
    if (mass <= 0.0) {
      throw std::invalid_argument("Species::set_fixed_mass: mass must be positive");
    }
    fixed_mass_ = mass;
    chemical_potential_ = 0.0;
  }

  void Species::clear_fixed_mass() noexcept {
    fixed_mass_.reset();
  }

  void Species::begin_force_calculation() {
    if (fixed_mass_) {
      double current_n = density_.number_of_atoms();
      if (current_n > 0.0) {
        density_.scale(*fixed_mass_ / current_n);
      }
    }
  }

  void Species::end_force_calculation() {
    if (fixed_mass_) {
      double d_v = density_.cell_volume();
      double dot = arma::dot(force_, density_.values());
      chemical_potential_ = dot / *fixed_mass_;
      force_ -= chemical_potential_ * d_v;
    }
  }

  // ── Convergence ─────────────────────────────────────────────────────────────

  double Species::convergence_monitor() const {
    double d_v = density_.cell_volume();
    return arma::max(arma::abs(force_)) / d_v;  // NOLINT(clang-analyzer-core.CallAndMessage)
  }

  // ── Alias coordinates ───────────────────────────────────────────────────────

  void Species::set_density_from_alias(const arma::vec& x) {
    arma::vec rho = RHO_MIN + arma::square(x);
    density_.set(rho);
  }

  arma::vec Species::density_alias() const {
    return arma::sqrt(arma::clamp(density_.values() - RHO_MIN, 0.0, arma::datum::inf));
  }

  arma::vec Species::alias_force(const arma::vec& x) const {
    return 2.0 * x % force_;
  }

  // ── External field energy ───────────────────────────────────────────────────

  double Species::external_field_energy(bool accumulate_force) {
    double d_v = density_.cell_volume();
    const arma::vec& rho = density_.values();
    const arma::vec& vext = density_.external_field();

    double energy = arma::dot(rho, vext) * d_v - chemical_potential_ * density_.number_of_atoms();

    if (accumulate_force) {
      force_ += (vext - chemical_potential_) * d_v;
    }
    return energy;
  }

}  // namespace dft::functional
