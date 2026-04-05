#ifndef DFT_FUNCTIONALS_BULK_THERMODYNAMICS_HPP
#define DFT_FUNCTIONALS_BULK_THERMODYNAMICS_HPP

#include "dft/functionals/fmt/models.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/physics/model.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <cmath>
#include <vector>

namespace dft::functionals::bulk {

  // Each contribution lives in its own namespace.
  // Totals (aggregating all contributions) stay at bulk:: level.

  namespace ideal {

    [[nodiscard]] inline auto free_energy_density(const arma::vec& rho) -> double {
      double f = 0.0;
      for (arma::uword i = 0; i < rho.n_elem; ++i) {
        if (rho(i) > 0.0) {
          f += rho(i) * (std::log(rho(i)) - 1.0);
        }
      }
      return f;
    }

    [[nodiscard]] inline auto chemical_potential(double rho) -> double {
      return std::log(rho);
    }

  }  // namespace ideal

  namespace hard_sphere {

    [[nodiscard]] inline auto free_energy_density(
        const fmt::FMTModel& model, const arma::vec& rho,
        const std::vector<Species>& species
    ) -> double {
      fmt::Measures total;
      for (arma::uword i = 0; i < rho.n_elem; ++i) {
        auto m = fmt::make_uniform_measures(rho(i), species[i].hard_sphere_diameter);
        total.eta += m.eta;
        total.n0 += m.n0;
        total.n1 += m.n1;
        total.n2 += m.n2;
        total.T += m.T;
      }
      total.products = fmt::inner_products(total);
      return fmt::phi(model, total);
    }

    [[nodiscard]] inline auto excess_chemical_potential(
        const fmt::FMTModel& model, const arma::vec& rho,
        const std::vector<Species>& species, arma::uword species_idx
    ) -> double {
      fmt::Measures total;
      for (arma::uword i = 0; i < rho.n_elem; ++i) {
        auto m = fmt::make_uniform_measures(rho(i), species[i].hard_sphere_diameter);
        total.eta += m.eta;
        total.n0 += m.n0;
        total.n1 += m.n1;
        total.n2 += m.n2;
        total.T += m.T;
      }
      total.products = fmt::inner_products(total);
      auto dm = fmt::d_phi(model, total);

      double d = species[species_idx].hard_sphere_diameter;
      double r = 0.5 * d;

      double dn3 = (std::numbers::pi / 6.0) * d * d * d;
      double dn2 = std::numbers::pi * d * d;
      double dn1 = r;
      double dn0 = 1.0;

      double mu = dm.d_eta * dn3 + dm.d_n2 * dn2 + dm.d_n1 * dn1 + dm.d_n0 * dn0;

      if (fmt::needs_tensor(model)) {
        double dt = std::numbers::pi * d * d / 3.0;
        for (int j = 0; j < 3; ++j) {
          mu += dm.d_T(j, j) * dt;
        }
      }

      return mu;
    }

  }  // namespace hard_sphere

  namespace mean_field {

    [[nodiscard]] inline auto free_energy_density(
        const MeanFieldWeights& weights, const arma::vec& rho
    ) -> double {
      double f = 0.0;
      for (const auto& iw : weights.interactions) {
        auto i = static_cast<arma::uword>(iw.species_i);
        auto j = static_cast<arma::uword>(iw.species_j);
        if (i == j) {
          f += 0.5 * iw.a_vdw * rho(i) * rho(j);
        } else {
          f += iw.a_vdw * rho(i) * rho(j);
        }
      }
      return f;
    }

    [[nodiscard]] inline auto chemical_potential(
        const MeanFieldWeights& weights, const arma::vec& rho, arma::uword species_idx
    ) -> double {
      double mu = 0.0;
      for (const auto& iw : weights.interactions) {
        auto i = static_cast<arma::uword>(iw.species_i);
        auto j = static_cast<arma::uword>(iw.species_j);
        if (i == species_idx && j == species_idx) {
          mu += iw.a_vdw * rho(j);
        } else if (i == species_idx) {
          mu += 0.5 * iw.a_vdw * rho(j);
        } else if (j == species_idx) {
          mu += 0.5 * iw.a_vdw * rho(i);
        }
      }
      return mu;
    }

  }  // namespace mean_field

  // Total Helmholtz free energy density (per unit volume, in kT units).

  [[nodiscard]] inline auto free_energy_density(
      const arma::vec& rho, const std::vector<Species>& species,
      const Weights& weights
  ) -> double {
    double f = ideal::free_energy_density(rho);
    f += hard_sphere::free_energy_density(weights.fmt_model, rho, species);
    f += mean_field::free_energy_density(weights.mean_field, rho);
    return f;
  }

  // Total chemical potential for species s (in kT units).

  [[nodiscard]] inline auto chemical_potential(
      const arma::vec& rho, const std::vector<Species>& species,
      const Weights& weights, arma::uword species_idx
  ) -> double {
    double mu = ideal::chemical_potential(rho(species_idx));
    mu += hard_sphere::excess_chemical_potential(weights.fmt_model, rho, species, species_idx);
    mu += mean_field::chemical_potential(weights.mean_field, rho, species_idx);
    return mu;
  }

  // Chemical potentials for all species.

  [[nodiscard]] inline auto chemical_potentials(
      const arma::vec& rho, const std::vector<Species>& species,
      const Weights& weights
  ) -> arma::vec {
    arma::vec mu(rho.n_elem);
    for (arma::uword s = 0; s < rho.n_elem; ++s) {
      mu(s) = chemical_potential(rho, species, weights, s);
    }
    return mu;
  }

  // Pressure via P = sum_i rho_i mu_i - f (in kT units).

  [[nodiscard]] inline auto pressure(
      const arma::vec& rho, const std::vector<Species>& species,
      const Weights& weights
  ) -> double {
    double f = free_energy_density(rho, species, weights);
    double p = -f;
    for (arma::uword s = 0; s < rho.n_elem; ++s) {
      p += rho(s) * chemical_potential(rho, species, weights, s);
    }
    return p;
  }

  // Grand potential density: omega = -P.

  [[nodiscard]] inline auto grand_potential_density(
      const arma::vec& rho, const std::vector<Species>& species,
      const Weights& weights
  ) -> double {
    return -pressure(rho, species, weights);
  }

  // Backward compatibility aliases for the old prefix-named functions.

  [[nodiscard]] inline auto ideal_free_energy_density(const arma::vec& rho) -> double {
    return ideal::free_energy_density(rho);
  }
  [[nodiscard]] inline auto ideal_chemical_potential(double rho) -> double {
    return ideal::chemical_potential(rho);
  }
  [[nodiscard]] inline auto hard_sphere_free_energy_density(
      const fmt::FMTModel& model, const arma::vec& rho, const std::vector<Species>& species
  ) -> double {
    return hard_sphere::free_energy_density(model, rho, species);
  }
  [[nodiscard]] inline auto hard_sphere_excess_chemical_potential(
      const fmt::FMTModel& model, const arma::vec& rho,
      const std::vector<Species>& species, arma::uword species_idx
  ) -> double {
    return hard_sphere::excess_chemical_potential(model, rho, species, species_idx);
  }
  [[nodiscard]] inline auto mean_field_free_energy_density(
      const MeanFieldWeights& weights, const arma::vec& rho
  ) -> double {
    return mean_field::free_energy_density(weights, rho);
  }
  [[nodiscard]] inline auto mean_field_chemical_potential(
      const MeanFieldWeights& weights, const arma::vec& rho, arma::uword species_idx
  ) -> double {
    return mean_field::chemical_potential(weights, rho, species_idx);
  }

}  // namespace dft::functionals::bulk

#endif  // DFT_FUNCTIONALS_BULK_THERMODYNAMICS_HPP
