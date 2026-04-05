#pragma once

#include "dft.hpp"

#include <armadillo>
#include <utility>
#include <vector>

namespace utils {

  // Extract the x-averaged 1D density profile from a 3D field.

  [[nodiscard]] inline auto extract_profile(const arma::vec& rho_3d, long nx, long ny, long nz) -> std::vector<double> {
    arma::mat rho_mat = arma::reshape(rho_3d, ny * nz, nx);
    arma::vec profile_avg = arma::mean(rho_mat, 0).as_col();
    return arma::conv_to<std::vector<double>>::from(profile_avg);
  }

  // Build a State from a density profile at a given chemical potential.

  [[nodiscard]] inline auto make_state(const dft::physics::Model& model, const arma::vec& rho, double mu)
      -> dft::State {
    auto s = dft::init::from_profile(model, rho);
    s.species[0].chemical_potential = mu;
    return s;
  }

  // Force function wrapping the full DFT functional for DDFT.

  [[nodiscard]] inline auto
  make_force_fn(const dft::physics::Model& model, const dft::functionals::Weights& weights, double mu) {
    return
        [&model, &weights, mu](const std::vector<arma::vec>& densities) -> std::pair<double, std::vector<arma::vec>> {
          auto state = make_state(model, densities[0], mu);
          auto result = dft::functionals::total(model, state, weights);
          return {result.grand_potential, result.forces};
        };
  }

} // namespace utils
