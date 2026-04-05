#pragma once

#include "dft.hpp"

#include <armadillo>
#include <utility>
#include <vector>

namespace utils {

  // Build a State from a density profile at a given chemical potential.

  [[nodiscard]] inline auto make_state(const dft::physics::Model& model, const arma::vec& rho, double mu)
      -> dft::State {
    auto s = dft::init::from_profile(model, rho);
    s.species[0].chemical_potential = mu;
    return s;
  }

  // Force function wrapping the full DFT functional.

  [[nodiscard]] inline auto
  make_force_fn(const dft::physics::Model& model, const dft::functionals::Weights& weights, double mu) {
    return
        [&model, &weights, mu](const std::vector<arma::vec>& densities) -> std::pair<double, std::vector<arma::vec>> {
          auto state = make_state(model, densities[0], mu);
          auto result = dft::functionals::total(model, state, weights);
          return { result.grand_potential, result.forces };
        };
  }

}  // namespace utils
