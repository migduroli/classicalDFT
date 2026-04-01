#ifndef DFT_FUNCTIONALS_IDEAL_GAS_HPP
#define DFT_FUNCTIONALS_IDEAL_GAS_HPP

#include "dft/grid.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <cmath>
#include <vector>

namespace dft::functionals {

  // A single functional contribution: free energy plus per-species forces.
  struct Contribution {
    double free_energy{0.0};
    std::vector<arma::vec> forces;
  };

  // Ideal gas free energy and forces for all species in the state.
  // F_id = kT sum_i int rho_i(r) [ln(rho_i(r)) - 1] dV
  // force_i(r) = kT [ln(rho_i(r)) - mu_i / kT] dV

  [[nodiscard]] inline auto ideal_gas(const Grid& grid, const State& state) -> Contribution {
    double dv = grid.cell_volume();
    double kT = state.temperature;

    Contribution result;
    result.forces.reserve(state.species.size());

    for (const auto& sp : state.species) {
      const arma::vec& rho = sp.density.values;
      arma::vec log_rho = arma::log(arma::clamp(rho, 1e-300, arma::datum::inf));

      result.free_energy += kT * arma::dot(rho, log_rho - 1.0) * dv;

      arma::vec f = (log_rho - sp.chemical_potential / kT) * dv;
      result.forces.push_back(std::move(f));
    }

    return result;
  }

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_IDEAL_GAS_HPP
