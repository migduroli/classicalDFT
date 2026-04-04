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
  // beta F_id = sum_i int rho_i(r) [ln(rho_i(r)) - 1] dV
  // force_i(r) = [ln(rho_i(r)) - mu_i / kT] dV

  [[nodiscard]] inline auto ideal_gas(const Grid& grid, const State& state) -> Contribution {
    double dv = grid.cell_volume();
    double kT = state.temperature;

    Contribution result;
    result.forces.reserve(state.species.size());

    for (const auto& sp : state.species) {
      const arma::vec& rho = sp.density.values;
      arma::vec log_rho = arma::log(arma::clamp(rho, 1e-300, arma::datum::inf));

      result.free_energy += arma::dot(rho, log_rho - 1.0) * dv;

      // Force: delta(beta Omega) / delta rho_i * dV.
      // chemical_potential is stored in dimensionless beta*mu units.
      arma::vec f = (log_rho - sp.chemical_potential) * dv;
      result.forces.push_back(std::move(f));
    }

    return result;
  }

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_IDEAL_GAS_HPP
