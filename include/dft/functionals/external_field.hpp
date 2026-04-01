#ifndef DFT_FUNCTIONALS_EXTERNAL_FIELD_HPP
#define DFT_FUNCTIONALS_EXTERNAL_FIELD_HPP

#include "dft/functionals/ideal_gas.hpp"
#include "dft/grid.hpp"
#include "dft/types.hpp"

#include <armadillo>

namespace dft::functionals {

  // External field contribution to the free energy and forces.
  // F_ext = sum_i int rho_i(r) V_ext_i(r) dV
  // force_i(r) = V_ext_i(r) dV

  [[nodiscard]] inline auto external_field(const Grid& grid, const State& state) -> Contribution {
    double dv = grid.cell_volume();

    Contribution result;
    result.forces.reserve(state.species.size());

    for (const auto& sp : state.species) {
      const arma::vec& rho = sp.density.values;
      const arma::vec& vext = sp.density.external_field;

      result.free_energy += arma::dot(rho, vext) * dv;
      result.forces.push_back(vext * dv);
    }

    return result;
  }

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_EXTERNAL_FIELD_HPP
