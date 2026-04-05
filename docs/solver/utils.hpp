#pragma once

#include "dft.hpp"

namespace utils {

  // Plot data containers for coexistence and spinodal curves.

  struct CoexData {
    std::string name;
    arma::vec T, rho_v, rho_l;
    double Tc{0.0}, rho_c{0.0};
  };

  struct SpinodalData {
    std::string name;
    arma::vec T, rho_lo, rho_hi;
  };

  // Temperature-dependent bulk weight factory.

  [[nodiscard]] inline auto make_weight_factory(
      const std::vector<dft::physics::Interaction>& interactions
  ) {
    return [&interactions](const dft::functionals::fmt::FMTModel& fmt_model, double kT) {
      return dft::functionals::make_bulk_weights(fmt_model, interactions, kT);
    };
  }

}  // namespace utils
