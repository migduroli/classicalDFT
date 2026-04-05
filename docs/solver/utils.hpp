#pragma once

#include "dft.hpp"

namespace utils {

  // Plot data containers for coexistence and spinodal curves.

  struct CoexData {
    std::string name;
    arma::vec T, rho_v, rho_l;
    double Tc{ 0.0 }, rho_c{ 0.0 };
  };

  struct SpinodalData {
    std::string name;
    arma::vec T, rho_lo, rho_hi;
  };

  // Jim's single-temperature coexistence points (scan + bisection).

  struct JimCoexPoints {
    std::vector<double> T, rho_v, rho_l;
  };

  // Jim's single-temperature spinodal points (scan + bisection).

  struct JimSpinodalPoints {
    std::vector<double> T, rho_lo, rho_hi;
  };

  // Temperature-dependent bulk EoS factory (parameterised by FMT model).

  [[nodiscard]] inline auto make_eos_factory(
      const std::vector<dft::Species>& species,
      const std::vector<dft::physics::Interaction>& interactions
  ) {
    return [&species, &interactions](const dft::functionals::fmt::FMTModel& fmt_model, double kT)
               -> dft::functionals::bulk::BulkThermodynamics {
      return dft::functionals::bulk::make_bulk_thermodynamics(
          species,
          dft::functionals::make_bulk_weights(fmt_model, interactions, kT)
      );
    };
  }

}  // namespace utils
