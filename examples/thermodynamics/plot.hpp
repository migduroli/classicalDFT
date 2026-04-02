#pragma once

#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

inline void hs_pressure(
    const std::vector<double>& eta,
    const std::vector<double>& cs,
    const std::vector<double>& pyv,
    const std::vector<double>& pyc
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot("Carnahan-Starling", eta, cs, "k-");
  plt::named_plot("PY (virial)", eta, pyv, "r--");
  plt::named_plot("PY (compressibility)", eta, pyc, "b:");
  plt::xlim(0.0, 0.5);
  plt::ylim(0.0, 30.0);
  plt::xlabel(R"($\eta$)");
  plt::ylabel(R"($P / \rho k_BT$)");
  plt::title("Hard-sphere compressibility factor");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/hs_pressure.png");
  plt::close();
  std::cout << "Plot saved: exports/hs_pressure.png\n";
}

inline void transport_viscosity(
    const std::vector<double>& rho,
    const std::vector<double>& shear,
    const std::vector<double>& bulk
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot(R"($\eta_\mathrm{shear}$)", rho, shear, "k-");
  plt::named_plot(R"($\eta_\mathrm{bulk}$)", rho, bulk, "r--");
  plt::xlim(0.0, 0.8);
  plt::xlabel(R"($\rho\sigma^3$)");
  plt::ylabel(R"(Viscosity / $(m k_BT)^{1/2} \sigma^{-2}$)");
  plt::title(R"(Enskog viscosities ($d = k_BT = 1$))");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/transport_viscosity.png");
  plt::close();
  std::cout << "Plot saved: exports/transport_viscosity.png\n";
}

}  // namespace plot

#endif
