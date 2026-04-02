#pragma once

#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

inline void free_energy(
    const std::vector<double>& eta,
    const std::vector<double>& f_ros,
    const std::vector<double>& f_wb1,
    const std::vector<double>& f_wb2
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot("Rosenfeld (= PY comp.)", eta, f_ros, "k-");
  plt::named_plot("White Bear I (= CS)", eta, f_wb1, "b-");
  plt::named_plot("White Bear II", eta, f_wb2, "g:");
  plt::xlim(0.0, 0.5);
  plt::ylim(0.0, 8.0);
  plt::xlabel(R"($\eta$)");
  plt::ylabel(R"($f_\mathrm{ex} / k_BT$)");
  plt::title("Excess free energy per particle: FMT vs EOS");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/fmt_free_energy.png");
  plt::close();
  std::cout << "Plot saved: exports/fmt_free_energy.png\n";
}

inline void pressure(
    const std::vector<double>& eta,
    const std::vector<double>& p_ros,
    const std::vector<double>& p_pyc,
    const std::vector<double>& p_wb1,
    const std::vector<double>& p_cs
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot("Rosenfeld (= PY comp.)", eta, p_ros, "k-");
  plt::named_plot("PY (comp.) exact", eta, p_pyc, "r--");
  plt::named_plot("White Bear I (= CS)", eta, p_wb1, "b-");
  plt::named_plot("CS exact", eta, p_cs, "m:");
  plt::xlim(0.0, 0.5);
  plt::ylim(0.0, 25.0);
  plt::xlabel(R"($\eta$)");
  plt::ylabel(R"($P / (\rho\, k_BT)$)");
  plt::title("Compressibility factor: FMT vs exact EOS");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/fmt_pressure.png");
  plt::close();
  std::cout << "Plot saved: exports/fmt_pressure.png\n";
}

inline void chemical_potential(
    const std::vector<double>& eta,
    const std::vector<double>& mu_ros,
    const std::vector<double>& mu_wb1,
    const std::vector<double>& mu_wb2
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot("Rosenfeld", eta, mu_ros, "k-");
  plt::named_plot("White Bear I", eta, mu_wb1, "b-");
  plt::named_plot("White Bear II", eta, mu_wb2, "g:");
  plt::xlim(0.0, 0.5);
  plt::xlabel(R"($\eta$)");
  plt::ylabel(R"($\mu_\mathrm{ex} / k_BT$)");
  plt::title("Excess chemical potential: FMT models");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/fmt_chemical_potential.png");
  plt::close();
  std::cout << "Plot saved: exports/fmt_chemical_potential.png\n";
}

}  // namespace plot

#endif
