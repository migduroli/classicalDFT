#pragma once

#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

inline void pressure_isotherm(
    const std::vector<double>& rho, const std::vector<double>& p,
    double rho_v, double rho_l, double p_coex, double temperature
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot(R"($P(\rho)$)", rho, p, "k-");
  plt::plot({rho_v, rho_l}, {p_coex, p_coex},
            {{"color", "r"}, {"linestyle", "--"}, {"label", "Coexistence"}});
  plt::plot({rho_v}, {p_coex}, "ro");
  plt::plot({rho_l}, {p_coex}, "ro");
  plt::xlim(0.0, 1.0);
  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel(R"($P \sigma^3 / \epsilon$)");
  char title[64];
  std::snprintf(title, sizeof(title), R"(Pressure isotherm at $T^* = %.1f$)", temperature);
  plt::title(title);
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/pressure_isotherm.png");
  plt::close();
  std::cout << "\nPlot saved: exports/pressure_isotherm.png\n";
}

inline void density_profile(
    const std::vector<double>& x, const std::vector<double>& profile,
    double rho_v, double rho_l
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot(R"($\rho(x)$)", x, profile, "b-");
  plt::plot({x.front(), x.back()}, {rho_v, rho_v},
            {{"color", "r"}, {"linestyle", ":"}, {"label", R"($\rho_\mathrm{vapor}$)"}});
  plt::plot({x.front(), x.back()}, {rho_l, rho_l},
            {{"color", "g"}, {"linestyle", ":"}, {"label", R"($\rho_\mathrm{liquid}$)"}});
  plt::xlabel(R"($x / \sigma$)");
  plt::ylabel(R"($\rho \sigma^3$)");
  plt::title("Liquid slab density profile");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/density_profile.png");
  plt::close();
  std::cout << "Plot saved: exports/density_profile.png\n";
}

inline void free_energy(
    const std::vector<double>& rho, const std::vector<double>& f, double temperature
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot(R"($f(\rho)$)", rho, f, "k-");
  plt::xlim(0.0, 1.0);
  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel(R"($f / k_BT$)");
  char title[64];
  std::snprintf(title, sizeof(title), R"(Free energy density at $T^* = %.1f$)", temperature);
  plt::title(title);
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/free_energy.png");
  plt::close();
  std::cout << "Plot saved: exports/free_energy.png\n";
}

}  // namespace plot

#endif
