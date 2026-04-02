#pragma once

#include <cstdio>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

inline void fire2_energy(const std::vector<double>& steps, const std::vector<double>& energies) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot("Energy", steps, energies, "b-");
  plt::xlabel("Iteration");
  plt::ylabel(R"($f(x,y)$)");
  plt::title(R"(FIRE2: minimisation of $f = (x-1)^2 + 4(y+2)^2$)");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/fire2_energy.png");
  plt::close();
  std::cout << "  Plot saved: exports/fire2_energy.png\n";
}

inline void ddft_variance(const std::vector<double>& times, const std::vector<double>& variances) {
  namespace plt = matplotlibcpp;
  plt::figure_size(800, 550);
  plt::named_plot(R"($\mathrm{Var}[\rho]$)", times, variances, "b-o");
  plt::xlabel(R"($t \, D / \sigma^2$)");
  plt::ylabel(R"($\mathrm{Var}[\rho(t)]$)");
  plt::title("DDFT: ideal gas density relaxation");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/ddft_variance.png");
  plt::close();
  std::cout << "  Plot saved: exports/ddft_variance.png\n";
}

inline void ddft_density_profiles(
    const std::vector<double>& z,
    const std::vector<std::vector<double>>& profiles,
    const std::vector<double>& snapshot_times,
    double rho0
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 600);

  std::vector<std::string> colors = {"#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#000000"};
  for (std::size_t i = 0; i < profiles.size(); ++i) {
    char label[32];
    std::snprintf(label, sizeof(label), R"($t = %.2f$)", snapshot_times[i]);
    plt::plot(z, profiles[i], {{"color", colors[i % colors.size()]}, {"linestyle", "-"}, {"label", label}});
  }

  // Equilibrium reference line.
  plt::plot({z.front(), z.back()}, {rho0, rho0},
            {{"color", "#00000066"}, {"linestyle", "--"}, {"label", R"($\rho_0$)"}});

  plt::xlabel(R"($z / \sigma$)");
  plt::ylabel(R"($\rho(z) \, \sigma^3$)");
  plt::title("DDFT: density profile evolution (ideal gas)");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/ddft_density_profiles.png");
  plt::close();
  std::cout << "  Plot saved: exports/ddft_density_profiles.png\n";
}

}  // namespace plot

#endif
