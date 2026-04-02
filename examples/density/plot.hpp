#pragma once

#include <cstdio>
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
  plt::figure_size(900, 600);
  plt::plot(rho, p, {{"color", "k"}, {"linestyle", "-"}, {"linewidth", "1.5"}, {"label", R"($P(\rho)$)"}});
  plt::plot({rho_v, rho_l}, {p_coex, p_coex},
            {{"color", "#d62728"}, {"linestyle", "--"}, {"linewidth", "1.2"}, {"label", "Coexistence"}});
  plt::plot({rho_v}, {p_coex}, {{"color", "#d62728"}, {"marker", "o"}, {"markersize", "7"}});
  plt::plot({rho_l}, {p_coex}, {{"color", "#d62728"}, {"marker", "o"}, {"markersize", "7"}});
  plt::plot({0.0, 1.0}, {0.0, 0.0}, {{"color", "#00000033"}, {"linestyle", ":"}});
  plt::xlim(0.0, 1.0);
  plt::ylim(-0.5, 2.0);
  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel(R"($P \sigma^3 / \epsilon$)");
  char title[80];
  std::snprintf(title, sizeof(title), R"(Pressure isotherm at $T^* = %.1f$)", temperature);
  plt::title(title);
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/pressure_isotherm.png");
  plt::close();
  std::cout << "\nPlot saved: exports/pressure_isotherm.png\n";
}

inline void free_energy(
    const std::vector<double>& rho, const std::vector<double>& f,
    double rho_v, double rho_l, double f_v, double f_l, double temperature
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 600);
  plt::plot(rho, f,
            {{"color", "k"}, {"linestyle", "-"}, {"linewidth", "1.5"}, {"label", R"($f(\rho)$)"}});
  plt::plot({rho_v, rho_l}, {f_v, f_l},
            {{"color", "#d62728"}, {"linestyle", "--"}, {"linewidth", "1.2"},
             {"label", "Common tangent"}});
  plt::plot({rho_v}, {f_v}, {{"color", "#d62728"}, {"marker", "o"}, {"markersize", "7"}});
  plt::plot({rho_l}, {f_l}, {{"color", "#d62728"}, {"marker", "o"}, {"markersize", "7"}});
  plt::xlim(0.0, 1.0);
  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel(R"($f(\rho) / k_BT$)");
  char title[80];
  std::snprintf(title, sizeof(title), R"(Free energy density at $T^* = %.1f$)", temperature);
  plt::title(title);
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/free_energy.png");
  plt::close();
  std::cout << "Plot saved: exports/free_energy.png\n";
}

inline void chemical_potential(
    const std::vector<double>& rho, const std::vector<double>& mu,
    double rho_v, double rho_l, double mu_coex, double temperature
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 600);
  plt::plot(rho, mu,
            {{"color", "k"}, {"linestyle", "-"}, {"linewidth", "1.5"}, {"label", R"($\mu(\rho)$)"}});
  plt::plot({0.0, 1.0}, {mu_coex, mu_coex},
            {{"color", "#d62728"}, {"linestyle", "--"}, {"linewidth", "1.2"},
             {"label", R"($\mu_\mathrm{coex}$)"}});
  plt::plot({rho_v}, {mu_coex}, {{"color", "#d62728"}, {"marker", "o"}, {"markersize", "7"}});
  plt::plot({rho_l}, {mu_coex}, {{"color", "#d62728"}, {"marker", "o"}, {"markersize", "7"}});
  plt::xlim(0.0, 1.0);
  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel(R"($\mu / k_BT$)");
  char title[80];
  std::snprintf(title, sizeof(title), R"(Chemical potential at $T^* = %.1f$)", temperature);
  plt::title(title);
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/chemical_potential.png");
  plt::close();
  std::cout << "Plot saved: exports/chemical_potential.png\n";
}

inline void density_evolution(
    const std::vector<double>& x,
    const std::vector<std::vector<double>>& snapshots,
    const std::vector<double>& snapshot_times,
    const std::vector<double>& initial,
    const std::vector<double>& final_profile,
    double rho_v, double rho_l
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 600);

  // Initial profile (dashed).
  plt::plot(x, initial,
            {{"color", "#00000044"}, {"linestyle", "--"}, {"linewidth", "1.0"},
             {"label", "Initial (tanh)"}});

  // Intermediate snapshots.
  std::vector<std::string> colors = {"#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"};
  for (std::size_t i = 1; i < snapshots.size(); ++i) {
    char label[32];
    std::snprintf(label, sizeof(label), R"($t = %.2f$)", snapshot_times[i]);
    plt::plot(x, snapshots[i],
              {{"color", colors[(i - 1) % colors.size()]}, {"linestyle", "-"},
               {"label", label}});
  }

  // Final relaxed profile (thick).
  plt::plot(x, final_profile,
            {{"color", "#1f77b4"}, {"linestyle", "-"}, {"linewidth", "2.0"},
             {"label", "Relaxed"}});

  // Coexistence reference.
  plt::plot({x.front(), x.back()}, {rho_v, rho_v},
            {{"color", "#d6272833"}, {"linestyle", ":"}});
  plt::plot({x.front(), x.back()}, {rho_l, rho_l},
            {{"color", "#2ca02c33"}, {"linestyle", ":"}});

  plt::xlim(x.front(), x.back());
  plt::xlabel(R"($x / \sigma$)");
  plt::ylabel(R"($\rho(x) \, \sigma^3$)");
  plt::title("DDFT relaxation of liquid slab (LJ, White Bear II)");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/density_evolution.png");
  plt::close();
  std::cout << "Plot saved: exports/density_evolution.png\n";
}

inline void grand_potential(
    const std::vector<double>& times, const std::vector<double>& omega
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 600);
  plt::plot(times, omega,
            {{"color", "#1f77b4"}, {"linestyle", "-"}, {"linewidth", "1.5"},
             {"marker", "o"}, {"markersize", "4"}, {"label", R"($\Omega(t)$)"}});
  plt::xlabel(R"($t \, D / \sigma^2$)");
  plt::ylabel(R"($\Omega / k_BT$)");
  plt::title("Grand potential during DDFT relaxation");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/grand_potential.png");
  plt::close();
  std::cout << "Plot saved: exports/grand_potential.png\n";
}

}  // namespace plot

#endif
