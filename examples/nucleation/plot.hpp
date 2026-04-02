#pragma once

#include <cstdio>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

inline void droplet_evolution(
    const std::vector<double>& x,
    const std::vector<std::vector<double>>& snapshots,
    const std::vector<double>& snapshot_times,
    const std::vector<double>& initial,
    const std::vector<double>& final_profile,
    double rho_v, double rho_l
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 600);

  plt::plot(x, initial,
            {{"color", "#00000044"}, {"linestyle", "--"}, {"linewidth", "1.0"},
             {"label", "Initial (tanh)"}});

  std::vector<std::string> colors = {"#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"};
  for (std::size_t i = 1; i < snapshots.size(); ++i) {
    char label[32];
    std::snprintf(label, sizeof(label), R"($t = %.3f$)", snapshot_times[i]);
    plt::plot(x, snapshots[i],
              {{"color", colors[(i - 1) % colors.size()]}, {"linestyle", "-"},
               {"label", label}});
  }

  plt::plot(x, final_profile,
            {{"color", "#1f77b4"}, {"linestyle", "-"}, {"linewidth", "2.0"},
             {"label", "Final"}});

  plt::plot({x.front(), x.back()}, {rho_v, rho_v},
            {{"color", "#d6272833"}, {"linestyle", ":"}});
  plt::plot({x.front(), x.back()}, {rho_l, rho_l},
            {{"color", "#2ca02c33"}, {"linestyle", ":"}});

  plt::xlim(x.front(), x.back());
  plt::xlabel(R"($x / \sigma$)");
  plt::ylabel(R"($\rho(x) \, \sigma^3$)");
  plt::title("DDFT nucleation: droplet in supersaturated vapor");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/droplet_evolution.png");
  plt::close();
  std::cout << "Plot saved: exports/droplet_evolution.png\n";
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
  plt::title("Grand potential during nucleation");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/grand_potential.png");
  plt::close();
  std::cout << "Plot saved: exports/grand_potential.png\n";
}

}  // namespace plot

#endif
