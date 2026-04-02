#pragma once

#include "dft/math/spline.hpp"

#include <cstdio>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

constexpr int fine_grid_points = 500;

inline auto spline_refine(
    const std::vector<double>& x, const std::vector<double>& y, int n_fine = fine_grid_points
) -> std::pair<std::vector<double>, std::vector<double>> {
  dft::math::CubicSpline spline(x, y);
  double x0 = x.front();
  double x1 = x.back();
  double dx = (x1 - x0) / (n_fine - 1);
  std::vector<double> xf(n_fine), yf(n_fine);
  for (int i = 0; i < n_fine; ++i) {
    xf[i] = x0 + i * dx;
    yf[i] = spline(xf[i]);
  }
  return {xf, yf};
}

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

  auto [xi, yi] = spline_refine(x, initial);
  plt::plot(xi, yi,
            {{"color", "#00000044"}, {"linestyle", "--"}, {"linewidth", "1.0"},
             {"label", "Initial (tanh)"}});

  std::vector<std::string> colors = {"#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"};
  for (std::size_t i = 1; i < snapshots.size(); ++i) {
    char label[32];
    std::snprintf(label, sizeof(label), R"($t = %.3f$)", snapshot_times[i]);
    auto [xs, ys] = spline_refine(x, snapshots[i]);
    plt::plot(xs, ys,
              {{"color", colors[(i - 1) % colors.size()]}, {"linestyle", "-"},
               {"label", label}});
  }

  auto [xf, yf] = spline_refine(x, final_profile);
  plt::plot(xf, yf,
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
