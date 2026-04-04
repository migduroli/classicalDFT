#pragma once

#include "dft/math/spline.hpp"

#include <format>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

namespace detail {

constexpr int fine_grid_points = 500;

// Evaluate a cubic spline on a uniform fine grid.
inline auto spline_refine(
    const std::vector<double>& x, const std::vector<double>& y, int n_fine = fine_grid_points
) -> std::pair<std::vector<double>, std::vector<double>> {
  dft::math::CubicSpline spline(x, y);
  double x0 = x.front();
  double x1 = x.back();
  double dx = (x1 - x0) / (n_fine - 1);
  std::vector<double> xf(n_fine), yf(n_fine);
  for (int i = 0; i < n_fine; ++i) {
    xf[i] = std::min(x0 + i * dx, x1);
    yf[i] = spline(xf[i]);
  }
  return {xf, yf};
}

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
  auto title = std::format(R"(Pressure isotherm at $T^* = {:.1f}$)", temperature);
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
  auto title = std::format(R"(Free energy density at $T^* = {:.1f}$)", temperature);
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
  auto title = std::format(R"(Chemical potential at $T^* = {:.1f}$)", temperature);
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

  // Initial profile (dashed, spline-interpolated).
  auto [xi, yi] = spline_refine(x, initial);
  plt::plot(xi, yi,
            {{"color", "#00000044"}, {"linestyle", "--"}, {"linewidth", "1.0"},
             {"label", "Initial (tanh)"}});

  // Intermediate snapshots (spline-interpolated).
  std::vector<std::string> colors = {"#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"};
  for (std::size_t i = 1; i < snapshots.size(); ++i) {
    auto label = std::format(R"($t = {:.2f}$)", snapshot_times[i]);
    auto [xs, ys] = spline_refine(x, snapshots[i]);
    plt::plot(xs, ys,
              {{"color", colors[(i - 1) % colors.size()]}, {"linestyle", "-"},
               {"label", label}});
  }

  // Final relaxed profile (thick, spline-interpolated).
  auto [xf, yf] = spline_refine(x, final_profile);
  plt::plot(xf, yf,
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

}  // namespace detail

inline void make_plots(
    const std::vector<double>& rho_range, const std::vector<double>& p_range,
    const std::vector<double>& f_range, const std::vector<double>& mu_range,
    double rho_v, double rho_l, double p_coex, double f_v, double f_l,
    double mu_coex, double temperature,
    const std::vector<double>& x_coords,
    const std::vector<std::vector<double>>& profile_snapshots,
    const std::vector<double>& snapshot_times,
    const std::vector<double>& initial_profile,
    const std::vector<double>& final_profile,
    const std::vector<double>& times, const std::vector<double>& energies
) {
  detail::pressure_isotherm(rho_range, p_range, rho_v, rho_l, p_coex, temperature);
  detail::free_energy(rho_range, f_range, rho_v, rho_l, f_v, f_l, temperature);
  detail::chemical_potential(rho_range, mu_range, rho_v, rho_l, mu_coex, temperature);
  detail::density_evolution(x_coords, profile_snapshots, snapshot_times,
                            initial_profile, final_profile, rho_v, rho_l);
  detail::grand_potential(times, energies);
}

}  // namespace plot

#endif
