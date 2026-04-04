#pragma once

#include "utils.hpp"

#include "dft/math/spline.hpp"

#include <algorithm>
#include <cstdio>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace detail {

    constexpr int FINE_POINTS = 200;

    [[nodiscard]] inline auto spline_refine(
        const std::vector<double>& x, const std::vector<double>& y
    ) -> std::pair<std::vector<double>, std::vector<double>> {
      if (x.size() < 4) return {x, y};
      dft::math::CubicSpline spline(x, y);
      double x0 = x.front(), x1 = x.back();
      double dx = (x1 - x0) / (FINE_POINTS - 1);
      std::vector<double> xf(FINE_POINTS), yf(FINE_POINTS);
      for (int i = 0; i < FINE_POINTS; ++i) {
        xf[i] = std::min(x0 + i * dx, x1);
        yf[i] = spline(xf[i]);
      }
      return {xf, yf};
    }

    [[nodiscard]] inline auto lerp_color(
        int r0, int g0, int b0, int r1, int g1, int b1, double frac
    ) -> std::string {
      auto lerp = [](int a, int b, double t) { return static_cast<int>(a + t * (b - a)); };
      char buf[8];
      std::snprintf(buf, sizeof(buf), "#%02X%02X%02X",
                    lerp(r0, r1, frac), lerp(g0, g1, frac), lerp(b0, b1, frac));
      return buf;
    }

    inline void plot_critical_cluster(
        const nucleation::SliceSnapshot& profile,
        const nucleation::SliceSnapshot& initial,
        double rho_v, double rho_l
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(800, 500);

      plt::plot(initial.x, initial.rho,
                {{"color", "#9E744F"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"($\rho_0$ initial)"}});

      plt::plot(profile.x, profile.rho,
                {{"color", "#008080"}, {"linewidth", "2.0"}, {"label", R"($\rho^*(x)$ critical)"}});

      double x_lo = profile.x.front(), x_hi = profile.x.back();
      plt::plot({x_lo, x_hi}, {rho_v, rho_v},
                {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_v$)"}});
      plt::plot({x_lo, x_hi}, {rho_l, rho_l},
                {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_l$)"}});

      plt::xlabel(R"($x / \sigma$)");
      plt::ylabel(R"($\rho \sigma^3$)");
      plt::title(R"(Critical cluster density profile (x-slice))");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/critical_cluster.png");
      plt::close();
      std::cout << "  exports/critical_cluster.png\n";
    }

    inline void plot_dynamics(
        const std::vector<nucleation::SliceSnapshot>& snaps,
        const nucleation::SliceSnapshot& critical,
        double rho_v, double rho_l,
        const std::string& title,
        int r0, int g0, int b0, int r1, int g1, int b1,
        const std::string& filename
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 550);

      plt::plot(critical.x, critical.rho,
                {{"color", "black"}, {"linestyle", "--"}, {"linewidth", "1.5"},
                 {"label", R"($\rho^*(x)$ critical)"}});

      int n = static_cast<int>(snaps.size());
      for (int i = 0; i < n; ++i) {
        double frac = static_cast<double>(i) / std::max(n - 1, 1);
        plt::plot(snaps[i].x, snaps[i].rho, {
            {"color", lerp_color(r0, g0, b0, r1, g1, b1, frac)},
            {"linewidth", "1.5"},
            {"label", std::format("t = {:.3f}", snaps[i].time)},
        });
      }

      double x_lo = critical.x.front(), x_hi = critical.x.back();
      plt::plot({x_lo, x_hi}, {rho_v, rho_v},
                {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_v$)"}});
      plt::plot({x_lo, x_hi}, {rho_l, rho_l},
                {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_l$)"}});

      plt::xlabel(R"($x / \sigma$)");
      plt::ylabel(R"($\rho \sigma^3$)");
      plt::ylim(0.0, rho_l * 1.15);
      plt::title(title);
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save(filename);
      plt::close();
      std::cout << "  " << filename << "\n";
    }

    inline void plot_energy_barrier(
        const std::vector<nucleation::PathwayPoint>& dissolution,
        const std::vector<nucleation::PathwayPoint>& growth,
        nucleation::PathwayPoint critical,
        double omega_background
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 500);

      auto unzip = [](const std::vector<nucleation::PathwayPoint>& pts) {
        std::vector<double> r(pts.size()), e(pts.size());
        for (std::size_t i = 0; i < pts.size(); ++i) {
          r[i] = pts[i].radius;
          e[i] = pts[i].energy;
        }
        return std::pair{r, e};
      };

      auto [r_d, e_d] = unzip(dissolution);
      auto [r_g, e_g] = unzip(growth);

      plt::plot(r_d, e_d, {{"color", "#00BBD5"}, {"linewidth", "1.5"}, {"marker", "o"},
                            {"markersize", "4"}, {"label", R"(Dissolution ($N < N^*$))"}});
      plt::plot(r_g, e_g, {{"color", "#E25822"}, {"linewidth", "1.5"}, {"marker", "o"},
                            {"markersize", "4"}, {"label", R"(Growth ($N > N^*$))"}});

      plt::plot({critical.radius}, {critical.energy},
                {{"color", "black"}, {"marker", "*"}, {"markersize", "14"}, {"label", R"($\Omega^*$ (saddle))"}});

      // Compute axis range from all data.
      double r_min_data = critical.radius, r_max_data = critical.radius;
      double e_min_data = std::min(omega_background, critical.energy);
      double e_max_data = std::max(omega_background, critical.energy);
      for (double r : r_d) { r_min_data = std::min(r_min_data, r); r_max_data = std::max(r_max_data, r); }
      for (double r : r_g) { r_min_data = std::min(r_min_data, r); r_max_data = std::max(r_max_data, r); }
      for (double e : e_d) { e_min_data = std::min(e_min_data, e); e_max_data = std::max(e_max_data, e); }
      for (double e : e_g) { e_min_data = std::min(e_min_data, e); e_max_data = std::max(e_max_data, e); }

      double r_pad = std::max(0.2 * (r_max_data - r_min_data), 0.5);
      double e_pad = std::max(0.1 * (e_max_data - e_min_data), 1.0);

      plt::plot({r_min_data - r_pad, r_max_data + r_pad}, {omega_background, omega_background},
                {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"},
                 {"label", R"($\Omega_{\mathrm{bg}}$)"}});

      plt::xlim(r_min_data - r_pad, r_max_data + r_pad);
      plt::ylim(e_min_data - e_pad, e_max_data + e_pad);

      plt::xlabel(R"($R_{\mathrm{eff}} / \sigma$)");
      plt::ylabel(R"($\Omega\; [k_BT]$)");
      plt::title(R"(Nucleation energy barrier)");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/energy_barrier.png");
      plt::close();
      std::cout << "  exports/energy_barrier.png\n";
    }

  }  // namespace detail

  // Single entry point for all nucleation plots.

  inline void make_plots(
      const nucleation::SliceSnapshot& critical_profile,
      const nucleation::SliceSnapshot& initial_profile,
      const nucleation::DynamicsResult& dissolution,
      const nucleation::DynamicsResult& growth,
      nucleation::PathwayPoint critical_point,
      double omega_background,
      double rho_v, double rho_l
  ) {
    detail::plot_critical_cluster(critical_profile, initial_profile, rho_v, rho_l);

    detail::plot_dynamics(
        dissolution.profiles, critical_profile, rho_v, rho_l,
        R"(Dissolution dynamics ($N < N^*$))",
        0, 187, 213, 0, 62, 62,
        "exports/dissolution.png");

    detail::plot_dynamics(
        growth.profiles, critical_profile, rho_v, rho_l,
        R"(Growth dynamics ($N > N^*$))",
        226, 88, 34, 62, 0, 0,
        "exports/growth.png");

    detail::plot_energy_barrier(
        dissolution.pathway, growth.pathway, critical_point, omega_background);
  }

}  // namespace plot

#endif
