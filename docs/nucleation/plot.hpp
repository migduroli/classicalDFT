#pragma once

#include "utils.hpp"

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

    [[nodiscard]] inline auto lerp_color(int r0, int g0, int b0, int r1, int g1, int b1, double frac) -> std::string {
      auto lerp = [](int a, int b, double t) {
        return static_cast<int>(a + t * (b - a));
      };
      char buf[8];
      std::snprintf(buf, sizeof(buf), "#%02X%02X%02X", lerp(r0, r1, frac), lerp(g0, g1, frac), lerp(b0, b1, frac));
      return buf;
    }

    inline void plot_critical_cluster(
        const nucleation::SliceSnapshot& profile,
        const nucleation::SliceSnapshot& initial,
        double rho_v,
        double rho_l,
        const std::string& model_name,
        const std::string& export_dir
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(800, 500);

      plt::plot(
          initial.x,
          initial.rho,
          {{"color", "#9E744F"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"($\rho_0$ initial)"}}
      );

      plt::plot(
          profile.x,
          profile.rho,
          {{"color", "#008080"}, {"linewidth", "2.0"}, {"label", R"($\rho^*(x)$ critical)"}}
      );

      double x_lo = profile.x.front(), x_hi = profile.x.back();
      plt::plot(
          {x_lo, x_hi},
          {rho_v, rho_v},
          {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_v(\mu)$)"}}
      );
      plt::plot(
          {x_lo, x_hi},
          {rho_l, rho_l},
          {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_l(\mu)$)"}}
      );

      plt::xlabel(R"($x / \sigma$)");
      plt::ylabel(R"($\rho \sigma^3$)");
      plt::title(std::format("Critical cluster density profile [{}]", model_name));
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      auto path = export_dir + "/critical_cluster.png";
      plt::save(path);
      plt::clf();
      plt::close();
      std::cout << "  " << path << "\n";
    }

    inline void plot_dynamics(
        const std::vector<nucleation::SliceSnapshot>& snaps,
        const nucleation::SliceSnapshot& critical,
        double rho_v,
        double rho_l,
        const std::string& title,
        int r0,
        int g0,
        int b0,
        int r1,
        int g1,
        int b1,
        const std::string& filename,
        const std::string& model_name
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 550);

      plt::plot(
          critical.x,
          critical.rho,
          {{"color", "black"}, {"linestyle", "--"}, {"linewidth", "1.5"}, {"label", R"($\rho^*$ critical)"}}
      );

      int n = static_cast<int>(snaps.size());
      for (int i = 0; i < n; ++i) {
        double frac = static_cast<double>(i) / std::max(n - 1, 1);
        std::string color = lerp_color(r0, g0, b0, r1, g1, b1, frac);

        // Only label the first (initial) and last (final) curves.
        if (i == 0) {
          plt::plot(
              snaps[i].x,
              snaps[i].rho,
              {
                  {"color", color},
                  {"linewidth", "1.5"},
                  {"label", std::format("t = {:.3f} (initial)", snaps[i].time)},
              }
          );
        } else if (i == n - 1) {
          plt::plot(
              snaps[i].x,
              snaps[i].rho,
              {
                  {"color", color},
                  {"linewidth", "1.5"},
                  {"label", std::format("t = {:.3f} (final)", snaps[i].time)},
              }
          );
        } else {
          plt::plot(snaps[i].x, snaps[i].rho, {{"color", color}, {"linewidth", "1.0"}});
        }
      }

      double x_lo = critical.x.front(), x_hi = critical.x.back();
      plt::plot(
          {x_lo, x_hi},
          {rho_v, rho_v},
          {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_v(\mu)$)"}}
      );
      plt::plot(
          {x_lo, x_hi},
          {rho_l, rho_l},
          {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_l(\mu)$)"}}
      );

      plt::xlabel(R"($x / \sigma$)");
      plt::ylabel(R"($\rho \sigma^3$)");
      plt::ylim(0.0, rho_l * 1.15);
      plt::title(std::format("{} [{}]", title, model_name));
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save(filename);
      plt::clf();
      plt::close();
      std::cout << "  " << filename << "\n";
    }

    // Filter pathway points: keep only those where the cluster is still
    // present (rho_center significantly above rho_background).
    [[nodiscard]] inline auto filter_pathway(const std::vector<nucleation::PathwayPoint>& pts, double rho_bg)
        -> std::vector<nucleation::PathwayPoint> {
      std::vector<nucleation::PathwayPoint> out;
      double thr = rho_bg * 1.5;
      for (const auto& p : pts) {
        if (p.rho_center > thr && p.radius > 0.0)
          out.push_back(p);
      }
      return out;
    }

    inline void plot_energy_barrier(
        const std::vector<nucleation::PathwayPoint>& dissolution,
        const std::vector<nucleation::PathwayPoint>& growth,
        nucleation::PathwayPoint critical,
        double omega_background,
        const std::string& model_name,
        const std::string& export_dir
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 500);

      auto filt_d = filter_pathway(dissolution, critical.rho_center * 0.01);
      auto filt_g = filter_pathway(growth, critical.rho_center * 0.01);

      auto unzip_re = [&](const std::vector<nucleation::PathwayPoint>& pts) {
        std::vector<double> r(pts.size()), de(pts.size());
        for (std::size_t i = 0; i < pts.size(); ++i) {
          r[i] = pts[i].radius;
          de[i] = pts[i].energy - omega_background;
        }
        return std::pair{r, de};
      };

      auto [r_d, de_d] = unzip_re(filt_d);
      auto [r_g, de_g] = unzip_re(filt_g);

      double delta_omega_critical = critical.energy - omega_background;

      plt::plot(
          r_d,
          de_d,
          {{"color", "#00BBD5"},
           {"linewidth", "1.5"},
           {"marker", "o"},
           {"markersize", "4"},
           {"label", R"(Dissolution ($N < N^*$))"}}
      );
      plt::plot(
          r_g,
          de_g,
          {{"color", "#E25822"},
           {"linewidth", "1.5"},
           {"marker", "o"},
           {"markersize", "4"},
           {"label", R"(Growth ($N > N^*$))"}}
      );

      plt::plot(
          {critical.radius},
          {delta_omega_critical},
          {{"color", "black"}, {"marker", "*"}, {"markersize", "14"}, {"label", R"($\Delta\Omega^*$ (saddle))"}}
      );

      double r_min_data = critical.radius, r_max_data = critical.radius;
      double e_min_data = std::min(0.0, delta_omega_critical);
      double e_max_data = std::max(0.0, delta_omega_critical);
      for (double r : r_d) {
        r_min_data = std::min(r_min_data, r);
        r_max_data = std::max(r_max_data, r);
      }
      for (double r : r_g) {
        r_min_data = std::min(r_min_data, r);
        r_max_data = std::max(r_max_data, r);
      }
      for (double e : de_d) {
        e_min_data = std::min(e_min_data, e);
        e_max_data = std::max(e_max_data, e);
      }
      for (double e : de_g) {
        e_min_data = std::min(e_min_data, e);
        e_max_data = std::max(e_max_data, e);
      }

      double r_pad = std::max(0.2 * (r_max_data - r_min_data), 0.5);
      double e_pad = std::max(0.1 * (e_max_data - e_min_data), 1.0);

      plt::plot(
          {r_min_data - r_pad, r_max_data + r_pad},
          {0.0, 0.0},
          {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\Delta\Omega = 0$)"}}
      );

      plt::xlim(r_min_data - r_pad, r_max_data + r_pad);
      plt::ylim(e_min_data - e_pad, e_max_data + e_pad);

      plt::xlabel(R"($R_{\mathrm{eff}} / \sigma$)");
      plt::ylabel(R"($\Delta\Omega\; [k_BT]$)");
      plt::title(std::format(R"(Nucleation energy barrier: $\Delta\Omega$ vs $R_{{\mathrm{{eff}}}}$ [{}])", model_name)
      );
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      auto path = export_dir + "/energy_barrier.png";
      plt::save(path);
      plt::clf();
      plt::close();
      std::cout << "  " << path << "\n";
    }

    inline void plot_rho_center_vs_radius(
        const std::vector<nucleation::PathwayPoint>& dissolution,
        const std::vector<nucleation::PathwayPoint>& growth,
        nucleation::PathwayPoint critical,
        double rho_v,
        double rho_l,
        const std::string& model_name,
        const std::string& export_dir
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 500);

      // Filter out dissolved-state points so the dissolution branch
      // traces cleanly from the saddle toward small R.
      auto filt_d = filter_pathway(dissolution, rho_v);
      auto filt_g = filter_pathway(growth, rho_v);

      auto unzip_rr = [](const std::vector<nucleation::PathwayPoint>& pts) {
        std::vector<double> r(pts.size()), rc(pts.size());
        for (std::size_t i = 0; i < pts.size(); ++i) {
          r[i] = pts[i].radius;
          rc[i] = pts[i].rho_center;
        }
        return std::pair{r, rc};
      };

      auto [r_d, rc_d] = unzip_rr(filt_d);
      auto [r_g, rc_g] = unzip_rr(filt_g);

      plt::plot(
          r_d,
          rc_d,
          {{"color", "#00BBD5"}, {"linewidth", "1.5"}, {"marker", "o"}, {"markersize", "4"}, {"label", R"(Dissolution)"}
          }
      );
      plt::plot(
          r_g,
          rc_g,
          {{"color", "#E25822"}, {"linewidth", "1.5"}, {"marker", "o"}, {"markersize", "4"}, {"label", R"(Growth)"}}
      );

      plt::plot(
          {critical.radius},
          {critical.rho_center},
          {{"color", "black"}, {"marker", "*"}, {"markersize", "14"}, {"label", R"($\rho_0^*$ (saddle))"}}
      );

      double r_min_data = critical.radius, r_max_data = critical.radius;
      for (double r : r_d) {
        r_min_data = std::min(r_min_data, r);
        r_max_data = std::max(r_max_data, r);
      }
      for (double r : r_g) {
        r_min_data = std::min(r_min_data, r);
        r_max_data = std::max(r_max_data, r);
      }
      double r_pad = std::max(0.2 * (r_max_data - r_min_data), 0.5);

      plt::plot(
          {0.0, r_max_data + r_pad},
          {rho_v, rho_v},
          {{"color", "#00976E"}, {"linestyle", "--"}, {"linewidth", "1.0"}, {"label", R"($\rho_v(\mu)$)"}}
      );
      plt::plot(
          {0.0, r_max_data + r_pad},
          {rho_l, rho_l},
          {{"color", "#00976E"}, {"linestyle", "--"}, {"linewidth", "1.0"}, {"label", R"($\rho_l(\mu)$)"}}
      );

      plt::xlim(0.0, r_max_data + r_pad);

      plt::xlabel(R"($R_{\mathrm{eff}} / \sigma$)");
      plt::ylabel(R"($\rho_0\, \sigma^3$)");
      plt::title(std::format(R"(Center density vs effective radius [{}])", model_name));
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      auto path = export_dir + "/rho_center.png";
      plt::save(path);
      plt::clf();
      plt::close();
      std::cout << "  " << path << "\n";
    }

  } // namespace detail

  // Single entry point for all nucleation plots.

  inline void make_plots(
      const nucleation::SliceSnapshot& critical_profile,
      const nucleation::SliceSnapshot& initial_profile,
      const nucleation::DynamicsResult& dissolution,
      const nucleation::DynamicsResult& growth,
      nucleation::PathwayPoint critical_point,
      double omega_background,
      double rho_v,
      double rho_l,
      const std::string& model_name,
      const std::string& export_dir
  ) {
    detail::plot_critical_cluster(critical_profile, initial_profile, rho_v, rho_l, model_name, export_dir);

    detail::plot_dynamics(
        dissolution.profiles,
        critical_profile,
        rho_v,
        rho_l,
        R"(Dissolution dynamics ($N < N^*$))",
        0,
        187,
        213,
        0,
        62,
        62,
        export_dir + "/dissolution.png",
        model_name
    );

    detail::plot_dynamics(
        growth.profiles,
        critical_profile,
        rho_v,
        rho_l,
        R"(Growth dynamics ($N > N^*$))",
        226,
        88,
        34,
        62,
        0,
        0,
        export_dir + "/growth.png",
        model_name
    );

    detail::plot_energy_barrier(
        dissolution.pathway,
        growth.pathway,
        critical_point,
        omega_background,
        model_name,
        export_dir
    );

    detail::plot_rho_center_vs_radius(
        dissolution.pathway,
        growth.pathway,
        critical_point,
        rho_v,
        rho_l,
        model_name,
        export_dir
    );
  }

} // namespace plot

#endif
