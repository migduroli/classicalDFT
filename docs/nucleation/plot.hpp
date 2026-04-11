#pragma once

#include "utils.hpp"

#include "dft/plotting/matplotlib.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
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

    struct ColorWindow {
      double vmin;
      double vmax;
    };

    struct FrameSize {
      int width;
      int height;
    };

    struct DensityViews {
      nucleation::Slice2D top;
      nucleation::Slice2D bottom;
    };

    [[nodiscard]] inline auto wall_axis_label(const nucleation::NucleationConfig& cfg) -> std::string {
      return cfg.wall.is_active() ? dft::Grid::axis_name(nucleation::wall_axis(cfg.wall.normal)) : "";
    }

    inline void add_wall_line_1d(const nucleation::NucleationConfig& cfg, std::string_view axis, double y_max) {
      if (!cfg.wall.is_active() || axis != wall_axis_label(cfg))
        return;

      namespace plt = matplotlibcpp;
      plt::plot(
          {0.0, 0.0},
          {0.0, y_max},
          {{"color", "black"}, {"linestyle", "-"}, {"linewidth", "1.2"}, {"label", "wall"}}
      );
    }

    [[nodiscard]] inline auto wall_lines_2d(const nucleation::NucleationConfig& cfg, const nucleation::Slice2D& slice)
        -> std::vector<dft::plotting::ContourfLine> {
      if (!cfg.wall.is_active()) {
        return {};
      }

      std::string axis = wall_axis_label(cfg);

      if (slice.y_label == axis) {
        double x0 = slice.x.front().front();
        double x1 = slice.x.front().back();
        return {{.x = {x0, x1}, .y = {0.0, 0.0}, .linewidth = 1.4}};
      }

      if (slice.x_label == axis) {
        double y0 = slice.y.front().front();
        double y1 = slice.y.back().front();
        return {{.x = {0.0, 0.0}, .y = {y0, y1}, .linewidth = 1.4}};
      }

      return {};
    }

    [[nodiscard]] inline auto packing_window(const nucleation::Slice2D& slice, double rho_v, double rho_l)
        -> ColorWindow {
      std::vector<double> values;
      values.reserve(static_cast<std::size_t>(slice.nx * slice.ny));

      double threshold = rho_v + 0.60 * (rho_l - rho_v);
      for (const auto& row : slice.z) {
        for (double value : row) {
          if (value > threshold)
            values.push_back(value);
        }
      }

      if (values.size() < 16)
        return {.vmin = rho_v, .vmax = rho_l};

      std::sort(values.begin(), values.end());
      auto sample = [&](double q) {
        std::size_t index = static_cast<std::size_t>(q * static_cast<double>(values.size() - 1));
        return values[index];
      };

      double lo = sample(0.05);
      double hi = sample(0.95);
      double span = std::max(hi - lo, 0.04 * std::max(rho_l, 1.0));
      double pad = 0.18 * span;
      double vmin = std::max(threshold, lo - pad);
      double vmax = hi + pad;
      if (vmax <= vmin)
        vmax = vmin + span;
      return {.vmin = vmin, .vmax = vmax};
    }

    [[nodiscard]] inline auto packing_window(const DensityViews& views, double rho_v, double rho_l) -> ColorWindow {
      std::vector<double> values;
      values.reserve(
          static_cast<std::size_t>(views.top.nx * views.top.ny)
          + static_cast<std::size_t>(views.bottom.nx * views.bottom.ny)
      );

      double threshold = rho_v + 0.60 * (rho_l - rho_v);
      auto collect = [&](const nucleation::Slice2D& slice) {
        for (const auto& row : slice.z) {
          for (double value : row) {
            if (value > threshold)
              values.push_back(value);
          }
        }
      };

      collect(views.top);
      collect(views.bottom);

      if (values.size() < 16)
        return {.vmin = rho_v, .vmax = rho_l};

      std::sort(values.begin(), values.end());
      auto sample = [&](double q) {
        std::size_t index = static_cast<std::size_t>(q * static_cast<double>(values.size() - 1));
        return values[index];
      };

      double lo = sample(0.05);
      double hi = sample(0.95);
      double span = std::max(hi - lo, 0.04 * std::max(rho_l, 1.0));
      double pad = 0.18 * span;
      double vmin = std::max(threshold, lo - pad);
      double vmax = hi + pad;
      if (vmax <= vmin)
        vmax = vmin + span;
      return {.vmin = vmin, .vmax = vmax};
    }

    [[nodiscard]] inline auto density_frame_size(const nucleation::Slice2D& slice) -> FrameSize {
      constexpr int SHORT_SIDE_PIXELS = 560;
      constexpr int MAX_SIDE_PIXELS = 1400;

      if (slice.x.empty() || slice.x.front().empty() || slice.y.empty() || slice.y.front().empty()) {
        return {.width = SHORT_SIDE_PIXELS, .height = SHORT_SIDE_PIXELS};
      }

      double x_span = std::abs(slice.x.front().back() - slice.x.front().front());
      double y_span = std::abs(slice.y.back().front() - slice.y.front().front());
      x_span = std::max(x_span, 1e-12);
      y_span = std::max(y_span, 1e-12);

      if (x_span >= y_span) {
        int width = std::clamp(
            static_cast<int>(std::lround(SHORT_SIDE_PIXELS * x_span / y_span)),
            SHORT_SIDE_PIXELS,
            MAX_SIDE_PIXELS
        );
        return {.width = width, .height = SHORT_SIDE_PIXELS};
      }

      int height = std::clamp(
          static_cast<int>(std::lround(SHORT_SIDE_PIXELS * y_span / x_span)),
          SHORT_SIDE_PIXELS,
          MAX_SIDE_PIXELS
      );
      return {.width = SHORT_SIDE_PIXELS, .height = height};
    }

    [[nodiscard]] inline auto density_figure_size(const DensityViews& views) -> FrameSize {
      auto top_size = density_frame_size(views.top);
      auto bottom_size = density_frame_size(views.bottom);
      return {
          .width = std::max(top_size.width, bottom_size.width) + 80,
          .height = top_size.height + bottom_size.height + 180,
      };
    }

    [[nodiscard]] inline auto
    density_views(const arma::vec& rho, const dft::Grid& grid, const nucleation::NucleationConfig& cfg)
        -> DensityViews {
      auto wall = cfg.wall.is_active() ? std::optional<nucleation::WallConfig>{cfg.wall} : std::nullopt;
      auto seed_origin = nucleation::seed_center(grid, cfg);
      long x_index = std::clamp(static_cast<long>(std::llround(seed_origin[0] / grid.dx)), 0L, grid.shape[0] - 1);
      long y_index = std::clamp(static_cast<long>(std::llround(seed_origin[1] / grid.dx)), 0L, grid.shape[1] - 1);
      return {
          .top = nucleation::extract_xz_slice(rho, grid, y_index, wall),
          .bottom = nucleation::extract_yz_slice(rho, grid, x_index, wall),
      };
    }

    [[nodiscard]] inline auto contour_panel(
        const nucleation::Slice2D& slice,
        double rho_min,
        double rho_max,
        const std::string& title,
        const nucleation::NucleationConfig& cfg,
        const std::string& cmap,
        int levels,
        const std::string& extend
    ) -> dft::plotting::ContourfPanel {
      return {
          .field = {.x = slice.x, .y = slice.y, .z = slice.z},
          .options =
              {.levels = levels,
               .cmap = cmap,
               .vmin = rho_min,
               .vmax = rho_max,
               .extend = extend,
               .xlabel = std::format(R"(${} / \sigma$)", slice.x_label),
               .ylabel = std::format(R"(${} / \sigma$)", slice.y_label),
               .title = title,
               .square_axes = true,
               .colorbar = true,
               .shrink = 0.82,
               .pad = 0.03},
          .lines = wall_lines_2d(cfg, slice),
      };
    }

    inline void plot_critical_cluster(
        const nucleation::SliceSnapshot& profile,
        const nucleation::SliceSnapshot& initial,
        double rho_v,
        double rho_l,
        const std::string& model_name,
        const nucleation::NucleationConfig& cfg,
        const std::string& export_dir
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(800, 500);

      plt::plot(
          initial.x,
          initial.values,
          {{"color", "#9E744F"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"($\rho_0$ initial)"}}
      );

      plt::plot(
          profile.x,
          profile.values,
          {{"color", "#008080"}, {"linewidth", "2.0"}, {"label", std::format(R"($\rho^*({})$ critical)", profile.axis)}}
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

      add_wall_line_1d(cfg, profile.axis, rho_l * 1.15);

      plt::xlabel(std::format(R"(${} / \sigma$)", profile.axis));
      plt::ylabel(R"($\rho \sigma^3$)");
      plt::title(std::format("Critical cluster density profile along {} [{}]", profile.axis, model_name));
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      std::filesystem::create_directories(export_dir + "/critical");
      auto path = export_dir + "/critical/profile.pdf";
      plt::save(path);
      plt::clf();
      plt::close();
      std::cout << "  " << path << "\n";
    }

    inline void plot_initial_profile(
        const nucleation::SliceSnapshot& initial,
        double rho_v,
        double rho_l,
        const std::string& model_name,
        const nucleation::NucleationConfig& cfg,
        const std::string& export_dir
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(800, 500);

      plt::plot(
          initial.x,
          initial.values,
          {{"color", "#9E744F"}, {"linewidth", "2.0"}, {"label", std::format(R"($\rho_0({})$)", initial.axis)}}
      );

      double x_lo = initial.x.front(), x_hi = initial.x.back();
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

      add_wall_line_1d(cfg, initial.axis, rho_l * 1.15);

      plt::xlabel(std::format(R"(${} / \sigma$)", initial.axis));
      plt::ylabel(R"($\rho \sigma^3$)");
      plt::title(std::format("Initial density profile along {} [{}]", initial.axis, model_name));
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      std::filesystem::create_directories(export_dir + "/initial");
      auto path = export_dir + "/initial/profile.pdf";
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
          critical.values,
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
              snaps[i].values,
              {
                  {"color", color},
                  {"linewidth", "1.5"},
                  {"label", std::format("t = {:.3f} (initial)", snaps[i].time)},
              }
          );
        } else if (i == n - 1) {
          plt::plot(
              snaps[i].x,
              snaps[i].values,
              {
                  {"color", color},
                  {"linewidth", "1.5"},
                  {"label", std::format("t = {:.3f} (final)", snaps[i].time)},
              }
          );
        } else {
          plt::plot(snaps[i].x, snaps[i].values, {{"color", color}, {"linewidth", "1.0"}});
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

      plt::xlabel(std::format(R"(${} / \sigma$)", critical.axis));
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
      std::filesystem::create_directories(export_dir + "/dynamics");
      auto path = export_dir + "/dynamics/energy_barrier.pdf";
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
      std::filesystem::create_directories(export_dir + "/dynamics");
      auto path = export_dir + "/dynamics/rho_center.pdf";
      plt::save(path);
      plt::clf();
      plt::close();
      std::cout << "  " << path << "\n";
    }

    inline void plot_density_views(
        const DensityViews& views,
        double rho_min,
        double rho_max,
        const std::string& title_prefix,
        const std::string& filepath,
        const nucleation::NucleationConfig& cfg,
        const std::string& cmap = "viridis",
        int levels = 64,
        const std::string& extend = "neither"
    ) {
      auto figure_size = density_figure_size(views);

      dft::plotting::Figure figure{
          .panels =
              {
                  contour_panel(views.top, rho_min, rho_max, "rho(x, z)", cfg, cmap, levels, extend),
                  contour_panel(views.bottom, rho_min, rho_max, "rho(y, z)", cfg, cmap, levels, extend),
              },
          .layout =
              {
                  .width = figure_size.width,
                  .height = figure_size.height,
                  .columns = 1,
                  .shared_colorbar = true,
                  .colorbar_orientation = "horizontal",
                  .shrink = 0.96,
                  .pad = 0.06,
                  .suptitle = title_prefix,
                  .suptitle_y = 0.985,
                  .left = 0.08,
                  .right = 0.97,
                  .bottom = 0.16,
                  .top = 0.95,
                  .wspace = 0.12,
                  .hspace = 0.34,
              },
      };
      figure.save(filepath);
      std::cout << "  " << filepath << "\n";
    }

  } // namespace detail

  inline void plot_density_frames(
      const dft::algorithms::dynamics::SimulationResult& sim,
      const dft::Grid& grid,
      const nucleation::NucleationConfig& cfg,
      double rho_min,
      double rho_max,
      const std::string& label,
      const std::string& frame_dir
  ) {
    std::filesystem::create_directories(frame_dir);
    std::println(std::cout, "  Writing {} frames to {}/", sim.snapshots.size(), frame_dir);
    for (std::size_t i = 0; i < sim.snapshots.size(); ++i) {
      auto views = detail::density_views(sim.snapshots[i].densities[0], grid, cfg);
      auto title = std::format(R"({}: $t = {:.4f}$)", label, sim.snapshots[i].time);
      auto filepath = std::format("{}/frame_{:05d}.pdf", frame_dir, i);
      detail::plot_density_views(views, rho_min, rho_max, title, filepath, cfg);
    }
  }

  inline void plot_initial_condition(
      const nucleation::SliceSnapshot& initial_profile,
      const arma::vec& initial_density,
      const dft::Grid& grid,
      const nucleation::NucleationConfig& cfg,
      double rho_v,
      double rho_l,
      const std::string& model_name,
      const std::string& export_dir
  ) {
    detail::plot_initial_profile(initial_profile, rho_v, rho_l, model_name, cfg, export_dir);

    auto views = detail::density_views(initial_density, grid, cfg);

    detail::plot_density_views(
        views,
        rho_v,
        rho_l,
        std::format(R"(Initial condition [{}])", model_name),
        export_dir + "/initial/sections.pdf",
        cfg,
        "cividis",
        96
    );

    auto window = detail::packing_window(views, rho_v, rho_l);

    detail::plot_density_views(
        views,
        window.vmin,
        window.vmax,
        std::format(R"(Initial condition packing contrast [{}])", model_name),
        export_dir + "/initial/sections_packing.pdf",
        cfg,
        "turbo",
        128,
        "both"
    );
  }

  // Single entry point for all nucleation plots.

  inline void make_plots(
      const nucleation::SliceSnapshot& critical_profile,
      const nucleation::SliceSnapshot& initial_profile,
      const arma::vec& critical_density,
      const nucleation::DynamicsResult& dissolution,
      const nucleation::DynamicsResult& growth,
      const dft::algorithms::dynamics::SimulationResult& sim_dissolution,
      const dft::algorithms::dynamics::SimulationResult& sim_growth,
      const dft::Grid& grid,
      const nucleation::NucleationConfig& cfg,
      nucleation::PathwayPoint critical_point,
      double omega_background,
      double rho_v,
      double rho_l,
      const std::string& model_name,
      const std::string& export_dir
  ) {
    std::filesystem::create_directories(export_dir + "/critical");
    std::filesystem::create_directories(export_dir + "/dynamics");

    detail::plot_critical_cluster(critical_profile, initial_profile, rho_v, rho_l, model_name, cfg, export_dir);

    auto views = detail::density_views(critical_density, grid, cfg);

    detail::plot_density_views(
        views,
        rho_v,
        rho_l,
        std::format(R"(Critical cluster [{}])", model_name),
        export_dir + "/critical/sections.pdf",
        cfg
    );

    auto critical_window = detail::packing_window(views, rho_v, rho_l);
    detail::plot_density_views(
        views,
        critical_window.vmin,
        critical_window.vmax,
        std::format(R"(Critical cluster packing contrast [{}])", model_name),
        export_dir + "/critical/sections_packing.pdf",
        cfg,
        "turbo",
        128,
        "both"
    );

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
        export_dir + "/dynamics/dissolution_profiles.pdf",
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
        export_dir + "/dynamics/growth_profiles.pdf",
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

    plot_density_frames(sim_growth, grid, cfg, rho_v, rho_l, "Growth", export_dir + "/frames/growth");
    plot_density_frames(sim_dissolution, grid, cfg, rho_v, rho_l, "Dissolution", export_dir + "/frames/dissolution");
  }

} // namespace plot

#endif
