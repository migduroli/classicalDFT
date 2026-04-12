#pragma once

#include <dft/plotting/matplotlib.hpp>

#include <cmath>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace detail {

    // Plot 1: energy landscape contour with initial, simplified, and full paths.

    inline void energy_landscape(
        const std::vector<double>& path_x_init,
        const std::vector<double>& path_y_init,
        const std::vector<double>& path_x_simplified,
        const std::vector<double>& path_y_simplified,
        const std::vector<double>& path_x_full,
        const std::vector<double>& path_y_full,
        const std::vector<double>& theo_mep_x,
        const std::vector<double>& theo_mep_y
    ) {
      // Build contour grid for V(x, y) = (x^2 - 1)^2 + 10*(y - x^2)^2.
      int nx = 200, ny = 200;
      double x_lo = -1.5, x_hi = 1.5;
      double y_lo = -0.3, y_hi = 1.8;

      dft::plotting::ContourfData field;
      field.x.resize(nx, std::vector<double>(ny));
      field.y.resize(nx, std::vector<double>(ny));
      field.z.resize(nx, std::vector<double>(ny));

      for (int i = 0; i < nx; ++i) {
        double x = x_lo + (x_hi - x_lo) * i / (nx - 1);
        for (int j = 0; j < ny; ++j) {
          double y = y_lo + (y_hi - y_lo) * j / (ny - 1);
          double r = x * x - 1.0;
          double s = y - x * x;
          field.x[i][j] = x;
          field.y[i][j] = y;
          field.z[i][j] = std::log10(r * r + 10.0 * s * s + 1e-6);
        }
      }

      dft::plotting::ContourfOptions opts{
          .width = 800,
          .height = 600,
          .levels = 40,
          .cmap = "viridis",
          .xlabel = "$x$",
          .ylabel = "$y$",
          .title = "String method: energy landscape and MEP",
          .square_axes = false,
          .colorbar = true,
      };

      // Reference parabola y = x^2, clipped to the plot range.
      double x_clip = std::sqrt(y_hi);  // x beyond this gives y > y_hi.
      dft::plotting::ContourfLine valley;
      valley.x.resize(100);
      valley.y.resize(100);
      for (int i = 0; i < 100; ++i) {
        valley.x[i] = -x_clip + 2.0 * x_clip * i / 99.0;
        valley.y[i] = valley.x[i] * valley.x[i];
      }
      valley.color = "#656775";
      valley.linestyle = ":";
      valley.linewidth = 1.0;
      valley.label = "Valley floor $y = x^2$";

      // Theoretical MEP (steepest descent from saddle).
      dft::plotting::ContourfLine theo_mep{
          .x = theo_mep_x,
          .y = theo_mep_y,
          .color = "black",
          .linestyle = "--",
          .linewidth = 1.5,
          .label = "Theoretical MEP",
      };

      // Simplified string method result.
      dft::plotting::ContourfLine simplified_line{
          .x = path_x_simplified,
          .y = path_y_simplified,
          .color = "#9E744F",
          .linestyle = "--",
          .linewidth = 2.0,
          .marker = "s",
          .markersize = 4.0,
          .label = "Simplified",
      };

      // Full string method result.
      dft::plotting::ContourfLine full_line{
          .x = path_x_full,
          .y = path_y_full,
          .color = "#E25822",
          .linestyle = "-",
          .linewidth = 2.5,
          .marker = "o",
          .markersize = 5.0,
          .label = "Full (perp. force)",
      };

      // Mark minima.
      dft::plotting::ContourfLine minima{
          .x = {-1.0, 1.0},
          .y = {1.0, 1.0},
          .color = "black",
          .linestyle = "None",
          .linewidth = 0.0,
          .marker = "*",
          .markersize = 14.0,
          .label = "Minima",
      };

      // Mark saddle point.
      dft::plotting::ContourfLine saddle{
          .x = {0.0},
          .y = {0.0},
          .color = "black",
          .linestyle = "None",
          .linewidth = 0.0,
          .marker = "D",
          .markersize = 10.0,
          .label = "Saddle point",
      };

      dft::plotting::ContourfPanel panel{
          .field = field,
          .options = opts,
          .lines = {valley, simplified_line, full_line, minima, saddle, theo_mep},
      };

      panel.save("exports/string_method_landscape.pdf");
      std::cout << "Plot saved: exports/string_method_landscape.pdf\n";
    }

    // Plot 2: energy along the path vs arc-length.

    inline void energy_along_path(
        const std::vector<double>& alpha_simplified,
        const std::vector<double>& energy_simplified,
        const std::vector<double>& alpha_full,
        const std::vector<double>& energy_full
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(800, 500);

      plt::plot(
          alpha_simplified, energy_simplified,
          {{"color", "#9E744F"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", "Simplified"}}
      );
      plt::plot(
          alpha_simplified, energy_simplified,
          {{"color", "#9E744F"}, {"marker", "s"}, {"markersize", "3"}, {"linestyle", "None"}}
      );

      plt::plot(
          alpha_full, energy_full,
          {{"color", "#008080"}, {"linewidth", "2.0"}, {"label", "Full (perp. force)"}}
      );
      plt::plot(
          alpha_full, energy_full,
          {{"color", "#008080"}, {"marker", "o"}, {"markersize", "4"}, {"linestyle", "None"}}
      );

      // Reference barrier line.
      plt::plot(
          {alpha_full.front(), alpha_full.back()}, {1.0, 1.0},
          {{"color", "#656775"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"(Saddle $V = 1$)"}}
      );

      plt::xlabel(R"(Arc length $\alpha$)");
      plt::ylabel(R"($V$)");
      plt::title("Energy along the path");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/string_method_energy.pdf");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/string_method_energy.pdf\n";
    }

    // Plot 3: convergence history (error vs iteration).

    inline void convergence(
        const std::vector<double>& iterations,
        const std::vector<double>& errors
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(800, 500);

      plt::named_semilogy("Error", iterations, errors, "o-");

      plt::xlabel("Iteration");
      plt::ylabel(R"($\max |(\nabla V)_\perp|$)");
      plt::title("String method convergence");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/string_method_convergence.pdf");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/string_method_convergence.pdf\n";
    }

  }  // namespace detail

  struct PlotData {
    std::vector<double> path_x_init;
    std::vector<double> path_y_init;
    std::vector<double> path_x_simplified;
    std::vector<double> path_y_simplified;
    std::vector<double> alpha_simplified;
    std::vector<double> energy_simplified;
    std::vector<double> path_x_full;
    std::vector<double> path_y_full;
    std::vector<double> alpha_full;
    std::vector<double> energy_full;
    std::vector<double> iter_history;
    std::vector<double> error_history;
    std::vector<double> theo_mep_x;
    std::vector<double> theo_mep_y;
  };

  inline void make_plots(const PlotData& data) {
    detail::energy_landscape(
        data.path_x_init, data.path_y_init, data.path_x_simplified, data.path_y_simplified, data.path_x_full,
        data.path_y_full, data.theo_mep_x, data.theo_mep_y
    );
    detail::energy_along_path(data.alpha_simplified, data.energy_simplified, data.alpha_full, data.energy_full);
    detail::convergence(data.iter_history, data.error_history);
  }

}  // namespace plot

#endif
