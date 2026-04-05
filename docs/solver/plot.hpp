#pragma once

#include "dft/functionals/bulk/phase_diagram.hpp"
#include "utils.hpp"

#include <armadillo>
#include <map>
#include <string>
#include <vector>

using utils::CoexData;
using utils::SpinodalData;

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace detail {

    inline void isotherms(
        const std::vector<std::vector<double>>& iso_rho,
        const std::vector<std::vector<double>>& iso_p,
        const std::vector<double>& temps
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 600);
      std::vector<std::string> styles = { "b-", "c-", "g-", "y-", "r-", "m-" };
      for (std::size_t t = 0; t < temps.size(); ++t) {
        char label[32];
        std::snprintf(label, sizeof(label), R"($T^* = %.1f$)", temps[t]);
        plt::named_plot(label, iso_rho[t], iso_p[t], styles[t % styles.size()]);
      }
      plt::xlim(0.0, 1.0);
      plt::ylim(-0.5, 2.0);
      plt::xlabel(R"($\rho \sigma^3$)");
      plt::ylabel(R"($P^* = P\sigma^3/\epsilon$)");
      plt::title("Pressure isotherms (White Bear II + mean-field)");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/isotherms.png");
      plt::clf();
      plt::close();
      std::cout << "\nPlot saved: exports/isotherms.png\n";
    }

    inline void coexistence(const std::vector<CoexData>& all_coex, const std::vector<SpinodalData>& all_spin = {}) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 650);
      // Solid colors for binodal, same with 66 hex (~40% opacity) suffix for spinodal.
      std::vector<std::string> colors = { "#1f77b4", "#2ca02c", "#d62728", "#000000" };
      std::vector<std::string> colors_alpha = { "#1f77b466", "#2ca02c66", "#d6272866", "#00000066" };

      for (std::size_t m = 0; m < all_coex.size(); ++m) {
        const auto& cd = all_coex[m];
        if (cd.T.is_empty())
          continue;

        const auto& col = colors[m % colors.size()];

        // Binodal dome (solid).
        arma::vec rho_dome = arma::join_cols(cd.rho_v, arma::reverse(cd.rho_l));
        arma::vec T_dome = arma::join_cols(cd.T, arma::reverse(cd.T));
        auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_dome);
        auto T_vec = arma::conv_to<std::vector<double>>::from(T_dome);

        plt::plot(rho_vec, T_vec, { { "color", col }, { "linestyle", "-" }, { "label", cd.name } });

        if (cd.Tc > 0) {
          plt::plot({ cd.rho_c }, { cd.Tc }, { { "color", col }, { "marker", "o" }, { "markersize", "5" } });
        }

        // Spinodal dome (dashed, same color, transparent via RGBA hex).
        if (m < all_spin.size() && !all_spin[m].T.is_empty()) {
          const auto& sd = all_spin[m];
          const auto& col_a = colors_alpha[m % colors_alpha.size()];
          arma::vec rho_sp = arma::join_cols(sd.rho_lo, arma::reverse(sd.rho_hi));
          arma::vec T_sp = arma::join_cols(sd.T, arma::reverse(sd.T));
          auto rho_sp_vec = arma::conv_to<std::vector<double>>::from(rho_sp);
          auto T_sp_vec = arma::conv_to<std::vector<double>>::from(T_sp);

          plt::plot(
              rho_sp_vec,
              T_sp_vec,
              { { "color", col_a }, { "linestyle", "--" }, { "label", sd.name + " (spinodal)" } }
          );
        }
      }
      plt::xlabel(R"($\rho \sigma^3$)");
      plt::ylabel(R"($T^* = k_BT / \epsilon$)");
      plt::title("Liquid-vapor phase diagram (continuation)");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/coexistence.png");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/coexistence.png\n";
    }

    inline void binodal(const CoexData& cd) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 650);
      if (!cd.T.is_empty()) {
        arma::vec rho_dome = arma::join_cols(cd.rho_v, arma::reverse(cd.rho_l));
        arma::vec T_dome = arma::join_cols(cd.T, arma::reverse(cd.T));
        auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_dome);
        auto T_vec = arma::conv_to<std::vector<double>>::from(T_dome);
        plt::named_plot("Binodal", rho_vec, T_vec, "k-o");
      }
      if (cd.Tc > 0) {
        plt::plot({ cd.rho_c }, { cd.Tc }, "rs");
      }
      plt::xlabel(R"($\rho \sigma^3$)");
      plt::ylabel(R"($T^* = k_BT / \epsilon$)");
      plt::title("Binodal: " + cd.name + " (continuation)");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/binodal.png");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/binodal.png\n";
    }

    inline void phase_diagram_plot(
        const CoexData& cd,
        const dft::functionals::bulk::SpinodalCurve& sp,
        const utils::JimCoexPoints& jim = {},
        const utils::JimSpinodalPoints& jim_sp = {}
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 650);

      // Binodal dome (continuation).
      if (!cd.T.is_empty()) {
        arma::vec rho_dome = arma::join_cols(cd.rho_v, arma::reverse(cd.rho_l));
        arma::vec T_dome = arma::join_cols(cd.T, arma::reverse(cd.T));
        auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_dome);
        auto T_vec = arma::conv_to<std::vector<double>>::from(T_dome);
        plt::plot(
            rho_vec,
            T_vec,
            { { "color", "k" }, { "linestyle", "-" }, { "linewidth", "1.5" }, { "label", "Binodal (continuation)" } }
        );
      }

      // Spinodal dome (continuation).
      if (!sp.temperature.is_empty()) {
        arma::vec rho_sp = arma::join_cols(sp.rho_low, arma::reverse(sp.rho_high));
        arma::vec T_sp = arma::join_cols(sp.temperature, arma::reverse(sp.temperature));
        auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_sp);
        auto T_vec = arma::conv_to<std::vector<double>>::from(T_sp);
        plt::plot(
            rho_vec,
            T_vec,
            { { "color", "#d62728" },
              { "linestyle", "--" },
              { "linewidth", "1.5" },
              { "label", "Spinodal (T-stepping)" } }
        );
      }

      // Jim's single-temperature coexistence points.
      if (!jim.T.empty()) {
        plt::plot(
            jim.rho_v,
            jim.T,
            { { "color", "#2ca02c" },
              { "marker", "o" },
              { "linestyle", "none" },
              { "markersize", "4" },
              { "label", "Coex (scan+bisect)" } }
        );
        plt::plot(
            jim.rho_l,
            jim.T,
            { { "color", "#2ca02c" }, { "marker", "o" }, { "linestyle", "none" }, { "markersize", "4" } }
        );
      }

      // Jim's single-temperature spinodal points.
      if (!jim_sp.T.empty()) {
        plt::plot(
            jim_sp.rho_lo,
            jim_sp.T,
            { { "color", "#ff7f0e" },
              { "marker", "x" },
              { "linestyle", "none" },
              { "markersize", "4" },
              { "label", "Spinodal (scan+bisect)" } }
        );
        plt::plot(
            jim_sp.rho_hi,
            jim_sp.T,
            { { "color", "#ff7f0e" }, { "marker", "x" }, { "linestyle", "none" }, { "markersize", "4" } }
        );
      }

      if (cd.Tc > 0) {
        plt::plot({ cd.rho_c }, { cd.Tc }, { { "color", "k" }, { "marker", "s" }, { "markersize", "6" } });
      }

      plt::xlabel(R"($\rho \sigma^3$)");
      plt::ylabel(R"($T^* = k_BT / \epsilon$)");
      plt::title("Phase diagram: " + cd.name);
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/phase_diagram.png");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/phase_diagram.png\n";
    }

  }  // namespace detail

  inline void make_plots(
      const std::vector<std::vector<double>>& iso_rho,
      const std::vector<std::vector<double>>& iso_p,
      const std::vector<double>& isotherm_temps,
      const std::vector<CoexData>& all_coex,
      const std::vector<SpinodalData>& all_spin,
      const CoexData& wb2_data,
      const dft::functionals::bulk::SpinodalCurve& wb2_sp,
      const utils::JimCoexPoints& jim_pts = {},
      const utils::JimSpinodalPoints& jim_sp = {}
  ) {
    detail::isotherms(iso_rho, iso_p, isotherm_temps);
    detail::coexistence(all_coex, all_spin);
    detail::binodal(wb2_data);
    if (!wb2_sp.temperature.is_empty()) {
      detail::phase_diagram_plot(wb2_data, wb2_sp, jim_pts, jim_sp);
    }
  }

}  // namespace plot

#endif
