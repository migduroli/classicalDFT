#pragma once

#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace detail {

    inline void wca_bh_splitting(
        const std::vector<double>& r,
        const std::vector<double>& v_full,
        const std::vector<double>& watt_wca,
        const std::vector<double>& watt_bh
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 600);
      plt::named_plot(R"($v_\mathrm{LJ}(r)$)", r, v_full, "k-");
      plt::named_plot(R"($w_\mathrm{att}^\mathrm{WCA}(r)$)", r, watt_wca, "b-");
      plt::named_plot(R"($w_\mathrm{att}^\mathrm{BH}(r)$)", r, watt_bh, "r--");
      plt::xlim(0.85, 2.7);
      plt::ylim(-2.0, 5.0);
      plt::xlabel(R"($r / \sigma$)");
      plt::ylabel(R"($w(r) / \epsilon$)");
      plt::title("LJ attractive tail: WCA vs BH splitting");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/interaction_wca_bh.png");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/interaction_wca_bh.png\n";
    }

    inline void mean_field_free_energy(const std::vector<double>& rho, const std::vector<double>& f) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 600);
      plt::named_plot(R"($f_\mathrm{mf}(\rho)$)", rho, f, "b-");
      plt::xlabel(R"($\rho \sigma^3$)");
      plt::ylabel(R"($f_\mathrm{mf} / k_BT$)");
      plt::title("Mean-field free energy density (LJ)");
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/interaction_free_energy.png");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/interaction_free_energy.png\n";
    }

    inline void
    grid_convergence(const std::vector<double>& dx_vals, const std::vector<double>& a_conv, double a_analytic) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 600);
      plt::named_plot(R"($a_\mathrm{vdw}(\Delta x)$)", dx_vals, a_conv, "bo-");
      plt::plot(
          std::vector<double>{ dx_vals.front(), dx_vals.back() },
          std::vector<double>{ a_analytic, a_analytic },
          { { "color", "r" }, { "linestyle", "--" }, { "label", "Analytic" } }
      );
      plt::xlabel(R"($\Delta x / \sigma$)");
      plt::ylabel(R"($a_\mathrm{vdw}$)");
      plt::title("Grid convergence of mean-field weight integral");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      plt::save("exports/interaction_convergence.png");
      plt::clf();
      plt::close();
      std::cout << "Plot saved: exports/interaction_convergence.png\n";
    }

  }  // namespace detail

  inline void make_plots(
      const std::vector<double>& r,
      const std::vector<double>& v_full,
      const std::vector<double>& watt_wca,
      const std::vector<double>& watt_bh,
      const std::vector<double>& rho,
      const std::vector<double>& f,
      const std::vector<double>& dx_vals,
      const std::vector<double>& a_conv,
      double a_analytic
  ) {
    detail::wca_bh_splitting(r, v_full, watt_wca, watt_bh);
    detail::mean_field_free_energy(rho, f);
    detail::grid_convergence(dx_vals, a_conv, a_analytic);
  }

}  // namespace plot

#endif
