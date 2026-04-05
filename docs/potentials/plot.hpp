#pragma once

#include <filesystem>
#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace detail {

    inline void save(const std::string& path) {
      matplotlibcpp::save(path);
      matplotlibcpp::close();
      std::cout << "Plot saved: " << path << "\n";
    }

    inline void potentials_comparison(
        const std::vector<double>& r,
        const std::vector<double>& v_lj,
        const std::vector<double>& v_twf,
        const std::vector<double>& v_wrdf
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 600);
      plt::named_plot("Lennard-Jones", r, v_lj, "k-");
      plt::named_plot("ten Wolde-Frenkel", r, v_twf, "b-");
      plt::named_plot("WRDF", r, v_wrdf, "r-");
      plt::xlim(0.75, 2.6);
      plt::ylim(-2.0, 10.0);
      plt::xlabel(R"($r / \sigma$)");
      plt::ylabel(R"($v(r) / \epsilon$)");
      plt::title("Intermolecular potentials comparison");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      save("exports/potentials_comparison.png");
    }

    inline void perturbation_decomposition(
        const std::vector<double>& r,
        const std::vector<double>& v_lj,
        const std::vector<double>& att,
        const std::vector<double>& rep
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 600);
      plt::named_plot(R"($v_\mathrm{LJ}(r)$)", r, v_lj, "k-");
      plt::named_plot(R"($w_\mathrm{att}(r)$)", r, att, "b-");
      plt::named_plot(R"($w_\mathrm{rep}(r)$)", r, rep, "r--");
      plt::xlim(0.75, 2.6);
      plt::ylim(-2.0, 10.0);
      plt::xlabel(R"($r / \sigma$)");
      plt::ylabel(R"($w(r) / \epsilon$)");
      plt::title("LJ: WCA perturbation decomposition");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      save("exports/perturbation_lj.png");
    }

    inline void potential_with_dhs(
        const std::vector<double>& r,
        const std::vector<double>& v_lj,
        double r_min,
        double v_min,
        double d_hs
    ) {
      namespace plt = matplotlibcpp;
      plt::figure_size(900, 600);
      plt::named_plot(R"($v_\mathrm{LJ}(r)$)", r, v_lj, "k-");
      plt::named_plot("Minimum", std::vector<double>{ r_min }, std::vector<double>{ v_min }, "rs");
      plt::plot(
          std::vector<double>{ d_hs, d_hs },
          std::vector<double>{ -2.0, 10.0 },
          { { "color", "grey" }, { "linestyle", "--" }, { "label", R"($d_\mathrm{HS}$)" } }
      );
      plt::xlim(0.75, 2.6);
      plt::ylim(-2.0, 10.0);
      plt::xlabel(R"($r / \sigma$)");
      plt::ylabel(R"($v(r) / \epsilon$)");
      plt::title("Lennard-Jones potential");
      plt::legend();
      plt::grid(true);
      plt::tight_layout();
      save("exports/potential_lj.png");
    }

  }  // namespace detail

  inline void make_plots(
      const std::vector<double>& r,
      const std::vector<double>& v_lj,
      const std::vector<double>& v_twf,
      const std::vector<double>& v_wrdf,
      const std::vector<double>& att_lj,
      const std::vector<double>& rep_lj,
      double r_min,
      double v_min,
      double d_hs
  ) {
    detail::potentials_comparison(r, v_lj, v_twf, v_wrdf);
    detail::perturbation_decomposition(r, v_lj, att_lj, rep_lj);
    detail::potential_with_dhs(r, v_lj, r_min, v_min, d_hs);
  }

}  // namespace plot

#endif
