#pragma once

#include <string>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

  namespace detail {

  inline void convergence(
      const std::vector<double>& iterations,
      const std::vector<double>& residuals
  ) {
    namespace plt = matplotlibcpp;
    plt::figure_size(800, 500);

    plt::named_semilogy("Residual", iterations, residuals, "o-");

    plt::xlabel("Iteration");
    plt::ylabel("Residual");
    plt::title("Picard iteration convergence");
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/picard_convergence.png");
    plt::clf();
    plt::close();
    std::cout << "Plot saved: exports/picard_convergence.png\n";
  }

  inline void density_profile(
      const std::vector<double>& x,
      const std::vector<double>& rho_initial,
      const std::vector<double>& rho_converged,
      double rho_bulk
  ) {
    namespace plt = matplotlibcpp;
    plt::figure_size(800, 500);

    plt::plot(x, rho_initial,
              {{"color", "#9E744F"}, {"linewidth", "1.5"}, {"linestyle", "--"}, {"label", R"($\rho_0$ initial)"}});
    plt::plot(x, rho_converged,
              {{"color", "#008080"}, {"linewidth", "2.0"}, {"label", R"($\rho^*$ converged)"}});

    double x_lo = x.front(), x_hi = x.back();
    plt::plot({x_lo, x_hi}, {rho_bulk, rho_bulk},
              {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_{\mathrm{bulk}}$)"}});

    plt::xlabel(R"($x / \sigma$)");
    plt::ylabel(R"($\rho \sigma^3$)");
    plt::title("Picard iteration: density equilibration");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/picard_density.png");
    plt::clf();
    plt::close();
    std::cout << "Plot saved: exports/picard_density.png\n";
  }

  }  // namespace detail

  inline void make_plots(
      const std::vector<double>& x,
      const std::vector<double>& rho_initial,
      const std::vector<double>& rho_converged,
      double rho_bulk
  ) {
    detail::density_profile(x, rho_initial, rho_converged, rho_bulk);
  }

}  // namespace plot

#endif
