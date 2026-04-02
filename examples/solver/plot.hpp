#pragma once

#include "dft/functionals/bulk/phase_diagram.hpp"

#include <armadillo>
#include <map>
#include <string>
#include <vector>

struct CoexData {
  std::string name;
  arma::vec T, rho_v, rho_l;
  double Tc{0.0};
  double rho_c{0.0};
};

struct SpinodalData {
  std::string name;
  arma::vec T, rho_lo, rho_hi;
};

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

namespace plot {

inline void isotherms(
    const std::vector<std::vector<double>>& iso_rho,
    const std::vector<std::vector<double>>& iso_p,
    const std::vector<double>& temps
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 600);
  std::vector<std::string> styles = {"b-", "c-", "g-", "y-", "r-", "m-"};
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
  plt::close();
  std::cout << "\nPlot saved: exports/isotherms.png\n";
}

inline void coexistence(const std::vector<CoexData>& all_coex) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 650);
  std::vector<std::string> colors = {"b", "g", "r", "k"};
  std::vector<std::string> styles = {"b-", "g--", "r-.", "k-"};

  for (std::size_t m = 0; m < all_coex.size(); ++m) {
    const auto& cd = all_coex[m];
    if (cd.T.is_empty()) continue;

    // Combine vapor + reversed liquid to draw a single dome.
    arma::vec rho_dome = arma::join_cols(cd.rho_v, arma::reverse(cd.rho_l));
    arma::vec T_dome = arma::join_cols(cd.T, arma::reverse(cd.T));
    auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_dome);
    auto T_vec = arma::conv_to<std::vector<double>>::from(T_dome);

    plt::named_plot(cd.name, rho_vec, T_vec, styles[m % styles.size()]);

    if (cd.Tc > 0) {
      plt::plot({cd.rho_c}, {cd.Tc}, colors[m % colors.size()] + "o");
    }
  }
  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel(R"($T^* = k_BT / \epsilon$)");
  plt::title("Liquid-vapor coexistence (continuation)");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/coexistence.png");
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
    plt::plot({cd.rho_c}, {cd.Tc}, "rs");
  }
  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel(R"($T^* = k_BT / \epsilon$)");
  plt::title("Binodal: " + cd.name + " (continuation)");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/binodal.png");
  plt::close();
  std::cout << "Plot saved: exports/binodal.png\n";
}

inline void phase_diagram_plot(
    const CoexData& cd,
    const dft::functionals::bulk::SpinodalCurve& sp
) {
  namespace plt = matplotlibcpp;
  plt::figure_size(900, 650);

  // Binodal dome.
  if (!cd.T.is_empty()) {
    arma::vec rho_dome = arma::join_cols(cd.rho_v, arma::reverse(cd.rho_l));
    arma::vec T_dome = arma::join_cols(cd.T, arma::reverse(cd.T));
    auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_dome);
    auto T_vec = arma::conv_to<std::vector<double>>::from(T_dome);
    plt::named_plot("Binodal", rho_vec, T_vec, "k-");
  }

  // Spinodal dome.
  if (!sp.temperature.is_empty()) {
    arma::vec rho_sp = arma::join_cols(sp.rho_low, arma::reverse(sp.rho_high));
    arma::vec T_sp = arma::join_cols(sp.temperature, arma::reverse(sp.temperature));
    auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_sp);
    auto T_vec = arma::conv_to<std::vector<double>>::from(T_sp);
    plt::named_plot("Spinodal", rho_vec, T_vec, "r--");
  }

  if (cd.Tc > 0) {
    plt::plot({cd.rho_c}, {cd.Tc}, "ks");
  }

  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel(R"($T^* = k_BT / \epsilon$)");
  plt::title("Phase diagram: " + cd.name);
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/phase_diagram.png");
  plt::close();
  std::cout << "Plot saved: exports/phase_diagram.png\n";
}

}  // namespace plot

#endif
