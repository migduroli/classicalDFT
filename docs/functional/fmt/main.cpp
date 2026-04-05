#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iostream>
#include <print>
#include <vector>

using namespace dft;
using namespace functionals;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  // FMT models and reference EOS models.

  fmt::Rosenfeld ros{};
  fmt::RSLT rslt{};
  fmt::WhiteBearI wb1{};
  fmt::WhiteBearII wb2{};

  physics::hard_spheres::CarnahanStarling cs{};
  physics::hard_spheres::PercusYevickCompressibility pyc{};

  // Bulk excess free energy per particle: FMT vs EOS.

  std::println(std::cout, "=== Bulk excess free energy per particle: FMT vs EOS ===\n");
  std::println(std::cout, "{:>8s}{:>16s}{:>16s}{:>16s}{:>16s}", "eta", "Rosenfeld", "PY(comp)", "WhiteBearI", "CS");

  constexpr int N = 200;
  arma::vec eta_arma = arma::linspace(0.005, 0.495, N);
  arma::vec f_ros_a(N), f_wb1_a(N), f_wb2_a(N);
  std::vector<Species> sp_list = {Species{.name = "HS", .hard_sphere_diameter = 1.0}};

  for (arma::uword i = 0; i < eta_arma.n_elem; ++i) {
    double eta = eta_arma(i);
    double rho = physics::hard_spheres::density_from_eta(eta);

    auto m = fmt::make_uniform_measures(rho, 1.0);
    m.products = m.inner_products();
    f_ros_a(i) = ros.phi(m) / rho;
    f_wb1_a(i) = wb1.phi(m) / rho;
    f_wb2_a(i) = wb2.phi(m) / rho;

    if (i % 50 == 0) {
      std::println(std::cout, "{:>8.3f}{:>16.8f}{:>16.8f}{:>16.8f}{:>16.8f}",
                   eta, f_ros_a(i),
                   physics::hard_spheres::PercusYevickCompressibility::excess_free_energy(eta),
                   f_wb1_a(i),
                   physics::hard_spheres::CarnahanStarling::excess_free_energy(eta));
    }
  }
  auto eta_vec = arma::conv_to<std::vector<double>>::from(eta_arma);
  auto f_ros = arma::conv_to<std::vector<double>>::from(f_ros_a);
  auto f_wb1 = arma::conv_to<std::vector<double>>::from(f_wb1_a);
  auto f_wb2 = arma::conv_to<std::vector<double>>::from(f_wb2_a);

  // Chemical potentials.

  std::println(std::cout, "\n=== Bulk excess chemical potential ===\n");
  arma::vec mu_ros_a(N), mu_wb1_a(N), mu_wb2_a(N);
  for (arma::uword i = 0; i < eta_arma.n_elem; ++i) {
    double rho = physics::hard_spheres::density_from_eta(eta_arma(i));
    mu_ros_a(i) = bulk::hard_sphere::excess_chemical_potential(ros, arma::vec{rho}, sp_list, 0);
    mu_wb1_a(i) = bulk::hard_sphere::excess_chemical_potential(wb1, arma::vec{rho}, sp_list, 0);
    mu_wb2_a(i) = bulk::hard_sphere::excess_chemical_potential(wb2, arma::vec{rho}, sp_list, 0);
  }
  auto mu_ros = arma::conv_to<std::vector<double>>::from(mu_ros_a);
  auto mu_wb1 = arma::conv_to<std::vector<double>>::from(mu_wb1_a);
  auto mu_wb2 = arma::conv_to<std::vector<double>>::from(mu_wb2_a);

  // Pressure consistency: P/(rho kT) = 1 + rho*(mu_ex - f_ex).

  std::println(std::cout, "\n=== Pressure: P/(rho kT) from mu - f ===\n");
  std::println(std::cout, "{:>8s}{:>16s}{:>16s}{:>16s}{:>16s}", "eta", "PY(comp)", "Rosenfeld", "CS", "WhiteBearI");

  arma::vec p_ros_a(N), p_wb1_a(N), p_cs_a(N), p_pyc_a(N);
  for (arma::uword i = 0; i < eta_arma.n_elem; ++i) {
    double rho = physics::hard_spheres::density_from_eta(eta_arma(i));
    p_ros_a(i) = 1.0 + rho * (mu_ros_a(i) - f_ros_a(i));
    p_wb1_a(i) = 1.0 + rho * (mu_wb1_a(i) - f_wb1_a(i));
    p_cs_a(i) = cs.pressure(eta_arma(i));
    p_pyc_a(i) = pyc.pressure(eta_arma(i));

    if (i % 50 == 0) {
      std::println(std::cout, "{:>8.3f}{:>16.8f}{:>16.8f}{:>16.8f}{:>16.8f}",
                   eta_arma(i), p_pyc_a(i), p_ros_a(i), p_cs_a(i), p_wb1_a(i));
    }
  }
  auto p_ros = arma::conv_to<std::vector<double>>::from(p_ros_a);
  auto p_wb1 = arma::conv_to<std::vector<double>>::from(p_wb1_a);
  auto p_cs_v = arma::conv_to<std::vector<double>>::from(p_cs_a);
  auto p_pyc_v = arma::conv_to<std::vector<double>>::from(p_pyc_a);

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(eta_vec, f_ros, f_wb1, f_wb2,
                   p_ros, p_pyc_v, p_wb1, p_cs_v,
                   mu_ros, mu_wb1, mu_wb2);
#endif
}
