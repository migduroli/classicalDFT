#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
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

  fmt::FMTModel ros = fmt::Rosenfeld{};
  fmt::FMTModel rslt = fmt::RSLT{};
  fmt::FMTModel wb1 = fmt::WhiteBearI{};
  fmt::FMTModel wb2 = fmt::WhiteBearII{};

  physics::hard_spheres::HardSphereModel cs = physics::hard_spheres::CarnahanStarling{};
  physics::hard_spheres::HardSphereModel pyc = physics::hard_spheres::PercusYevickCompressibility{};

  // Bulk excess free energy per particle: FMT vs EOS.

  std::cout << "=== Bulk excess free energy per particle: FMT vs EOS ===\n\n";
  std::cout << std::fixed << std::setprecision(8);
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "Rosenfeld"
            << std::setw(16) << "PY(comp)"
            << std::setw(16) << "WhiteBearI"
            << std::setw(16) << "CS"
            << "\n";

  constexpr int N = 200;
  arma::vec eta_arma = arma::linspace(0.005, 0.495, N);
  arma::vec f_ros_a(N), f_wb1_a(N), f_wb2_a(N);
  std::vector<Species> sp_list = {Species{.name = "HS", .hard_sphere_diameter = 1.0}};

  for (arma::uword i = 0; i < eta_arma.n_elem; ++i) {
    double eta = eta_arma(i);
    double rho = physics::hard_spheres::density_from_eta(eta);

    auto m = fmt::make_uniform_measures(rho, 1.0);
    m.products = fmt::inner_products(m);
    f_ros_a(i) = fmt::phi(ros, m) / rho;
    f_wb1_a(i) = fmt::phi(wb1, m) / rho;
    f_wb2_a(i) = fmt::phi(wb2, m) / rho;

    if (i % 50 == 0) {
      std::cout << std::setw(8) << eta
                << std::setw(16) << f_ros_a(i)
                << std::setw(16) << physics::hard_spheres::PercusYevickCompressibility::excess_free_energy(eta)
                << std::setw(16) << f_wb1_a(i)
                << std::setw(16) << physics::hard_spheres::CarnahanStarling::excess_free_energy(eta)
                << "\n";
    }
  }
  auto eta_vec = arma::conv_to<std::vector<double>>::from(eta_arma);
  auto f_ros = arma::conv_to<std::vector<double>>::from(f_ros_a);
  auto f_wb1 = arma::conv_to<std::vector<double>>::from(f_wb1_a);
  auto f_wb2 = arma::conv_to<std::vector<double>>::from(f_wb2_a);

  // Chemical potentials.

  std::cout << "\n=== Bulk excess chemical potential ===\n\n";
  arma::vec mu_ros_a(N), mu_wb1_a(N), mu_wb2_a(N);
  for (arma::uword i = 0; i < eta_arma.n_elem; ++i) {
    double rho = physics::hard_spheres::density_from_eta(eta_arma(i));
    mu_ros_a(i) = bulk::hard_sphere_excess_chemical_potential(ros, arma::vec{rho}, sp_list, 0);
    mu_wb1_a(i) = bulk::hard_sphere_excess_chemical_potential(wb1, arma::vec{rho}, sp_list, 0);
    mu_wb2_a(i) = bulk::hard_sphere_excess_chemical_potential(wb2, arma::vec{rho}, sp_list, 0);
  }
  auto mu_ros = arma::conv_to<std::vector<double>>::from(mu_ros_a);
  auto mu_wb1 = arma::conv_to<std::vector<double>>::from(mu_wb1_a);
  auto mu_wb2 = arma::conv_to<std::vector<double>>::from(mu_wb2_a);

  // Pressure consistency: P/(rho kT) = 1 + rho*(mu_ex - f_ex).

  std::cout << "\n=== Pressure: P/(rho kT) from mu - f ===\n\n";
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "PY(comp)"
            << std::setw(16) << "Rosenfeld"
            << std::setw(16) << "CS"
            << std::setw(16) << "WhiteBearI"
            << "\n";

  arma::vec p_ros_a(N), p_wb1_a(N), p_cs_a(N), p_pyc_a(N);
  for (arma::uword i = 0; i < eta_arma.n_elem; ++i) {
    double rho = physics::hard_spheres::density_from_eta(eta_arma(i));
    p_ros_a(i) = 1.0 + rho * (mu_ros_a(i) - f_ros_a(i));
    p_wb1_a(i) = 1.0 + rho * (mu_wb1_a(i) - f_wb1_a(i));
    p_cs_a(i) = physics::hard_spheres::pressure(cs, eta_arma(i));
    p_pyc_a(i) = physics::hard_spheres::pressure(pyc, eta_arma(i));

    if (i % 50 == 0) {
      std::cout << std::setw(8) << eta_arma(i)
                << std::setw(16) << p_pyc_a(i)
                << std::setw(16) << p_ros_a(i)
                << std::setw(16) << p_cs_a(i)
                << std::setw(16) << p_wb1_a(i)
                << "\n";
    }
  }
  auto p_ros = arma::conv_to<std::vector<double>>::from(p_ros_a);
  auto p_wb1 = arma::conv_to<std::vector<double>>::from(p_wb1_a);
  auto p_cs_v = arma::conv_to<std::vector<double>>::from(p_cs_a);
  auto p_pyc_v = arma::conv_to<std::vector<double>>::from(p_pyc_a);

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  plot::free_energy(eta_vec, f_ros, f_wb1, f_wb2);
  plot::pressure(eta_vec, p_ros, p_pyc_v, p_wb1, p_cs_v);
  plot::chemical_potential(eta_vec, mu_ros, mu_wb1, mu_wb2);
#endif
}
