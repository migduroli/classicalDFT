#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iostream>
#include <print>
#include <vector>

using namespace dft;
using namespace physics;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  // Hard-sphere compressibility factor: CS vs PY.

  std::println(std::cout, "=== Hard-sphere fluid models ===\n");
  std::println(std::cout, "{:>8s}{:>16s}{:>16s}{:>16s}", "eta", "P/(rho kT) CS", "P/(rho kT) PYv", "P/(rho kT) PYc");

  hard_spheres::HardSphereModel cs = hard_spheres::CarnahanStarling{};
  hard_spheres::HardSphereModel pyv = hard_spheres::PercusYevickVirial{};
  hard_spheres::HardSphereModel pyc = hard_spheres::PercusYevickCompressibility{};

  for (double eta = 0.05; eta <= 0.49; eta += 0.05) {
    std::println(std::cout, "{:>8.2f}{:>16.6f}{:>16.6f}{:>16.6f}",
                 eta, hard_spheres::pressure(cs, eta), hard_spheres::pressure(pyv, eta), hard_spheres::pressure(pyc, eta));
  }

  // Thermodynamic consistency check.

  std::println(std::cout, "\n=== Consistency check (CS, eta=0.3) ===");
  double rho = hard_spheres::density_from_eta(0.3);
  double mu = hard_spheres::chemical_potential(cs, rho);
  double f = hard_spheres::free_energy(cs, rho);
  double p = hard_spheres::pressure(cs, 0.3);
  std::println(std::cout, "  mu/kT         = {:.6f}", mu);
  std::println(std::cout, "  f/kT          = {:.6f}", f);
  std::println(std::cout, "  P/(rho*kT)    = {:.6f}", p);
  std::println(std::cout, "  mu - f - P    = {:.6f}  (should be 0)", mu - f - p);

  // Transport coefficients.

  std::println(std::cout, "\n=== Enskog transport coefficients ===\n");
  std::println(std::cout, "{:>8s}{:>14s}{:>14s}{:>14s}{:>14s}", "rho", "shear", "bulk", "lambda", "Gamma");

  for (double density = 0.1; density <= 0.8; density += 0.1) {
    double chi = hard_spheres::contact_value(hard_spheres::packing_fraction(density));
    std::println(std::cout, "{:>8.1f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}",
                 density,
                 hard_spheres::transport::shear_viscosity(density, chi),
                 hard_spheres::transport::bulk_viscosity(density, chi),
                 hard_spheres::transport::thermal_conductivity(density, chi),
                 hard_spheres::transport::sound_damping(density, chi));
  }

  // Equations of state (kT = 1.5).

  std::println(std::cout, "\n=== Equations of state (kT = 1.5) ===\n");
  double kT = 1.5;
  eos::EosModel ideal = eos::IdealGas{.kT = kT};
  eos::EosModel py_eos = eos::PercusYevick{.kT = kT};
  eos::EosModel jzg = eos::make_lennard_jones_jzg(kT);
  eos::EosModel mecke = eos::make_lennard_jones_mecke(kT);

  std::println(std::cout, "{:>8s}{:>14s}{:>14s}{:>14s}{:>14s}",
               "rho", eos::name(ideal), eos::name(py_eos), eos::name(jzg), eos::name(mecke));

  for (double density = 0.05; density <= 0.8; density += 0.05) {
    std::println(std::cout, "{:>8.2f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}",
                 density,
                 eos::pressure(ideal, density),
                 eos::pressure(py_eos, density),
                 eos::pressure(jzg, density),
                 eos::pressure(mecke, density));
  }

  // Collect plot data.

  constexpr int M_hs = 100;
  arma::vec eta_a = arma::linspace(0.01, 0.49, M_hs);
  arma::vec cs_a(M_hs), pyv_a(M_hs), pyc_a(M_hs);
  for (arma::uword i = 0; i < eta_a.n_elem; ++i) {
    cs_a(i) = hard_spheres::pressure(cs, eta_a(i));
    pyv_a(i) = hard_spheres::pressure(pyv, eta_a(i));
    pyc_a(i) = hard_spheres::pressure(pyc, eta_a(i));
  }
  auto eta_v = arma::conv_to<std::vector<double>>::from(eta_a);
  auto cs_v = arma::conv_to<std::vector<double>>::from(cs_a);
  auto pyv_v = arma::conv_to<std::vector<double>>::from(pyv_a);
  auto pyc_v = arma::conv_to<std::vector<double>>::from(pyc_a);

  constexpr int M_tr = 80;
  arma::vec rho_a = arma::linspace(0.01, 0.8, M_tr);
  arma::vec sh_a(M_tr), bk_a(M_tr);
  for (arma::uword i = 0; i < rho_a.n_elem; ++i) {
    double chi = hard_spheres::contact_value(hard_spheres::packing_fraction(rho_a(i)));
    sh_a(i) = hard_spheres::transport::shear_viscosity(rho_a(i), chi);
    bk_a(i) = hard_spheres::transport::bulk_viscosity(rho_a(i), chi);
  }
  auto rho_v = arma::conv_to<std::vector<double>>::from(rho_a);
  auto sh = arma::conv_to<std::vector<double>>::from(sh_a);
  auto bk = arma::conv_to<std::vector<double>>::from(bk_a);

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(eta_v, cs_v, pyv_v, pyc_v, rho_v, sh, bk);
#endif
}
