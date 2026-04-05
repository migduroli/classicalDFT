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

  hard_spheres::CarnahanStarling cs{};
  hard_spheres::PercusYevickVirial pyv{};
  hard_spheres::PercusYevickCompressibility pyc{};

  for (double eta = 0.05; eta <= 0.49; eta += 0.05) {
    std::println(
        std::cout,
        "{:>8.2f}{:>16.6f}{:>16.6f}{:>16.6f}",
        eta,
        cs.pressure(eta),
        pyv.pressure(eta),
        pyc.pressure(eta)
    );
  }

  // Thermodynamic consistency check.

  std::println(std::cout, "\n=== Consistency check (CS, eta=0.3) ===");
  double rho = hard_spheres::density_from_eta(0.3);
  double mu = cs.chemical_potential(rho);
  double f = cs.free_energy(rho);
  double p = cs.pressure(0.3);
  std::println(std::cout, "  mu/kT         = {:.6f}", mu);
  std::println(std::cout, "  f/kT          = {:.6f}", f);
  std::println(std::cout, "  P/(rho*kT)    = {:.6f}", p);
  std::println(std::cout, "  mu - f - P    = {:.6f}  (should be 0)", mu - f - p);

  // Transport coefficients.

  std::println(std::cout, "\n=== Enskog transport coefficients ===\n");
  std::println(std::cout, "{:>8s}{:>14s}{:>14s}{:>14s}{:>14s}", "rho", "shear", "bulk", "lambda", "Gamma");

  for (double density = 0.1; density <= 0.8; density += 0.1) {
    double chi = hard_spheres::contact_value(hard_spheres::packing_fraction(density));
    std::println(
        std::cout,
        "{:>8.1f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}",
        density,
        hard_spheres::transport::shear_viscosity(density, chi),
        hard_spheres::transport::bulk_viscosity(density, chi),
        hard_spheres::transport::thermal_conductivity(density, chi),
        hard_spheres::transport::sound_damping(density, chi)
    );
  }

  // Equations of state (kT = 1.5).

  std::println(std::cout, "\n=== Equations of state (kT = 1.5) ===\n");
  double kT = 1.5;
  auto ideal = eos::IdealGas{ .kT = kT };
  auto py_eos = eos::PercusYevick{ .kT = kT };
  auto jzg = eos::make_lennard_jones_jzg(kT);
  auto mecke = eos::make_lennard_jones_mecke(kT);

  std::println(std::cout, "{:>8s}{:>14s}{:>14s}{:>14s}{:>14s}", "rho", ideal.NAME, py_eos.NAME, jzg.NAME, mecke.NAME);

  for (double density = 0.05; density <= 0.8; density += 0.05) {
    std::println(
        std::cout,
        "{:>8.2f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}",
        density,
        ideal.pressure(density),
        py_eos.pressure(density),
        jzg.pressure(density),
        mecke.pressure(density)
    );
  }

  // Collect plot data.

  constexpr int M_hs = 100;
  arma::vec eta_a = arma::linspace(0.01, 0.49, M_hs);
  arma::vec cs_a(M_hs), pyv_a(M_hs), pyc_a(M_hs);
  for (arma::uword i = 0; i < eta_a.n_elem; ++i) {
    cs_a(i) = cs.pressure(eta_a(i));
    pyv_a(i) = pyv.pressure(eta_a(i));
    pyc_a(i) = pyc.pressure(eta_a(i));
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
