#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
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

  std::cout << "=== Hard-sphere fluid models ===\n\n";
  std::cout << std::fixed << std::setprecision(6);
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "P/(rho kT) CS"
            << std::setw(16) << "P/(rho kT) PYv"
            << std::setw(16) << "P/(rho kT) PYc"
            << "\n";

  hard_spheres::HardSphereModel cs = hard_spheres::CarnahanStarling{};
  hard_spheres::HardSphereModel pyv = hard_spheres::PercusYevickVirial{};
  hard_spheres::HardSphereModel pyc = hard_spheres::PercusYevickCompressibility{};

  for (double eta = 0.05; eta <= 0.49; eta += 0.05) {
    std::cout << std::setw(8) << eta
              << std::setw(16) << hard_spheres::pressure(cs, eta)
              << std::setw(16) << hard_spheres::pressure(pyv, eta)
              << std::setw(16) << hard_spheres::pressure(pyc, eta)
              << "\n";
  }

  // Thermodynamic consistency check.

  std::cout << "\n=== Consistency check (CS, eta=0.3) ===\n";
  double rho = hard_spheres::density_from_eta(0.3);
  double mu = hard_spheres::chemical_potential(cs, rho);
  double f = hard_spheres::free_energy(cs, rho);
  double p = hard_spheres::pressure(cs, 0.3);
  std::cout << "  mu/kT         = " << mu << "\n";
  std::cout << "  f/kT          = " << f << "\n";
  std::cout << "  P/(rho*kT)    = " << p << "\n";
  std::cout << "  mu - f - P    = " << mu - f - p << "  (should be 0)\n";

  // Transport coefficients.

  std::cout << "\n=== Enskog transport coefficients ===\n\n";
  std::cout << std::setw(8) << "rho"
            << std::setw(14) << "shear"
            << std::setw(14) << "bulk"
            << std::setw(14) << "lambda"
            << std::setw(14) << "Gamma"
            << "\n";

  for (double density = 0.1; density <= 0.8; density += 0.1) {
    double chi = hard_spheres::contact_value(hard_spheres::packing_fraction(density));
    std::cout << std::setw(8) << density
              << std::setw(14) << hard_spheres::transport::shear_viscosity(density, chi)
              << std::setw(14) << hard_spheres::transport::bulk_viscosity(density, chi)
              << std::setw(14) << hard_spheres::transport::thermal_conductivity(density, chi)
              << std::setw(14) << hard_spheres::transport::sound_damping(density, chi)
              << "\n";
  }

  // Equations of state (kT = 1.5).

  std::cout << "\n=== Equations of state (kT = 1.5) ===\n\n";
  double kT = 1.5;
  eos::EosModel ideal = eos::IdealGas{.kT = kT};
  eos::EosModel py_eos = eos::PercusYevick{.kT = kT};
  eos::EosModel jzg = eos::make_lennard_jones_jzg(kT);
  eos::EosModel mecke = eos::make_lennard_jones_mecke(kT);

  std::cout << std::setw(8) << "rho"
            << std::setw(14) << eos::name(ideal)
            << std::setw(14) << eos::name(py_eos)
            << std::setw(14) << eos::name(jzg)
            << std::setw(14) << eos::name(mecke)
            << "\n";

  for (double density = 0.05; density <= 0.8; density += 0.05) {
    std::cout << std::setw(8) << density
              << std::setw(14) << eos::pressure(ideal, density)
              << std::setw(14) << eos::pressure(py_eos, density)
              << std::setw(14) << eos::pressure(jzg, density)
              << std::setw(14) << eos::pressure(mecke, density)
              << "\n";
  }

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB

  // Hard-sphere pressure.
  {
    constexpr int M = 100;
    arma::vec eta_a = arma::linspace(0.01, 0.49, M);
    arma::vec cs_a(M), pyv_a(M), pyc_a(M);
    for (arma::uword i = 0; i < eta_a.n_elem; ++i) {
      cs_a(i) = hard_spheres::pressure(cs, eta_a(i));
      pyv_a(i) = hard_spheres::pressure(pyv, eta_a(i));
      pyc_a(i) = hard_spheres::pressure(pyc, eta_a(i));
    }
    auto eta_v = arma::conv_to<std::vector<double>>::from(eta_a);
    auto cs_v = arma::conv_to<std::vector<double>>::from(cs_a);
    auto pyv_v = arma::conv_to<std::vector<double>>::from(pyv_a);
    auto pyc_v = arma::conv_to<std::vector<double>>::from(pyc_a);

    plot::hs_pressure(eta_v, cs_v, pyv_v, pyc_v);
  }

  // Transport coefficients.
  {
    constexpr int M = 80;
    arma::vec rho_a = arma::linspace(0.01, 0.8, M);
    arma::vec sh_a(M), bk_a(M);
    for (arma::uword i = 0; i < rho_a.n_elem; ++i) {
      double chi = hard_spheres::contact_value(hard_spheres::packing_fraction(rho_a(i)));
      sh_a(i) = hard_spheres::transport::shear_viscosity(rho_a(i), chi);
      bk_a(i) = hard_spheres::transport::bulk_viscosity(rho_a(i), chi);
    }
    auto rho_v = arma::conv_to<std::vector<double>>::from(rho_a);
    auto sh = arma::conv_to<std::vector<double>>::from(sh_a);
    auto bk = arma::conv_to<std::vector<double>>::from(bk_a);

    plot::transport_viscosity(rho_v, sh, bk);
  }
#endif
}
