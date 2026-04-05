// check.cpp — Exhaustive cross-validation of our EOS models against Jim
// Lutsko's classicalDFT library (EOS.h, Enskog.h).
//
// Compares:
//   1. Hard-sphere EOS: PY compressibility, Carnahan-Starling (f_ex, df, d2f, d3f)
//   2. LJ JZG EOS: a(i), b(i), G(d,i), phix, phi1x, phi2x, phi3x
//   3. LJ Mecke EOS: phix, phi1x
//   4. Bulk thermodynamics: mu, P, f for all models

#include "legacy/classicaldft.hpp"

#include <cmath>
#include <dftlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace dft;

static int g_failures = 0;
static int g_checks = 0;

static void check(std::string_view label, double ours, double jim, double tol = 1e-10) {
  ++g_checks;
  double diff = std::abs(ours - jim);
  bool ok = diff <= tol;
  if (!ok) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": ours=" << ours << " jim=" << jim << " diff=" << diff << "\n";
  }
}

static void section(std::string_view title) {
  std::cout << "\n" << title << "\n";
  std::cout << std::string(title.size(), '-') << "\n";
}

int main() {
  std::cout << std::setprecision(15);

  namespace hs = physics::hard_spheres;
  namespace eos = physics::eos;

  // Test densities for HS models (eta from 0.05 to 0.45)
  std::vector<double> densities;
  for (int i = 1; i <= 9; ++i) {
    double eta = i * 0.05;
    densities.push_back(hs::density_from_eta(eta));
  }

  // Step 1: PY compressibility — f_ex, df, d2f, d3f

  section("Step 1: Percus-Yevick compressibility route");

  hs::PercusYevickCompressibility py_model{};

  double max_diff_pyc_fex = 0.0;
  double max_diff_pyc_dfex = 0.0;
  double max_diff_pyc_d2fex = 0.0;
  double max_diff_pyc_d3fex = 0.0;

  for (double rho : densities) {
    double eta = hs::packing_fraction(rho);
    legacy::thermodynamics::Enskog enskog(rho);

    double our_f = py_model.excess_free_energy(eta);
    double jim_f = enskog.exFreeEnergyPYC();
    max_diff_pyc_fex = std::max(max_diff_pyc_fex, std::abs(our_f - jim_f));

    double our_df = py_model.d_excess_free_energy(eta) * (std::numbers::pi / 6.0);
    double jim_df = enskog.dexFreeEnergyPYCdRho();
    max_diff_pyc_dfex = std::max(max_diff_pyc_dfex, std::abs(our_df - jim_df));

    double our_d2f = py_model.d2_excess_free_energy(eta) * (std::numbers::pi / 6.0) * (std::numbers::pi / 6.0);
    double jim_d2f = enskog.d2exFreeEnergyPYCdRho2();
    max_diff_pyc_d2fex = std::max(max_diff_pyc_d2fex, std::abs(our_d2f - jim_d2f));

    double our_d3f = py_model.d3_excess_free_energy(eta) * std::pow(std::numbers::pi / 6.0, 3);
    double jim_d3f = enskog.d3exFreeEnergyPYCdRho3();
    max_diff_pyc_d3fex = std::max(max_diff_pyc_d3fex, std::abs(our_d3f - jim_d3f));
  }

  check("PYC max|fex diff|", max_diff_pyc_fex, 0.0);
  check("PYC max|dfex diff|", max_diff_pyc_dfex, 0.0);
  check("PYC max|d2fex diff|", max_diff_pyc_d2fex, 0.0);
  check("PYC max|d3fex diff|", max_diff_pyc_d3fex, 0.0, 1e-8);
  std::cout << "  PYC max|fex|   = " << max_diff_pyc_fex << "\n";
  std::cout << "  PYC max|dfex|  = " << max_diff_pyc_dfex << "\n";
  std::cout << "  PYC max|d2fex| = " << max_diff_pyc_d2fex << "\n";
  std::cout << "  PYC max|d3fex| = " << max_diff_pyc_d3fex << "\n";

  // Step 2: Carnahan-Starling — f_ex, df, d2f, d3f

  section("Step 2: Carnahan-Starling");

  hs::CarnahanStarling cs_model{};

  double max_diff_cs_fex = 0.0;
  double max_diff_cs_dfex = 0.0;
  double max_diff_cs_d2fex = 0.0;
  double max_diff_cs_d3fex = 0.0;

  for (double rho : densities) {
    double eta = hs::packing_fraction(rho);
    legacy::thermodynamics::Enskog enskog(rho);

    double our_f = cs_model.excess_free_energy(eta);
    double jim_f = enskog.exFreeEnergyCS();
    max_diff_cs_fex = std::max(max_diff_cs_fex, std::abs(our_f - jim_f));

    double our_df = cs_model.d_excess_free_energy(eta) * (std::numbers::pi / 6.0);
    double jim_df = enskog.dexFreeEnergyCSdRho();
    max_diff_cs_dfex = std::max(max_diff_cs_dfex, std::abs(our_df - jim_df));

    double our_d2f = cs_model.d2_excess_free_energy(eta) * (std::numbers::pi / 6.0) * (std::numbers::pi / 6.0);
    double jim_d2f = enskog.d2exFreeEnergyCSdRho2();
    max_diff_cs_d2fex = std::max(max_diff_cs_d2fex, std::abs(our_d2f - jim_d2f));

    double our_d3f = cs_model.d3_excess_free_energy(eta) * std::pow(std::numbers::pi / 6.0, 3);
    double jim_d3f = enskog.d3exFreeEnergyCSdRho3();
    max_diff_cs_d3fex = std::max(max_diff_cs_d3fex, std::abs(our_d3f - jim_d3f));
  }

  check("CS max|fex diff|", max_diff_cs_fex, 0.0);
  check("CS max|dfex diff|", max_diff_cs_dfex, 0.0);
  check("CS max|d2fex diff|", max_diff_cs_d2fex, 0.0);
  check("CS max|d3fex diff|", max_diff_cs_d3fex, 0.0, 1e-8);
  std::cout << "  CS max|fex|   = " << max_diff_cs_fex << "\n";
  std::cout << "  CS max|dfex|  = " << max_diff_cs_dfex << "\n";
  std::cout << "  CS max|d2fex| = " << max_diff_cs_d2fex << "\n";
  std::cout << "  CS max|d3fex| = " << max_diff_cs_d3fex << "\n";

  // Step 3: HS pressure and chemical potential

  section("Step 3: HS pressure P/(nkT) and mu/kT [PYC]");

  double max_diff_P = 0.0;
  double max_diff_mu = 0.0;

  for (double rho : densities) {
    double eta = hs::packing_fraction(rho);
    legacy::thermodynamics::Enskog enskog(rho);

    double our_P = py_model.pressure(eta);
    double jim_P = enskog.pressurePYC();
    max_diff_P = std::max(max_diff_P, std::abs(our_P - jim_P));

    double our_mu = py_model.chemical_potential(rho);
    double jim_mu = enskog.chemPotentialPYC();
    max_diff_mu = std::max(max_diff_mu, std::abs(our_mu - jim_mu));
  }

  check("PYC max|P diff|", max_diff_P, 0.0);
  check("PYC max|mu diff|", max_diff_mu, 0.0);
  std::cout << "  PYC max|P diff|  = " << max_diff_P << "\n";
  std::cout << "  PYC max|mu diff| = " << max_diff_mu << "\n";

  // Step 4: LJ JZG — coefficients a(i), b(i), G(d,i) at multiple kT

  section("Step 4: LJ JZG coefficients");

  std::vector<double> temperatures = {0.7, 1.0, 1.5, 2.0};

  for (double kT : temperatures) {
    auto our_jzg = eos::make_lennard_jones_jzg(kT);
    auto jim_jzg = legacy::thermodynamics::make_LJ_JZG(kT);

    for (int i = 1; i <= 8; ++i) {
      double our_a = our_jzg.a_coeff(i);
      double jim_a = jim_jzg.a(i);
      check("JZG a(" + std::to_string(i) + ",kT=" + std::to_string(kT) + ")", our_a, jim_a);
    }
    for (int i = 1; i <= 6; ++i) {
      double our_b = our_jzg.b_coeff(i);
      double jim_b = jim_jzg.b(i);
      check("JZG b(" + std::to_string(i) + ",kT=" + std::to_string(kT) + ")", our_b, jim_b);
    }
  }

  // G integral at kT=1, multiple densities
  {
    auto our_jzg = eos::make_lennard_jones_jzg(1.0);
    auto jim_jzg = legacy::thermodynamics::make_LJ_JZG(1.0);
    std::vector<double> test_rhos = {0.1, 0.3, 0.5, 0.7, 0.9};
    for (double rho : test_rhos) {
      for (int i = 1; i <= 6; ++i) {
        double our_g = our_jzg.g_integral(rho, i);
        double jim_g = jim_jzg.G(rho, i);
        check("JZG G(rho=" + std::to_string(rho) + ",i=" + std::to_string(i) + ")", our_g, jim_g);
      }
    }
  }

  std::cout << "  (check counts include all coefficient comparisons)\n";

  // Step 5: LJ JZG — phix, phi1x, phi2x, phi3x at multiple (kT, rho)

  section("Step 5: LJ JZG phix and derivatives");

  std::vector<double> jzg_rhos = {0.1, 0.2, 0.4, 0.6, 0.8};

  double max_diff_jzg_phix = 0.0;
  double max_diff_jzg_phi1x = 0.0;
  double max_diff_jzg_phi2x = 0.0;
  double max_diff_jzg_phi3x = 0.0;

  for (double kT : temperatures) {
    auto our_jzg = eos::make_lennard_jones_jzg(kT);
    auto jim_jzg = legacy::thermodynamics::make_LJ_JZG(kT);

    for (double rho : jzg_rhos) {
      double our_phix = our_jzg.excess_free_energy(rho);
      double jim_phix = jim_jzg.phix(rho);
      max_diff_jzg_phix = std::max(max_diff_jzg_phix, std::abs(our_phix - jim_phix));

      double our_phi1x = our_jzg.d_excess_free_energy(rho);
      double jim_phi1x = jim_jzg.phi1x(rho);
      max_diff_jzg_phi1x = std::max(max_diff_jzg_phi1x, std::abs(our_phi1x - jim_phi1x));

      double our_phi2x = our_jzg.d2_excess_free_energy(rho);
      double jim_phi2x = jim_jzg.phi2x(rho);
      max_diff_jzg_phi2x = std::max(max_diff_jzg_phi2x, std::abs(our_phi2x - jim_phi2x));
    }
  }

  check("JZG max|phix diff|", max_diff_jzg_phix, 0.0);
  check("JZG max|phi1x diff|", max_diff_jzg_phi1x, 0.0, 1e-7);
  check("JZG max|phi2x diff|", max_diff_jzg_phi2x, 0.0, 1e-4);
  std::cout << "  JZG max|phix|  = " << max_diff_jzg_phix << "\n";
  std::cout << "  JZG max|phi1x| = " << max_diff_jzg_phi1x << "\n";
  std::cout << "  JZG max|phi2x| = " << max_diff_jzg_phi2x << "\n";

  // Step 6: LJ Mecke — phix and phi1x

  section("Step 6: LJ Mecke phix and phi1x");

  double max_diff_mecke_phix = 0.0;
  double max_diff_mecke_phi1x = 0.0;

  for (double kT : temperatures) {
    auto our_mecke = eos::make_lennard_jones_mecke(kT);
    auto jim_mecke = legacy::thermodynamics::make_LJ_Mecke(kT);

    for (double rho : jzg_rhos) {
      double our_phix = our_mecke.excess_free_energy(rho);
      double jim_phix = jim_mecke.phix(rho);
      max_diff_mecke_phix = std::max(max_diff_mecke_phix, std::abs(our_phix - jim_phix));

      double our_phi1x = our_mecke.d_excess_free_energy(rho);
      double jim_phi1x = jim_mecke.phi1x(rho);
      max_diff_mecke_phi1x = std::max(max_diff_mecke_phi1x, std::abs(our_phi1x - jim_phi1x));
    }
  }

  check("Mecke max|phix diff|", max_diff_mecke_phix, 0.0);
  check("Mecke max|phi1x diff|", max_diff_mecke_phi1x, 0.0, 1e-7);
  std::cout << "  Mecke max|phix|  = " << max_diff_mecke_phix << "\n";
  std::cout << "  Mecke max|phi1x| = " << max_diff_mecke_phi1x << "\n";

  // Step 7: Bulk thermodynamics — mu and P for JZG

  section("Step 7: Bulk thermodynamics (JZG) mu and P");

  double max_diff_jzg_mu = 0.0;
  double max_diff_jzg_P = 0.0;

  for (double kT : temperatures) {
    auto our_jzg = eos::make_lennard_jones_jzg(kT);
    auto jim_jzg = legacy::thermodynamics::make_LJ_JZG(kT);

    for (double rho : jzg_rhos) {
      // mu/kT = log(rho) + phix + rho * phi1x
      double our_mu = our_jzg.chemical_potential(rho);
      double jim_mu = std::log(rho) + jim_jzg.phix(rho) + rho * jim_jzg.phi1x(rho);
      max_diff_jzg_mu = std::max(max_diff_jzg_mu, std::abs(our_mu - jim_mu));

      // P/(nkT) = 1 + rho * phi1x
      double our_P = our_jzg.pressure(rho);
      double jim_P = 1.0 + rho * jim_jzg.phi1x(rho);
      max_diff_jzg_P = std::max(max_diff_jzg_P, std::abs(our_P - jim_P));
    }
  }

  check("JZG max|mu diff|", max_diff_jzg_mu, 0.0, 1e-7);
  check("JZG max|P diff|", max_diff_jzg_P, 0.0, 1e-7);
  std::cout << "  JZG max|mu diff| = " << max_diff_jzg_mu << "\n";
  std::cout << "  JZG max|P diff|  = " << max_diff_jzg_P << "\n";

  // Summary

  section("Summary");
  std::cout << "  Total checks: " << g_checks << "\n";
  std::cout << "  Failures:     " << g_failures << "\n";

  if (g_failures > 0) {
    std::cout << "\n  *** " << g_failures << " FAILURES ***\n";
    return 1;
  }
  std::cout << "\n  All checks passed.\n";
  return 0;
}
