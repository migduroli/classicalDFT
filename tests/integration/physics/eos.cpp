// Cross-validation of hard-sphere and LJ EOS against Jim's code.

#include "dft.hpp"
#include "legacy/classicaldft.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;
namespace hs = physics::hard_spheres;
namespace eos = physics::eos;
using Catch::Approx;

static const std::vector<double> ETA_VALUES = { 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45 };

TEST_CASE("PY compressibility excess free energy matches legacy", "[integration][thermodynamics]") {
  auto py_model = hs::PercusYevickCompressibility{};
  for (double eta : ETA_VALUES) {
    legacy::thermodynamics::Enskog enskog(eta * 6.0 / std::numbers::pi);
    CHECK(py_model.excess_free_energy(eta) == Approx(enskog.exFreeEnergyPYC()).margin(1e-10));
  }
}

TEST_CASE("PY compressibility derivatives match legacy", "[integration][thermodynamics]") {
  auto py_model = hs::PercusYevickCompressibility{};
  double de = std::numbers::pi / 6.0;
  for (double eta : ETA_VALUES) {
    legacy::thermodynamics::Enskog enskog(eta * 6.0 / std::numbers::pi);
    CHECK(py_model.d_excess_free_energy(eta) * de == Approx(enskog.dexFreeEnergyPYCdRho()).margin(1e-10));
    CHECK(py_model.d2_excess_free_energy(eta) * de * de == Approx(enskog.d2exFreeEnergyPYCdRho2()).margin(1e-10));
    CHECK(py_model.d3_excess_free_energy(eta) * de * de * de == Approx(enskog.d3exFreeEnergyPYCdRho3()).margin(1e-8));
  }
}

TEST_CASE("Carnahan-Starling excess free energy matches legacy", "[integration][thermodynamics]") {
  auto cs_model = hs::CarnahanStarling{};
  for (double eta : ETA_VALUES) {
    legacy::thermodynamics::Enskog enskog(eta * 6.0 / std::numbers::pi);
    CHECK(cs_model.excess_free_energy(eta) == Approx(enskog.exFreeEnergyCS()).margin(1e-10));
  }
}

TEST_CASE("CS derivatives match legacy", "[integration][thermodynamics]") {
  auto cs_model = hs::CarnahanStarling{};
  double de = std::numbers::pi / 6.0;
  for (double eta : ETA_VALUES) {
    legacy::thermodynamics::Enskog enskog(eta * 6.0 / std::numbers::pi);
    CHECK(cs_model.d_excess_free_energy(eta) * de == Approx(enskog.dexFreeEnergyCSdRho()).margin(1e-10));
    CHECK(cs_model.d2_excess_free_energy(eta) * de * de == Approx(enskog.d2exFreeEnergyCSdRho2()).margin(1e-10));
    CHECK(cs_model.d3_excess_free_energy(eta) * de * de * de == Approx(enskog.d3exFreeEnergyCSdRho3()).margin(1e-8));
  }
}

TEST_CASE("HS pressure and chemical potential match legacy", "[integration][thermodynamics]") {
  auto py_model = hs::PercusYevickCompressibility{};
  for (double eta : ETA_VALUES) {
    double rho = eta * 6.0 / std::numbers::pi;
    legacy::thermodynamics::Enskog enskog(rho);
    CHECK(py_model.pressure(eta) == Approx(enskog.pressurePYC()).margin(1e-10));
    CHECK(py_model.chemical_potential(rho) == Approx(enskog.chemPotentialPYC()).margin(1e-10));
  }
}

TEST_CASE("LJ JZG coefficients match legacy", "[integration][thermodynamics]") {
  for (double kT : { 0.7, 1.0, 1.5, 2.0 }) {
    auto our_jzg = eos::make_lennard_jones_jzg(kT, 2.5, true);
    auto jim_jzg = legacy::thermodynamics::make_LJ_JZG(kT, 2.5);
    for (int i = 1; i <= 8; ++i)
      CHECK(our_jzg.a_coeff(i) == Approx(jim_jzg.a(i)).margin(1e-10));
    for (int i = 1; i <= 6; ++i)
      CHECK(our_jzg.b_coeff(i) == Approx(jim_jzg.b(i)).margin(1e-10));
    for (double rho : { 0.1, 0.4, 0.8 })
      for (int i = 1; i <= 6; ++i)
        CHECK(our_jzg.g_integral(rho, i) == Approx(jim_jzg.G(rho, i)).margin(1e-10));
  }
}

TEST_CASE("JZG phix and derivatives match legacy", "[integration][thermodynamics]") {
  for (double kT : { 0.7, 1.0, 1.5, 2.0 }) {
    auto our_jzg = eos::make_lennard_jones_jzg(kT, 2.5, true);
    auto jim_jzg = legacy::thermodynamics::make_LJ_JZG(kT, 2.5);
    for (double rho : { 0.1, 0.2, 0.4, 0.6, 0.8 }) {
      CHECK(our_jzg.excess_free_energy(rho) == Approx(jim_jzg.phix(rho)).margin(1e-10));
      CHECK(our_jzg.d_excess_free_energy(rho) == Approx(jim_jzg.phi1x(rho)).epsilon(1e-7));
      CHECK(our_jzg.d2_excess_free_energy(rho) == Approx(jim_jzg.phi2x(rho)).epsilon(1e-4));
    }
  }
}

TEST_CASE("Mecke phix and phi1x match legacy", "[integration][thermodynamics]") {
  for (double kT : { 0.7, 1.0, 1.5, 2.0 }) {
    auto our_mecke = eos::make_lennard_jones_mecke(kT, 2.5, true);
    auto jim_mecke = legacy::thermodynamics::make_LJ_Mecke(kT, 2.5);
    for (double rho : { 0.1, 0.2, 0.4, 0.6, 0.8 }) {
      CHECK(our_mecke.excess_free_energy(rho) == Approx(jim_mecke.phix(rho)).margin(1e-10));
      CHECK(our_mecke.d_excess_free_energy(rho) == Approx(jim_mecke.phi1x(rho)).epsilon(1e-7));
    }
  }
}
