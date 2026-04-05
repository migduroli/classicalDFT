#include "dft/functionals/fmt/models.hpp"

#include "dft/physics/hard_spheres.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <numbers>

using namespace dft::functionals::fmt;
namespace hs = dft::physics::hard_spheres;

static double numerical_derivative(std::function<double(double)> f, double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

// Name and needs_tensor queries

TEST_CASE("fmt model names", "[fmt][models]") {
  CHECK(Rosenfeld::NAME == "Rosenfeld");
  CHECK(RSLT::NAME == "RSLT");
  CHECK(WhiteBearI::NAME == "WhiteBearI");
  CHECK(WhiteBearII::NAME == "WhiteBearII");
}

TEST_CASE("fmt model needs_tensor", "[fmt][models]") {
  CHECK_FALSE(Rosenfeld::NEEDS_TENSOR);
  CHECK_FALSE(RSLT::NEEDS_TENSOR);
  CHECK(WhiteBearI::NEEDS_TENSOR);
  CHECK(WhiteBearII::NEEDS_TENSOR);
}

// Dilute limits (eta -> 0): all models approach ideal gas behavior

TEST_CASE("Rosenfeld dilute limits", "[fmt][models]") {
  Rosenfeld model{};
  double eta = 1e-10;
  CHECK(model.f1(eta) == Catch::Approx(std::log(1.0 - eta)).margin(1e-14));
  CHECK(model.f2(eta) == Catch::Approx(1.0).margin(1e-9));
  CHECK(model.f3(eta) == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("RSLT dilute limits", "[fmt][models]") {
  RSLT model{};
  double eta = 1e-10;
  CHECK(model.f1(eta) == Catch::Approx(std::log(1.0 - eta)).margin(1e-14));
  CHECK(model.f2(eta) == Catch::Approx(1.0).margin(1e-8));
  CHECK(model.f3(eta) == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("WhiteBearI dilute limits", "[fmt][models]") {
  WhiteBearI model{};
  CHECK(model.f3(1e-10) == Catch::Approx(1.5).margin(1e-6));
}

TEST_CASE("WhiteBearII dilute limits", "[fmt][models]") {
  WhiteBearII model{};
  CHECK(model.f3(1e-10) == Catch::Approx(1.5).margin(1e-6));
}

// f-function derivative consistency via numerical differentiation

TEST_CASE("Rosenfeld f1 derivative matches numerical", "[fmt][models]") {
  Rosenfeld model{};
  CHECK(
      model.d_f1(0.3) ==
      Catch::Approx(numerical_derivative([&](double e) { return model.f1(e); }, 0.3)).margin(1e-8)
  );
}

TEST_CASE("Rosenfeld f2 derivative matches numerical", "[fmt][models]") {
  Rosenfeld model{};
  CHECK(
      model.d_f2(0.3) ==
      Catch::Approx(numerical_derivative([&](double e) { return model.f2(e); }, 0.3)).margin(1e-8)
  );
}

TEST_CASE("Rosenfeld f3 derivative matches numerical", "[fmt][models]") {
  Rosenfeld model{};
  CHECK(
      model.d_f3(0.3) ==
      Catch::Approx(numerical_derivative([&](double e) { return model.f3(e); }, 0.3)).margin(1e-8)
  );
}

TEST_CASE("RSLT f3 derivatives are consistent", "[fmt][models]") {
  RSLT model{};
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    CHECK(
        model.d_f3(eta) ==
        Catch::Approx(numerical_derivative([&](double e) { return model.f3(e); }, eta)).margin(1e-5)
    );
  }
}

TEST_CASE("WhiteBearI f3 derivatives are consistent", "[fmt][models]") {
  WhiteBearI model{};
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    CHECK(
        model.d_f3(eta) ==
        Catch::Approx(numerical_derivative([&](double e) { return model.f3(e); }, eta)).margin(1e-5)
    );
  }
}

TEST_CASE("WhiteBearII f2 and f3 derivatives are consistent", "[fmt][models]") {
  WhiteBearII model{};
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    CHECK(
        model.d_f2(eta) ==
        Catch::Approx(numerical_derivative([&](double e) { return model.f2(e); }, eta)).margin(1e-5)
    );
    CHECK(
        model.d_f3(eta) ==
        Catch::Approx(numerical_derivative([&](double e) { return model.f3(e); }, eta)).margin(1e-5)
    );
  }
}

// RSLT f3 Taylor expansion branch (eta < 1e-6)

TEST_CASE("RSLT f3 small eta uses Taylor branch", "[fmt][models]") {
  RSLT model{};
  double eta = 1e-8;
  CHECK(model.f3(eta) == Catch::Approx(1.0).margin(1e-6));
}

// Free energy cross-checks against hard-sphere models

TEST_CASE("Rosenfeld free energy matches PY compressibility", "[fmt][models]") {
  Rosenfeld model{};
  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    double eta = hs::packing_fraction(rho);
    double f_fmt = model.free_energy_density(rho, 1.0) / rho;
    double f_py = hs::PercusYevickCompressibility::excess_free_energy(eta);
    CHECK(f_fmt == Catch::Approx(f_py).margin(1e-10));
  }
}

TEST_CASE("WhiteBearI free energy matches Carnahan-Starling", "[fmt][models]") {
  WhiteBearI model{};
  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    double eta = hs::packing_fraction(rho);
    double f_fmt = model.free_energy_density(rho, 1.0) / rho;
    double f_cs = hs::CarnahanStarling::excess_free_energy(eta);
    CHECK(f_fmt == Catch::Approx(f_cs).margin(1e-10));
  }
}

// Bulk chemical potential cross-check

TEST_CASE("Rosenfeld chemical potential matches numerical derivative", "[fmt][models]") {
  Rosenfeld model{};
  double rho = 0.5;
  double mu_fmt = model.excess_chemical_potential(rho, 1.0);
  double mu_num = numerical_derivative([&](double r) { return model.free_energy_density(r, 1.0); }, rho);
  CHECK(mu_fmt == Catch::Approx(mu_num).margin(1e-6));
}

TEST_CASE("WhiteBearI chemical potential matches numerical derivative", "[fmt][models]") {
  WhiteBearI model{};
  double rho = 0.5;
  double mu_fmt = model.excess_chemical_potential(rho, 1.0);
  double mu_num = numerical_derivative([&](double r) { return model.free_energy_density(r, 1.0); }, rho);
  CHECK(mu_fmt == Catch::Approx(mu_num).margin(1e-6));
}

TEST_CASE("WhiteBearII chemical potential matches numerical derivative", "[fmt][models]") {
  WhiteBearII model{};
  double rho = 0.5;
  double mu_fmt = model.excess_chemical_potential(rho, 1.0);
  double mu_num = numerical_derivative([&](double r) { return model.free_energy_density(r, 1.0); }, rho);
  CHECK(mu_fmt == Catch::Approx(mu_num).margin(1e-6));
}

// phi and d_phi on uniform measures

TEST_CASE("phi on uniform measures equals free_energy_density", "[fmt][models]") {
  WhiteBearI model{};
  double rho = 0.6;
  auto m = make_uniform_measures(rho, 1.0);
  CHECK(model.phi(m) == Catch::Approx(model.free_energy_density(rho, 1.0)).margin(1e-14));
}

TEST_CASE("d_phi returns non-zero derivatives for moderate packing", "[fmt][models]") {
  WhiteBearI model{};
  auto m = make_uniform_measures(0.5, 1.0);
  auto dm = model.d_phi(m);

  CHECK(dm.d_eta != 0.0);
  CHECK(dm.d_n0 != 0.0);
  CHECK(dm.d_n1 != 0.0);
  CHECK(dm.d_n2 != 0.0);
}

TEST_CASE("d_phi tensor derivatives vanish for non-tensor models", "[fmt][models]") {
  Rosenfeld model{};
  auto m = make_uniform_measures(0.5, 1.0);
  auto dm = model.d_phi(m);
  CHECK(arma::accu(arma::abs(dm.d_T)) == 0.0);
}

TEST_CASE("d_phi tensor derivatives are nonzero for tensor models", "[fmt][models]") {
  WhiteBearI model{};
  auto m = make_uniform_measures(0.5, 1.0);
  auto dm = model.d_phi(m);
  CHECK(arma::accu(arma::abs(dm.d_T)) > 0.0);
}

// RSLT phi3 zero guard for n2 ~ 0

TEST_CASE("RSLT phi3 returns zero for negligible n2", "[fmt][models]") {
  Measures m;
  m.n2 = 1e-35;
  m.products = m.inner_products();
  CHECK(RSLT::phi3(m) == 0.0);
  CHECK(RSLT::d_phi3_d_n2(m) == 0.0);
  CHECK(arma::norm(RSLT::d_phi3_d_v1(m)) == 0.0);
}

// WhiteBearII f2 and f3 Taylor branches (eta < 1e-6)

TEST_CASE("WhiteBearII f2 Taylor branch matches standard formula", "[fmt][models]") {
  WhiteBearII model{};
  double eta = 1e-8;
  double f2_taylor = model.f2(eta);
  // As eta -> 0, f2 -> 1 (from the Taylor expansion: 1 + eta + ...)
  CHECK(f2_taylor == Catch::Approx(1.0).margin(1e-6));

  // Check continuity at the branch boundary
  double eta_below = 9e-7;
  double eta_above = 2e-6;
  double f2_below = model.f2(eta_below);
  double f2_above = model.f2(eta_above);
  CHECK(f2_below == Catch::Approx(f2_above).epsilon(1e-4));
}

TEST_CASE("WhiteBearII f3 Taylor branch matches standard formula", "[fmt][models]") {
  WhiteBearII model{};
  double eta = 1e-8;
  double f3_taylor = model.f3(eta);
  // As eta -> 0, f3 -> 1.5 (from the Taylor expansion)
  CHECK(f3_taylor == Catch::Approx(1.5).margin(1e-6));
}

// EsFMT model

TEST_CASE("EsFMT model name and needs_tensor", "[fmt][models]") {
  CHECK(EsFMT::NAME == "esFMT");
  CHECK(EsFMT::NEEDS_TENSOR);
}

TEST_CASE("EsFMT f-functions delegate to Rosenfeld", "[fmt][models]") {
  EsFMT esfmt{};
  Rosenfeld rosenfeld{};
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    CHECK(esfmt.f1(eta) == Catch::Approx(rosenfeld.f1(eta)));
    CHECK(esfmt.f2(eta) == Catch::Approx(rosenfeld.f2(eta)));
    CHECK(esfmt.f3(eta) == Catch::Approx(rosenfeld.f3(eta)));
  }
}

TEST_CASE("EsFMT phi3 on uniform measures is nonzero", "[fmt][models]") {
  EsFMT func{.A = 1.0, .B = 0.0};
  auto m = make_uniform_measures(0.5, 1.0);
  double p3 = func.phi3(m);
  CHECK(p3 != 0.0);
}

TEST_CASE("EsFMT d_phi3_d_n2 is nonzero for moderate packing", "[fmt][models]") {
  EsFMT func{.A = 1.0, .B = 0.0};
  auto m = make_uniform_measures(0.5, 1.0);
  double dp3 = func.d_phi3_d_n2(m);
  CHECK(dp3 != 0.0);
}

TEST_CASE("EsFMT d_phi3_d_v1 vanishes for uniform measures", "[fmt][models]") {
  EsFMT func{.A = 1.0, .B = 0.0};
  auto m = make_uniform_measures(0.5, 1.0);
  auto dv = func.d_phi3_d_v1(m);
  CHECK(arma::norm(dv) == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("EsFMT d_phi3_d_T is nonzero for moderate packing", "[fmt][models]") {
  EsFMT func{.A = 1.0, .B = 0.0};
  auto m = make_uniform_measures(0.5, 1.0);
  double dT00 = func.d_phi3_d_T(0, 0, m);
  CHECK(dT00 != 0.0);
}

TEST_CASE("EsFMT with B parameter changes phi3", "[fmt][models]") {
  auto m = make_uniform_measures(0.5, 1.0);
  EsFMT func_b0{.A = 1.0, .B = 0.0};
  EsFMT func_b1{.A = 1.0, .B = 1.0};
  CHECK(func_b0.phi3(m) != Catch::Approx(func_b1.phi3(m)).margin(1e-14));
  CHECK(func_b0.d_phi3_d_n2(m) != Catch::Approx(func_b1.d_phi3_d_n2(m)).margin(1e-14));
  CHECK(func_b0.d_phi3_d_T(0, 0, m) != Catch::Approx(func_b1.d_phi3_d_T(0, 0, m)).margin(1e-14));
}

TEST_CASE("EsFMT free energy density is finite for moderate density", "[fmt][models]") {
  EsFMT model{.A = 1.0, .B = 0.0};
  double f = model.free_energy_density(0.5, 1.0);
  CHECK(std::isfinite(f));
  CHECK(f > 0.0);
}

TEST_CASE("EsFMT chemical potential matches numerical derivative", "[fmt][models]") {
  EsFMT model{.A = 1.0, .B = 0.0};
  double rho = 0.5;
  double mu_fmt = model.excess_chemical_potential(rho, 1.0);
  double mu_num = numerical_derivative([&](double r) { return model.free_energy_density(r, 1.0); }, rho);
  CHECK(mu_fmt == Catch::Approx(mu_num).margin(1e-6));
}

TEST_CASE("EsFMT d_phi returns non-zero tensor derivatives", "[fmt][models]") {
  EsFMT model{.A = 1.0, .B = 0.0};
  auto m = make_uniform_measures(0.5, 1.0);
  auto dm = model.d_phi(m);
  CHECK(arma::accu(arma::abs(dm.d_T)) > 0.0);
  CHECK(dm.d_eta != 0.0);
  CHECK(dm.d_n2 != 0.0);
}
