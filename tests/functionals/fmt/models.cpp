#include "dft/functionals/fmt/models.hpp"

#include "dft/physics/hard_spheres.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <functional>
#include <numbers>

using namespace dft::functionals::fmt;
namespace hs = dft::physics::hard_spheres;

static double numerical_derivative(std::function<double(double)> f, double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}

// Name and needs_tensor queries

TEST_CASE("fmt model names", "[fmt][models]") {
  CHECK(name(FMTModel{Rosenfeld{}}) == "Rosenfeld");
  CHECK(name(FMTModel{RSLT{}}) == "RSLT");
  CHECK(name(FMTModel{WhiteBearI{}}) == "WhiteBearI");
  CHECK(name(FMTModel{WhiteBearII{}}) == "WhiteBearII");
}

TEST_CASE("fmt model needs_tensor", "[fmt][models]") {
  CHECK_FALSE(needs_tensor(FMTModel{Rosenfeld{}}));
  CHECK_FALSE(needs_tensor(FMTModel{RSLT{}}));
  CHECK(needs_tensor(FMTModel{WhiteBearI{}}));
  CHECK(needs_tensor(FMTModel{WhiteBearII{}}));
}

// Dilute limits (eta -> 0): all models approach ideal gas behavior

TEST_CASE("Rosenfeld dilute limits", "[fmt][models]") {
  FMTModel model = Rosenfeld{};
  double eta = 1e-10;
  CHECK(ideal_factor(model, eta) == Catch::Approx(std::log(1.0 - eta)).margin(1e-14));
  CHECK(pair_factor(model, eta) == Catch::Approx(1.0).margin(1e-9));
  CHECK(triplet_factor(model, eta) == Catch::Approx(1.0).margin(1e-9));
}

TEST_CASE("RSLT dilute limits", "[fmt][models]") {
  FMTModel model = RSLT{};
  double eta = 1e-10;
  CHECK(ideal_factor(model, eta) == Catch::Approx(std::log(1.0 - eta)).margin(1e-14));
  CHECK(pair_factor(model, eta) == Catch::Approx(1.0).margin(1e-8));
  CHECK(triplet_factor(model, eta) == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("WhiteBearI dilute limits", "[fmt][models]") {
  FMTModel model = WhiteBearI{};
  CHECK(triplet_factor(model, 1e-10) == Catch::Approx(1.0).margin(1e-6));
}

TEST_CASE("WhiteBearII dilute limits", "[fmt][models]") {
  FMTModel model = WhiteBearII{};
  CHECK(triplet_factor(model, 1e-10) == Catch::Approx(1.0).margin(1e-6));
}

// f-function derivative consistency via numerical differentiation

TEST_CASE("Rosenfeld f1 derivative matches numerical", "[fmt][models]") {
  FMTModel model = Rosenfeld{};
  CHECK(
      d_ideal_factor(model, 0.3) ==
      Catch::Approx(numerical_derivative([&](double e) { return ideal_factor(model, e); }, 0.3)).margin(1e-8)
  );
}

TEST_CASE("Rosenfeld f2 derivative matches numerical", "[fmt][models]") {
  FMTModel model = Rosenfeld{};
  CHECK(
      d_pair_factor(model, 0.3) ==
      Catch::Approx(numerical_derivative([&](double e) { return pair_factor(model, e); }, 0.3)).margin(1e-8)
  );
}

TEST_CASE("Rosenfeld f3 derivative matches numerical", "[fmt][models]") {
  FMTModel model = Rosenfeld{};
  CHECK(
      d_triplet_factor(model, 0.3) ==
      Catch::Approx(numerical_derivative([&](double e) { return triplet_factor(model, e); }, 0.3)).margin(1e-8)
  );
}

TEST_CASE("RSLT f3 derivatives are consistent", "[fmt][models]") {
  FMTModel model = RSLT{};
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    CHECK(
        d_triplet_factor(model, eta) ==
        Catch::Approx(numerical_derivative([&](double e) { return triplet_factor(model, e); }, eta)).margin(1e-5)
    );
  }
}

TEST_CASE("WhiteBearI f3 derivatives are consistent", "[fmt][models]") {
  FMTModel model = WhiteBearI{};
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    CHECK(
        d_triplet_factor(model, eta) ==
        Catch::Approx(numerical_derivative([&](double e) { return triplet_factor(model, e); }, eta)).margin(1e-5)
    );
  }
}

TEST_CASE("WhiteBearII f2 and f3 derivatives are consistent", "[fmt][models]") {
  FMTModel model = WhiteBearII{};
  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    CHECK(
        d_pair_factor(model, eta) ==
        Catch::Approx(numerical_derivative([&](double e) { return pair_factor(model, e); }, eta)).margin(1e-5)
    );
    CHECK(
        d_triplet_factor(model, eta) ==
        Catch::Approx(numerical_derivative([&](double e) { return triplet_factor(model, e); }, eta)).margin(1e-5)
    );
  }
}

// RSLT f3 Taylor expansion branch (eta < 1e-6)

TEST_CASE("RSLT f3 small eta uses Taylor branch", "[fmt][models]") {
  FMTModel model = RSLT{};
  double eta = 1e-8;
  CHECK(triplet_factor(model, eta) == Catch::Approx(1.0).margin(1e-6));
}

// Free energy cross-checks against hard-sphere models

TEST_CASE("Rosenfeld free energy matches PY compressibility", "[fmt][models]") {
  FMTModel model = Rosenfeld{};
  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    double eta = hs::packing_fraction(rho);
    double f_fmt = free_energy_density(model, rho, 1.0) / rho;
    double f_py = hs::PercusYevickCompressibility::excess_free_energy(eta);
    CHECK(f_fmt == Catch::Approx(f_py).margin(1e-10));
  }
}

TEST_CASE("WhiteBearI free energy matches Carnahan-Starling", "[fmt][models]") {
  FMTModel model = WhiteBearI{};
  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.9}) {
    double eta = hs::packing_fraction(rho);
    double f_fmt = free_energy_density(model, rho, 1.0) / rho;
    double f_cs = hs::CarnahanStarling::excess_free_energy(eta);
    CHECK(f_fmt == Catch::Approx(f_cs).margin(1e-10));
  }
}

// Bulk chemical potential cross-check

TEST_CASE("Rosenfeld chemical potential matches numerical derivative", "[fmt][models]") {
  FMTModel model = Rosenfeld{};
  double rho = 0.5;
  double mu_fmt = excess_chemical_potential(model, rho, 1.0);
  double mu_num = numerical_derivative([&](double r) { return free_energy_density(model, r, 1.0); }, rho);
  CHECK(mu_fmt == Catch::Approx(mu_num).margin(1e-6));
}

TEST_CASE("WhiteBearI chemical potential matches numerical derivative", "[fmt][models]") {
  FMTModel model = WhiteBearI{};
  double rho = 0.5;
  double mu_fmt = excess_chemical_potential(model, rho, 1.0);
  double mu_num = numerical_derivative([&](double r) { return free_energy_density(model, r, 1.0); }, rho);
  CHECK(mu_fmt == Catch::Approx(mu_num).margin(1e-6));
}

TEST_CASE("WhiteBearII chemical potential matches numerical derivative", "[fmt][models]") {
  FMTModel model = WhiteBearII{};
  double rho = 0.5;
  double mu_fmt = excess_chemical_potential(model, rho, 1.0);
  double mu_num = numerical_derivative([&](double r) { return free_energy_density(model, r, 1.0); }, rho);
  CHECK(mu_fmt == Catch::Approx(mu_num).margin(1e-6));
}

// phi and d_phi on uniform measures

TEST_CASE("phi on uniform measures equals free_energy_density", "[fmt][models]") {
  FMTModel model = WhiteBearI{};
  double rho = 0.6;
  auto m = make_uniform_measures(rho, 1.0);
  CHECK(phi(model, m) == Catch::Approx(free_energy_density(model, rho, 1.0)).margin(1e-14));
}

TEST_CASE("d_phi returns non-zero derivatives for moderate packing", "[fmt][models]") {
  FMTModel model = WhiteBearI{};
  auto m = make_uniform_measures(0.5, 1.0);
  auto dm = d_phi(model, m);

  CHECK(dm.eta != 0.0);
  CHECK(dm.n0 != 0.0);
  CHECK(dm.n1 != 0.0);
  CHECK(dm.n2 != 0.0);
}

TEST_CASE("d_phi tensor derivatives vanish for non-tensor models", "[fmt][models]") {
  FMTModel model = Rosenfeld{};
  auto m = make_uniform_measures(0.5, 1.0);
  auto dm = d_phi(model, m);
  CHECK(arma::accu(arma::abs(dm.T)) == 0.0);
}

TEST_CASE("d_phi tensor derivatives are nonzero for tensor models", "[fmt][models]") {
  FMTModel model = WhiteBearI{};
  auto m = make_uniform_measures(0.5, 1.0);
  auto dm = d_phi(model, m);
  CHECK(arma::accu(arma::abs(dm.T)) > 0.0);
}

// RSLT phi3 zero guard for n2 ~ 0

TEST_CASE("RSLT phi3 returns zero for negligible n2", "[fmt][models]") {
  Measures m;
  m.n2 = 1e-35;
  m.products = inner_products(m);
  CHECK(RSLT::phi3(m) == 0.0);
  CHECK(RSLT::d_phi3_d_n2(m) == 0.0);
  CHECK(arma::norm(RSLT::d_phi3_d_v1(m)) == 0.0);
}
