// Cross-validation of FMT model functions against Jim's code.

#include "dft.hpp"
#include "legacy/classicaldft.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;
namespace fmt = functionals::fmt;
using Catch::Approx;

static const std::vector<double> ETAS = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60};

TEST_CASE("Rosenfeld f1, f2, f3 match legacy", "[integration][fmt]") {
  for (double eta : ETAS) {
    CHECK(fmt::Rosenfeld{}.f1(eta) == Approx(legacy::fmt::Rosenfeld_model::f1(eta)).margin(1e-10));
    CHECK(fmt::Rosenfeld{}.f2(eta) == Approx(legacy::fmt::Rosenfeld_model::f2(eta)).margin(1e-10));
    CHECK(fmt::Rosenfeld{}.f3(eta) == Approx(legacy::fmt::Rosenfeld_model::f3(eta)).margin(1e-10));
  }
}

TEST_CASE("RSLT f1, f2, f3 match legacy", "[integration][fmt]") {
  for (double eta : ETAS) {
    CHECK(fmt::RSLT{}.f1(eta) == Approx(legacy::fmt::RSLT_model::f1(eta)).margin(1e-10));
    CHECK(fmt::RSLT{}.f2(eta) == Approx(legacy::fmt::RSLT_model::f2(eta)).margin(1e-10));
    CHECK(fmt::RSLT{}.f3(eta) == Approx(legacy::fmt::RSLT_model::f3(eta)).margin(1e-10));
  }
}

TEST_CASE("WhiteBearI f1, f2, f3 match legacy", "[integration][fmt]") {
  for (double eta : ETAS) {
    CHECK(fmt::WhiteBearI{}.f1(eta) == Approx(legacy::fmt::WhiteBearI_model::f1(eta)).margin(1e-10));
    CHECK(fmt::WhiteBearI{}.f2(eta) == Approx(legacy::fmt::WhiteBearI_model::f2(eta)).margin(1e-10));
    CHECK(fmt::WhiteBearI{}.f3(eta) == Approx(legacy::fmt::WhiteBearI_model::f3(eta)).margin(1e-10));
  }
}

TEST_CASE("WhiteBearII f1, f2, f3 match legacy", "[integration][fmt]") {
  for (double eta : ETAS) {
    CHECK(fmt::WhiteBearII{}.f1(eta) == Approx(legacy::fmt::WhiteBearII_model::f1(eta)).margin(1e-10));
    CHECK(fmt::WhiteBearII{}.f2(eta) == Approx(legacy::fmt::WhiteBearII_model::f2(eta)).margin(1e-10));
    CHECK(fmt::WhiteBearII{}.f3(eta) == Approx(legacy::fmt::WhiteBearII_model::f3(eta)).margin(1e-10));
  }
}

TEST_CASE("esFMT f1, f2, f3 match legacy", "[integration][fmt]") {
  for (double eta : ETAS) {
    auto esfmt = fmt::EsFMT{1.0, 0.0};
    CHECK(esfmt.f1(eta) == Approx(legacy::fmt::esFMT_model::f1(eta)).margin(1e-10));
    CHECK(esfmt.f2(eta) == Approx(legacy::fmt::esFMT_model::f2(eta)).margin(1e-10));
    CHECK(esfmt.f3(eta) == Approx(legacy::fmt::esFMT_model::f3(eta)).margin(1e-10));
  }
}

// Helper to build Jim's FundamentalMeasures from our Measures.
static auto to_legacy_fm(const fmt::Measures& m) -> legacy::fmt::FundamentalMeasures {
  legacy::fmt::FundamentalMeasures fm;
  fm.eta = m.eta;
  fm.s0 = m.n0;
  fm.s1 = m.n1;
  fm.s2 = m.n2;
  for (int i = 0; i < 3; ++i) {
    fm.v1[i] = m.v0[i];
    fm.v2[i] = m.v1[i];
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      fm.T[i][j] = m.T(i, j);
  fm.calculate_derived_quantities();
  return fm;
}

static auto make_test_measures(double eta, double n2, double v_mag, double T_diag) -> fmt::Measures {
  double R = 0.5;
  fmt::Measures m{};
  m.eta = eta;
  m.n2 = n2;
  m.n1 = n2 / (4. * std::numbers::pi * R);
  m.n0 = n2 / (4. * std::numbers::pi * R * R);
  m.v1 = {v_mag, -0.3 * v_mag, 0.1 * v_mag};
  m.v0 = {
      v_mag / (4. * std::numbers::pi * R),
      -0.3 * v_mag / (4. * std::numbers::pi * R),
      0.1 * v_mag / (4. * std::numbers::pi * R)
  };
  m.T = {{T_diag, 0.1, -0.05}, {0.1, T_diag * 0.8, 0.02}, {-0.05, 0.02, T_diag * 0.6}};
  m.products = fmt::inner_products(m);
  return m;
}

TEST_CASE("Rosenfeld Phi3 and derivatives match legacy", "[integration][fmt]") {
  auto m = make_test_measures(0.3, 2.5, 0.4, 0.3);
  auto fm = to_legacy_fm(m);

  CHECK(fmt::Rosenfeld{}.phi3(m) == Approx(legacy::fmt::Rosenfeld_model::Phi3(fm)).margin(1e-10));
  CHECK(fmt::Rosenfeld{}.d_phi3_d_n2(m) == Approx(legacy::fmt::Rosenfeld_model::dPhi3_dS2(fm)).margin(1e-10));
  for (int k = 0; k < 3; ++k)
    CHECK(fmt::Rosenfeld{}.d_phi3_d_v1(m)(k) == Approx(legacy::fmt::Rosenfeld_model::dPhi3_dV2(k, fm)).margin(1e-10));
}

TEST_CASE("RSLT Phi3 and derivatives match legacy", "[integration][fmt]") {
  auto m = make_test_measures(0.3, 2.5, 0.4, 0.3);
  auto fm = to_legacy_fm(m);

  CHECK(fmt::RSLT{}.phi3(m) == Approx(legacy::fmt::RSLT_model::Phi3(fm)).margin(1e-10));
  CHECK(fmt::RSLT{}.d_phi3_d_n2(m) == Approx(legacy::fmt::RSLT_model::dPhi3_dS2(fm)).margin(1e-10));
  for (int k = 0; k < 3; ++k)
    CHECK(fmt::RSLT{}.d_phi3_d_v1(m)(k) == Approx(legacy::fmt::RSLT_model::dPhi3_dV2(k, fm)).margin(1e-10));
}

TEST_CASE("esFMT Phi3 and derivatives match legacy", "[integration][fmt]") {
  auto m = make_test_measures(0.25, 2.0, 0.3, 0.25);
  auto fm = to_legacy_fm(m);

  for (auto [A, B] : std::vector<std::pair<double, double>>{{1.0, 0.0}, {1.0, -1.0}}) {
    auto model = fmt::EsFMT{A, B};
    INFO("A=" << A << " B=" << B);
    CHECK(model.phi3(m) == Approx(legacy::fmt::esFMT_model::Phi3(A, B, fm)).margin(1e-10));
    CHECK(model.d_phi3_d_n2(m) == Approx(legacy::fmt::esFMT_model::dPhi3_dS2(A, B, fm)).margin(1e-10));
    for (int k = 0; k < 3; ++k)
      CHECK(model.d_phi3_d_v1(m)(k) == Approx(legacy::fmt::esFMT_model::dPhi3_dV2(A, k, fm)).margin(1e-10));
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j)
        CHECK(model.d_phi3_d_T(i, j, m) == Approx(legacy::fmt::esFMT_model::dPhi3_dT(A, B, i, j, fm)).margin(1e-10));
  }
}

TEST_CASE("WhiteBearI tensor_phi3 matches esFMT(1,-1)", "[integration][fmt]") {
  auto m = make_test_measures(0.25, 2.0, 0.3, 0.25);
  auto fm = to_legacy_fm(m);

  CHECK(fmt::WhiteBearI{}.phi3(m) == Approx(legacy::fmt::esFMT_model::Phi3(1.0, -1.0, fm)).margin(1e-10));
  CHECK(fmt::WhiteBearI{}.d_phi3_d_n2(m) == Approx(legacy::fmt::esFMT_model::dPhi3_dS2(1.0, -1.0, fm)).margin(1e-10));
  for (int i = 0; i < 3; ++i)
    for (int j = i; j < 3; ++j)
      CHECK(
          fmt::WhiteBearI{}.d_phi3_d_T(i, j, m) ==
          Approx(legacy::fmt::esFMT_model::dPhi3_dT(1.0, -1.0, i, j, fm)).margin(1e-10)
      );
}
