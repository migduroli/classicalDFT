// check.cpp — Exhaustive cross-validation of our FMT models against Jim
// Lutsko's classicalDFT library (FMT.h).
//
// Compares:
//   1. f1(eta), f2(eta), f3(eta) for all 5 models at many eta values
//   2. Phi3 and dPhi3 derivatives at non-trivial measures
//   3. Bulk fex(eta) for all models
//   4. Full Phi assembly at uniform-density measures

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

// Create a non-trivial FundamentalMeasures for Jim's code from our Measures
static auto make_jim_fm(const functionals::fmt::Measures& m) -> legacy::fmt::FundamentalMeasures {
  legacy::fmt::FundamentalMeasures fm;
  fm.eta = m.eta;
  fm.s0 = m.n0;
  fm.s1 = m.n1;
  fm.s2 = m.n2;
  // Jim's v2 = our v1
  for (int i = 0; i < 3; ++i) {
    fm.v1[i] = m.v0(i);
    fm.v2[i] = m.v1(i);
    for (int j = 0; j < 3; ++j) {
      fm.T[i][j] = m.T(i, j);
    }
  }
  fm.calculate_derived_quantities();
  return fm;
}

int main() {
  std::cout << std::setprecision(15);

  namespace fmt = functionals::fmt;

  // Test eta values (avoiding 0 and 1)
  std::vector<double> etas;
  for (int i = 1; i <= 12; ++i) {
    etas.push_back(i * 0.05);
  }

  // Step 1: f1, f2, f3 for Rosenfeld

  section("Step 1: Rosenfeld f1, f2, f3");

  double max_diff = 0.0;
  for (double eta : etas) {
    check("Ros f1(" + std::to_string(eta) + ")", fmt::Rosenfeld::f1(eta), legacy::fmt::Rosenfeld_model::f1(eta));
    check("Ros f2(" + std::to_string(eta) + ")", fmt::Rosenfeld::f2(eta), legacy::fmt::Rosenfeld_model::f2(eta));
    check("Ros f3(" + std::to_string(eta) + ")", fmt::Rosenfeld::f3(eta), legacy::fmt::Rosenfeld_model::f3(eta));
  }

  // Step 2: f1, f2, f3 for RSLT

  section("Step 2: RSLT f1, f2, f3");

  for (double eta : etas) {
    check("RSLT f1(" + std::to_string(eta) + ")", fmt::RSLT::f1(eta), legacy::fmt::RSLT_model::f1(eta));
    check("RSLT f2(" + std::to_string(eta) + ")", fmt::RSLT::f2(eta), legacy::fmt::RSLT_model::f2(eta));
    check("RSLT f3(" + std::to_string(eta) + ")", fmt::RSLT::f3(eta), legacy::fmt::RSLT_model::f3(eta));
  }

  // Step 3: f1, f2, f3 for WhiteBearI

  section("Step 3: WhiteBearI f1, f2, f3");

  for (double eta : etas) {
    check("WBI f1(" + std::to_string(eta) + ")", fmt::WhiteBearI::f1(eta), legacy::fmt::WhiteBearI_model::f1(eta));
    check("WBI f2(" + std::to_string(eta) + ")", fmt::WhiteBearI::f2(eta), legacy::fmt::WhiteBearI_model::f2(eta));
    check("WBI f3(" + std::to_string(eta) + ")", fmt::WhiteBearI::f3(eta), legacy::fmt::WhiteBearI_model::f3(eta));
  }

  // Step 4: f1, f2, f3 for WhiteBearII

  section("Step 4: WhiteBearII f1, f2, f3");

  for (double eta : etas) {
    check("WBII f1(" + std::to_string(eta) + ")", fmt::WhiteBearII::f1(eta), legacy::fmt::WhiteBearII_model::f1(eta));
    check("WBII f2(" + std::to_string(eta) + ")", fmt::WhiteBearII::f2(eta), legacy::fmt::WhiteBearII_model::f2(eta));
    check("WBII f3(" + std::to_string(eta) + ")", fmt::WhiteBearII::f3(eta), legacy::fmt::WhiteBearII_model::f3(eta));
  }

  // Step 5: f1, f2, f3 for esFMT (same as Rosenfeld)

  section("Step 5: esFMT f1, f2, f3 (= Rosenfeld)");

  fmt::EsFMT esfmt{{}, 1.0, 0.0};
  for (double eta : etas) {
    check("esFMT f1(" + std::to_string(eta) + ")", esfmt.f1(eta), legacy::fmt::esFMT_model::f1(eta));
    check("esFMT f2(" + std::to_string(eta) + ")", esfmt.f2(eta), legacy::fmt::esFMT_model::f2(eta));
    check("esFMT f3(" + std::to_string(eta) + ")", esfmt.f3(eta), legacy::fmt::esFMT_model::f3(eta));
  }

  // Step 6: Bulk fex(eta) for all models

  section("Step 6: Bulk fex(eta) for all models");

  for (double eta : etas) {
    // Rosenfeld: PYC EOS
    // Our bulk fex via fmt::phi at uniform measures
    double rho = 6.0 * eta / std::numbers::pi; // d=1
    auto um = fmt::make_uniform_measures(rho, 1.0);

    fmt::Rosenfeld ros_model{};
    double our_ros_fex = ros_model.phi(um) / rho;
    double jim_ros_fex = legacy::fmt::Rosenfeld_model::fex(eta);
    check("Ros fex(" + std::to_string(eta) + ")", our_ros_fex, jim_ros_fex, 1e-8);

    fmt::RSLT rslt_model{};
    double our_rslt_fex = rslt_model.phi(um) / rho;
    double jim_rslt_fex = legacy::fmt::RSLT_model::fex(eta);
    check("RSLT fex(" + std::to_string(eta) + ")", our_rslt_fex, jim_rslt_fex, 1e-8);

    fmt::WhiteBearI wbi_model{};
    double our_wbi_fex = wbi_model.phi(um) / rho;
    double jim_wbi_fex = legacy::fmt::WhiteBearI_model::fex(eta);
    check("WBI fex(" + std::to_string(eta) + ")", our_wbi_fex, jim_wbi_fex, 1e-8);

    fmt::WhiteBearII wbii_model{};
    double our_wbii_fex = wbii_model.phi(um) / rho;
    double jim_wbii_fex = legacy::fmt::WhiteBearII_model::fex(eta);
    check("WBII fex(" + std::to_string(eta) + ")", our_wbii_fex, jim_wbii_fex, 1e-8);
  }

  // Step 7: Phi3 and dPhi3 for Rosenfeld at non-trivial measures

  section("Step 7: Rosenfeld Phi3 and derivatives");

  {
    fmt::Measures m;
    m.eta = 0.3;
    m.n2 = 2.5;
    m.n1 = 0.8;
    m.n0 = 0.3;
    m.v1 = {0.1, -0.2, 0.15};
    m.v0 = {0.01, -0.02, 0.015};
    m.T = {{0.9, 0.1, -0.05}, {0.1, 0.85, 0.08}, {-0.05, 0.08, 0.75}};
    m.products = m.inner_products();

    auto fm = make_jim_fm(m);

    double our_phi3 = fmt::Rosenfeld::phi3(m);
    double jim_phi3 = legacy::fmt::Rosenfeld_model::Phi3(fm);
    check("Ros Phi3", our_phi3, jim_phi3);

    double our_dphi3_n2 = fmt::Rosenfeld::d_phi3_d_n2(m);
    double jim_dphi3_s2 = legacy::fmt::Rosenfeld_model::dPhi3_dS2(fm);
    check("Ros dPhi3/dn2", our_dphi3_n2, jim_dphi3_s2);

    auto our_dphi3_v1 = fmt::Rosenfeld::d_phi3_d_v1(m);
    for (int k = 0; k < 3; ++k) {
      double jim_dphi3_v2k = legacy::fmt::Rosenfeld_model::dPhi3_dV2(k, fm);
      check("Ros dPhi3/dv1(" + std::to_string(k) + ")", our_dphi3_v1(k), jim_dphi3_v2k);
    }
  }

  // Step 8: Phi3 and dPhi3 for RSLT

  section("Step 8: RSLT Phi3 and derivatives");

  {
    fmt::Measures m;
    m.eta = 0.3;
    m.n2 = 2.5;
    m.n1 = 0.8;
    m.n0 = 0.3;
    m.v1 = {0.1, -0.2, 0.15};
    m.v0 = {0.01, -0.02, 0.015};
    m.T = {{0.9, 0.1, -0.05}, {0.1, 0.85, 0.08}, {-0.05, 0.08, 0.75}};
    m.products = m.inner_products();

    auto fm = make_jim_fm(m);

    double our_phi3 = fmt::RSLT::phi3(m);
    double jim_phi3 = legacy::fmt::RSLT_model::Phi3(fm);
    check("RSLT Phi3", our_phi3, jim_phi3);

    double our_dphi3_n2 = fmt::RSLT::d_phi3_d_n2(m);
    double jim_dphi3_s2 = legacy::fmt::RSLT_model::dPhi3_dS2(fm);
    check("RSLT dPhi3/dn2", our_dphi3_n2, jim_dphi3_s2);

    auto our_dphi3_v1 = fmt::RSLT::d_phi3_d_v1(m);
    for (int k = 0; k < 3; ++k) {
      double jim_dphi3_v2k = legacy::fmt::RSLT_model::dPhi3_dV2(k, fm);
      check("RSLT dPhi3/dv1(" + std::to_string(k) + ")", our_dphi3_v1(k), jim_dphi3_v2k);
    }
  }

  // Step 9: Phi3 and dPhi3 for esFMT (A=1, B=0 and A=1, B=-1)

  section("Step 9: esFMT Phi3 and derivatives");

  {
    fmt::Measures m;
    m.eta = 0.3;
    m.n2 = 2.5;
    m.n1 = 0.8;
    m.n0 = 0.3;
    m.v1 = {0.1, -0.2, 0.15};
    m.v0 = {0.01, -0.02, 0.015};
    m.T = {{0.9, 0.1, -0.05}, {0.1, 0.85, 0.08}, {-0.05, 0.08, 0.75}};
    m.products = m.inner_products();

    auto fm = make_jim_fm(m);

    // A=1, B=0
    {
      fmt::EsFMT es{{}, 1.0, 0.0};
      double our_phi3 = es.phi3(m);
      double jim_phi3 = legacy::fmt::esFMT_model::Phi3(1.0, 0.0, fm);
      check("esFMT(1,0) Phi3", our_phi3, jim_phi3);

      double our_dphi3_n2 = es.d_phi3_d_n2(m);
      double jim_dphi3_s2 = legacy::fmt::esFMT_model::dPhi3_dS2(1.0, 0.0, fm);
      check("esFMT(1,0) dPhi3/dn2", our_dphi3_n2, jim_dphi3_s2);

      auto our_dphi3_v1 = es.d_phi3_d_v1(m);
      for (int k = 0; k < 3; ++k) {
        double jim_dphi3_v2 = legacy::fmt::esFMT_model::dPhi3_dV2(1.0, k, fm);
        check("esFMT(1,0) dPhi3/dv1(" + std::to_string(k) + ")", our_dphi3_v1(k), jim_dphi3_v2);
      }

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          double our_dt = es.d_phi3_d_T(i, j, m);
          double jim_dt = legacy::fmt::esFMT_model::dPhi3_dT(1.0, 0.0, i, j, fm);
          check("esFMT(1,0) dPhi3/dT(" + std::to_string(i) + "," + std::to_string(j) + ")", our_dt, jim_dt);
        }
      }
    }

    // A=1, B=-1 (WhiteBear)
    {
      fmt::EsFMT es{{}, 1.0, -1.0};
      double our_phi3 = es.phi3(m);
      double jim_phi3 = legacy::fmt::esFMT_model::Phi3(1.0, -1.0, fm);
      check("esFMT(1,-1) Phi3", our_phi3, jim_phi3);

      double our_dphi3_n2 = es.d_phi3_d_n2(m);
      double jim_dphi3_s2 = legacy::fmt::esFMT_model::dPhi3_dS2(1.0, -1.0, fm);
      check("esFMT(1,-1) dPhi3/dn2", our_dphi3_n2, jim_dphi3_s2);

      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          double our_dt = es.d_phi3_d_T(i, j, m);
          double jim_dt = legacy::fmt::esFMT_model::dPhi3_dT(1.0, -1.0, i, j, fm);
          check("esFMT(1,-1) dPhi3/dT(" + std::to_string(i) + "," + std::to_string(j) + ")", our_dt, jim_dt);
        }
      }
    }
  }

  // Step 10: WhiteBear tensor Phi3 vs esFMT(A=1, B=-1) Phi3

  section("Step 10: WhiteBearI tensor_phi3 vs esFMT(1,-1)");

  {
    fmt::Measures m;
    m.eta = 0.25;
    m.n2 = 3.0;
    m.n1 = 1.0;
    m.n0 = 0.4;
    m.v1 = {0.3, -0.1, 0.2};
    m.v0 = {0.03, -0.01, 0.02};
    m.T = {{0.95, 0.15, -0.1}, {0.15, 0.9, 0.05}, {-0.1, 0.05, 0.8}};
    m.products = m.inner_products();

    auto fm = make_jim_fm(m);

    // Our WhiteBearI::phi3 uses tensor_phi3
    double our_wbi_phi3 = fmt::WhiteBearI::phi3(m);
    // Jim's uses esFMT(A=1, B=-1)
    double jim_esfmt_phi3 = legacy::fmt::esFMT_model::Phi3(1.0, -1.0, fm);
    check("WBI phi3 vs esFMT(1,-1)", our_wbi_phi3, jim_esfmt_phi3);

    double our_wbi_dphi3_n2 = fmt::WhiteBearI::d_phi3_d_n2(m);
    double jim_es_dphi3_s2 = legacy::fmt::esFMT_model::dPhi3_dS2(1.0, -1.0, fm);
    check("WBI dPhi3/dn2 vs esFMT(1,-1)", our_wbi_dphi3_n2, jim_es_dphi3_s2);

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double our_dt = fmt::WhiteBearI::d_phi3_d_T(i, j, m);
        double jim_dt = legacy::fmt::esFMT_model::dPhi3_dT(1.0, -1.0, i, j, fm);
        check("WBI dPhi3/dT(" + std::to_string(i) + "," + std::to_string(j) + ")", our_dt, jim_dt);
      }
    }
  }

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
