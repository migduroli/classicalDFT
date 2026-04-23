// check.cpp — Exhaustive cross-validation of our potentials against Jim
// Lutsko's classicalDFT library (Potential1.h, Potential.cpp).
//
// Compares: V(r), V2(r2), V0(r), Watt(r), Watt2(r2), derived quantities
// (shift, rmin, Vmin, r0, hard_core), HSD(kT), and a_vdw(kT) for all three
// potentials (LJ, tWF, WHDF) under WCA splitting at multiple temperatures.

#include "legacy/classicaldft.hpp"

#include <cmath>
#include <dftlib>
#include <iomanip>
#include <iostream>
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

  namespace pot = physics::potentials;

  // Parameters
  constexpr double sigma = 1.0;
  constexpr double eps = 1.0;

  // LJ with cutoff 2.5
  constexpr double lj_rcut = 2.5;
  // tWF with cutoff at -1 (convention: Jim uses rcut=-1.0 for sigma_based cutoff)
  // For cross-validation we use a positive cutoff
  constexpr double twf_alpha = 50.0;
  constexpr double twf_rcut = 2.5;
  // WHDF with cutoff 3.0
  constexpr double whdf_rcut = 3.0;

  // Create Jim's potentials
  auto jim_lj = legacy::potentials::make_LJ(sigma, eps, lj_rcut);
  auto jim_twf = legacy::potentials::make_tWF(sigma, eps, twf_rcut, twf_alpha);
  auto jim_whdf = legacy::potentials::make_WHDF(sigma, eps, whdf_rcut);

  // Create our potentials
  auto our_lj = pot::make_lennard_jones(sigma, eps, lj_rcut);
  auto our_twf = pot::make_ten_wolde_frenkel(sigma, eps, twf_rcut, twf_alpha);
  auto our_whdf = pot::make_wang_ramirez_dobnikar_frenkel(sigma, eps, whdf_rcut);

  pot::Potential plj = our_lj;
  pot::Potential ptwf = our_twf;
  pot::Potential pwhdf = our_whdf;

  // Step 1: Derived quantities (shift, rmin, Vmin, r0, hard_core)

  section("Step 1: Derived quantities");

  check("LJ shift", our_lj.epsilon_shift, jim_lj.shift);
  check("LJ rmin", our_lj.r_min, jim_lj.rmin);
  check("LJ Vmin", our_lj.v_min, jim_lj.Vmin);
  check("LJ r0", our_lj.r_zero, jim_lj.r0);
  check("LJ hard_core", plj.hard_core_diameter(), jim_lj.getHardCore());

  check("tWF shift", our_twf.epsilon_shift, jim_twf.shift);
  check("tWF rmin", our_twf.r_min, jim_twf.rmin);
  check("tWF Vmin", our_twf.v_min, jim_twf.Vmin);
  check("tWF r0", our_twf.r_zero, jim_twf.r0);
  check("tWF hard_core", ptwf.hard_core_diameter(), jim_twf.getHardCore());

  check("WHDF eps_eff", our_whdf.epsilon_effective, jim_whdf.eps_rescaled);
  check("WHDF rmin", our_whdf.r_min, jim_whdf.rmin);
  check("WHDF Vmin", our_whdf.v_min, jim_whdf.Vmin);
  check("WHDF hard_core", pwhdf.hard_core_diameter(), jim_whdf.getHardCore());

  std::cout << "  LJ:   shift=" << our_lj.epsilon_shift << " rmin=" << our_lj.r_min << " Vmin=" << our_lj.v_min
            << " r0=" << our_lj.r_zero << "\n";
  std::cout << "  tWF:  shift=" << our_twf.epsilon_shift << " rmin=" << our_twf.r_min << " Vmin=" << our_twf.v_min
            << " r0=" << our_twf.r_zero << "\n";
  std::cout << "  WHDF: eps_eff=" << our_whdf.epsilon_effective << " rmin=" << our_whdf.r_min
            << " Vmin=" << our_whdf.v_min << "\n";

  // Step 2: Raw potential vr(r) at many points

  section("Step 2: Raw potential vr(r)");

  std::vector<double> r_points;
  for (int i = 1; i <= 100; ++i) {
    r_points.push_back(0.8 + i * 0.02);
  }

  double max_diff_lj_vr = 0.0;
  double max_diff_twf_vr = 0.0;
  double max_diff_whdf_vr = 0.0;

  for (double r : r_points) {
    double our_v = our_lj(r);
    double jim_v = legacy::potentials::LJ::vr(sigma, eps, r);
    max_diff_lj_vr = std::max(max_diff_lj_vr, std::abs(our_v - jim_v));

    if (r > sigma + 1e-6) {
      double our_v2 = our_twf(r);
      double jim_v2 = legacy::potentials::tWF::vr(sigma, eps, twf_alpha, r);
      max_diff_twf_vr = std::max(max_diff_twf_vr, std::abs(our_v2 - jim_v2));
    }

    if (r < whdf_rcut) {
      double our_v3 = our_whdf(r);
      double jim_v3 = legacy::potentials::WHDF::vr(jim_whdf.eps_rescaled, sigma, whdf_rcut, r);
      max_diff_whdf_vr = std::max(max_diff_whdf_vr, std::abs(our_v3 - jim_v3));
    }
  }

  check("LJ max|vr diff|", max_diff_lj_vr, 0.0);
  check("tWF max|vr diff|", max_diff_twf_vr, 0.0);
  check("WHDF max|vr diff|", max_diff_whdf_vr, 0.0);
  std::cout << "  LJ  max|vr diff| = " << max_diff_lj_vr << "\n";
  std::cout << "  tWF max|vr diff| = " << max_diff_twf_vr << "\n";
  std::cout << "  WHDF max|vr diff| = " << max_diff_whdf_vr << "\n";

  // Step 3: vr2(r2) — potential from r-squared

  section("Step 3: Raw potential vr2(r2)");

  double max_diff_lj_vr2 = 0.0;
  double max_diff_twf_vr2 = 0.0;
  double max_diff_whdf_vr2 = 0.0;

  for (double r : r_points) {
    double r2 = r * r;
    double our_v = our_lj.from_r2(r2);
    double jim_v = legacy::potentials::LJ::vr2(sigma, eps, r2);
    max_diff_lj_vr2 = std::max(max_diff_lj_vr2, std::abs(our_v - jim_v));

    if (r > sigma + 1e-6) {
      double our_v2 = our_twf.from_r2(r2);
      double jim_v2 = legacy::potentials::tWF::vr2(sigma, eps, twf_alpha, r2);
      max_diff_twf_vr2 = std::max(max_diff_twf_vr2, std::abs(our_v2 - jim_v2));
    }

    if (r < whdf_rcut) {
      double our_v3 = our_whdf.from_r2(r2);
      double jim_v3 = legacy::potentials::WHDF::vr2(jim_whdf.eps_rescaled, sigma, whdf_rcut, r2);
      max_diff_whdf_vr2 = std::max(max_diff_whdf_vr2, std::abs(our_v3 - jim_v3));
    }
  }

  check("LJ max|vr2 diff|", max_diff_lj_vr2, 0.0);
  check("tWF max|vr2 diff|", max_diff_twf_vr2, 0.0);
  check("WHDF max|vr2 diff|", max_diff_whdf_vr2, 0.0);
  std::cout << "  LJ  max|vr2 diff| = " << max_diff_lj_vr2 << "\n";
  std::cout << "  tWF max|vr2 diff| = " << max_diff_twf_vr2 << "\n";
  std::cout << "  WHDF max|vr2 diff| = " << max_diff_whdf_vr2 << "\n";

  // Step 4: Cut-and-shifted V(r)

  section("Step 4: Cut-and-shifted V(r)");

  double max_diff_lj_V = 0.0;
  double max_diff_twf_V = 0.0;
  double max_diff_whdf_V = 0.0;

  for (double r : r_points) {
    double our_v = plj.energy(r);
    double jim_v = legacy::potentials::V(jim_lj, r);
    max_diff_lj_V = std::max(max_diff_lj_V, std::abs(our_v - jim_v));

    if (r > sigma + 1e-6) {
      double our_v2 = ptwf.energy(r);
      double jim_v2 = legacy::potentials::V(jim_twf, r);
      max_diff_twf_V = std::max(max_diff_twf_V, std::abs(our_v2 - jim_v2));
    }

    if (r < whdf_rcut) {
      double our_v3 = pwhdf.energy(r);
      double jim_v3 = legacy::potentials::V(jim_whdf, r);
      max_diff_whdf_V = std::max(max_diff_whdf_V, std::abs(our_v3 - jim_v3));
    }
  }

  check("LJ max|V diff|", max_diff_lj_V, 0.0);
  check("tWF max|V diff|", max_diff_twf_V, 0.0);
  check("WHDF max|V diff|", max_diff_whdf_V, 0.0);
  std::cout << "  LJ  max|V diff| = " << max_diff_lj_V << "\n";
  std::cout << "  tWF max|V diff| = " << max_diff_twf_V << "\n";
  std::cout << "  WHDF max|V diff| = " << max_diff_whdf_V << "\n";

  // Step 5: Repulsive part V0(r) under WCA splitting

  section("Step 5: Repulsive part V0(r) [WCA]");

  double max_diff_lj_V0 = 0.0;
  double max_diff_twf_V0 = 0.0;
  double max_diff_whdf_V0 = 0.0;

  for (double r : r_points) {
    double our_v0 = plj.repulsive(r, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_v0 = legacy::potentials::V0(jim_lj, r);
    max_diff_lj_V0 = std::max(max_diff_lj_V0, std::abs(our_v0 - jim_v0));

    if (r > sigma + 1e-6) {
      double our_v02 = ptwf.repulsive(r, pot::SplitScheme::WeeksChandlerAndersen);
      double jim_v02 = legacy::potentials::V0(jim_twf, r);
      max_diff_twf_V0 = std::max(max_diff_twf_V0, std::abs(our_v02 - jim_v02));
    }

    double our_v03 = pwhdf.repulsive(r, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_v03 = legacy::potentials::V0(jim_whdf, r);
    max_diff_whdf_V0 = std::max(max_diff_whdf_V0, std::abs(our_v03 - jim_v03));
  }

  check("LJ max|V0 diff|", max_diff_lj_V0, 0.0);
  check("tWF max|V0 diff|", max_diff_twf_V0, 0.0);
  check("WHDF max|V0 diff|", max_diff_whdf_V0, 0.0);
  std::cout << "  LJ  max|V0 diff| = " << max_diff_lj_V0 << "\n";
  std::cout << "  tWF max|V0 diff| = " << max_diff_twf_V0 << "\n";
  std::cout << "  WHDF max|V0 diff| = " << max_diff_whdf_V0 << "\n";

  // Step 6: Attractive tail Watt(r) under WCA splitting

  section("Step 6: Attractive tail Watt(r) [WCA]");

  double max_diff_lj_W = 0.0;
  double max_diff_twf_W = 0.0;
  double max_diff_whdf_W = 0.0;

  for (double r : r_points) {
    double our_w = plj.attractive(r, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_w = legacy::potentials::Watt(jim_lj, r);
    max_diff_lj_W = std::max(max_diff_lj_W, std::abs(our_w - jim_w));

    if (r > sigma + 1e-6) {
      double our_w2 = ptwf.attractive(r, pot::SplitScheme::WeeksChandlerAndersen);
      double jim_w2 = legacy::potentials::Watt(jim_twf, r);
      max_diff_twf_W = std::max(max_diff_twf_W, std::abs(our_w2 - jim_w2));
    }

    double our_w3 = pwhdf.attractive(r, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_w3 = legacy::potentials::Watt(jim_whdf, r);
    max_diff_whdf_W = std::max(max_diff_whdf_W, std::abs(our_w3 - jim_w3));
  }

  check("LJ max|Watt diff|", max_diff_lj_W, 0.0);
  check("tWF max|Watt diff|", max_diff_twf_W, 0.0);
  check("WHDF max|Watt diff|", max_diff_whdf_W, 0.0);
  std::cout << "  LJ  max|Watt diff| = " << max_diff_lj_W << "\n";
  std::cout << "  tWF max|Watt diff| = " << max_diff_twf_W << "\n";
  std::cout << "  WHDF max|Watt diff| = " << max_diff_whdf_W << "\n";

  // Step 7: Hard-sphere diameter at multiple temperatures

  section("Step 7: Hard-sphere diameter HSD(kT) [WCA]");

  std::vector<double> temperatures = {0.5, 0.7, 1.0, 1.5, 2.0, 5.0};

  for (double kT : temperatures) {
    double our_d = plj.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_d = legacy::potentials::getHSD(jim_lj, kT);
    check("LJ HSD(kT=" + std::to_string(kT) + ")", our_d, jim_d, 1e-6);
    std::cout << "  LJ  HSD(kT=" << kT << "): ours=" << our_d << " jim=" << jim_d << " diff=" << std::abs(our_d - jim_d)
              << "\n";
  }

  for (double kT : temperatures) {
    double our_d = ptwf.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_d = legacy::potentials::getHSD(jim_twf, kT);
    check("tWF HSD(kT=" + std::to_string(kT) + ")", our_d, jim_d, 1e-6);
    std::cout << "  tWF HSD(kT=" << kT << "): ours=" << our_d << " jim=" << jim_d << " diff=" << std::abs(our_d - jim_d)
              << "\n";
  }

  for (double kT : temperatures) {
    double our_d = pwhdf.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_d = legacy::potentials::getHSD(jim_whdf, kT);
    check("WHDF HSD(kT=" + std::to_string(kT) + ")", our_d, jim_d, 1e-6);
    std::cout << "  WHDF HSD(kT=" << kT << "): ours=" << our_d << " jim=" << jim_d
              << " diff=" << std::abs(our_d - jim_d) << "\n";
  }

  // Step 8: Van der Waals integral at multiple temperatures

  section("Step 8: Van der Waals integral a_vdw(kT) [WCA]");

  for (double kT : temperatures) {
    double our_a = plj.vdw_integral(kT, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_a = legacy::potentials::getVDW(jim_lj, kT);
    check("LJ a_vdw(kT=" + std::to_string(kT) + ")", our_a, jim_a, 1e-6);
    std::cout << "  LJ  a_vdw(kT=" << kT << "): ours=" << our_a << " jim=" << jim_a
              << " diff=" << std::abs(our_a - jim_a) << "\n";
  }

  for (double kT : temperatures) {
    double our_a = ptwf.vdw_integral(kT, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_a = legacy::potentials::getVDW(jim_twf, kT);
    check("tWF a_vdw(kT=" + std::to_string(kT) + ")", our_a, jim_a, 1e-6);
    std::cout << "  tWF a_vdw(kT=" << kT << "): ours=" << our_a << " jim=" << jim_a
              << " diff=" << std::abs(our_a - jim_a) << "\n";
  }

  for (double kT : temperatures) {
    double our_a = pwhdf.vdw_integral(kT, pot::SplitScheme::WeeksChandlerAndersen);
    double jim_a = legacy::potentials::getVDW(jim_whdf, kT);
    check("WHDF a_vdw(kT=" + std::to_string(kT) + ")", our_a, jim_a, 1e-6);
    std::cout << "  WHDF a_vdw(kT=" << kT << "): ours=" << our_a << " jim=" << jim_a
              << " diff=" << std::abs(our_a - jim_a) << "\n";
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
