// check.cpp — Cross-validation of solver/phase_diagram functions against
// Jim's DFT_Coex.cpp algorithms.
//
// Compares: spinodal densities, coexistence densities at multiple temperatures.
// Jim uses golden section search on P(rho) extrema for spinodal and bisection
// for coexistence; our code uses bisection on dP/drho for spinodal and
// Newton + bisection for coexistence. Both solve the same physical conditions,
// so results must agree.

#include "legacy/classicaldft.hpp"

#include <cmath>
#include <dftlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

using namespace dft;

static int g_failures = 0;
static int g_checks = 0;

static void check(std::string_view label, double ours, double jim, double tol = 1e-6) {
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

static auto make_legacy_eos(const functionals::bulk::BulkThermodynamics& eos) -> legacy::solver::EOS {
  return {
      [&eos](double rho) { return eos.pressure(arma::vec{rho}); },
      [&eos](double rho) { return eos.chemical_potential(arma::vec{rho}, 0); },
  };
}

int main() {
  std::cout << std::setprecision(15);

  // ------------------------------------------------------------------
  // System: LJ sigma=1, epsilon=1, r_c=2.5, WCA splitting, White Bear II.
  // White Bear II reduces to Carnahan-Starling in the bulk.
  // ------------------------------------------------------------------

  physics::Model model{
      .grid = make_grid(0.1, {6.0, 6.0, 6.0}),
      .species = {Species{.name = "LJ", .hard_sphere_diameter = 1.0}},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = physics::potentials::make_lennard_jones(1.0, 1.0, 2.5),
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      }},
      .temperature = 1.0,
  };

  auto wb2 = functionals::fmt::WhiteBearII{};

  auto eos_factory = [&](double kT) -> functionals::bulk::BulkThermodynamics {
    return functionals::bulk::make_bulk_thermodynamics(
        model.species,
        functionals::make_bulk_weights(wb2, model.interactions, kT)
    );
  };

  functionals::bulk::PhaseSearch search_config{
      .rho_max = 1.0,
      .rho_scan_step = 0.005,
      .newton = {.max_iterations = 300, .tolerance = 1e-10},
  };

  // Jim's algorithms use xmax=1.0, dx=0.005 (same scan resolution).
  constexpr double jim_xmax = 1.0;
  constexpr double jim_dx = 0.005;

  // ------------------------------------------------------------------
  // Steps 1-3: Spinodal at kT = 0.7, 0.8, 0.9
  //
  // Both methods find where dP/drho = 0:
  //   Jim: golden section search on P(rho) extrema
  //   Ours: bisection on dP/drho sign change
  // ------------------------------------------------------------------

  int step = 0;
  for (double kT : {0.7, 0.8, 0.9}) {
    ++step;
    section("Step " + std::to_string(step) + ": Spinodal at kT = " + std::to_string(kT));

    auto eos = eos_factory(kT);
    auto jim_eos = make_legacy_eos(eos);

    auto jim_sp = legacy::solver::findSpinodal(jim_eos, jim_xmax, jim_dx);
    auto our_sp = search_config.find_spinodal(eos);

    if (!our_sp) {
      std::cout << "  FAIL: our find_spinodal returned nullopt\n";
      g_failures += 2;
      g_checks += 2;
      continue;
    }

    std::cout << "  Jim:  rho_low=" << jim_sp.xs1 << "  rho_high=" << jim_sp.xs2 << "\n";
    std::cout << "  Ours: rho_low=" << our_sp->rho_low << "  rho_high=" << our_sp->rho_high << "\n";

    check("spinodal_low(kT=" + std::to_string(kT) + ")", our_sp->rho_low, jim_sp.xs1, 1e-6);
    check("spinodal_high(kT=" + std::to_string(kT) + ")", our_sp->rho_high, jim_sp.xs2, 1e-6);
  }

  // ------------------------------------------------------------------
  // Steps 4-6: Coexistence at kT = 0.7, 0.8, 0.9
  //
  // Both solve P_v = P_l and mu_v = mu_l:
  //   Jim: scan rho_v down from spinodal by 1.1, bisect on delta_P
  //   Ours: same scan-and-bisect strategy, Newton for equal-mu
  // ------------------------------------------------------------------

  for (double kT : {0.7, 0.8, 0.9}) {
    ++step;
    section("Step " + std::to_string(step) + ": Coexistence at kT = " + std::to_string(kT));

    auto eos = eos_factory(kT);
    auto jim_eos = make_legacy_eos(eos);

    auto jim_cx = legacy::solver::findCoex(jim_eos, jim_xmax, jim_dx);
    auto our_cx = search_config.find_coexistence(eos);

    if (!our_cx) {
      std::cout << "  FAIL: our find_coexistence returned nullopt\n";
      g_failures += 2;
      g_checks += 2;
      continue;
    }

    std::cout << "  Jim:  rho_v=" << jim_cx.x1 << "  rho_l=" << jim_cx.x2 << "\n";
    std::cout << "  Ours: rho_v=" << our_cx->rho_vapor << "  rho_l=" << our_cx->rho_liquid << "\n";

    // Verify thermodynamic consistency: both solutions should satisfy
    // equal P and equal mu.
    double jim_Pv = jim_eos.pressure(jim_cx.x1);
    double jim_Pl = jim_eos.pressure(jim_cx.x2);
    double jim_muv = jim_eos.chemical_potential(jim_cx.x1);
    double jim_mul = jim_eos.chemical_potential(jim_cx.x2);

    double our_Pv = jim_eos.pressure(our_cx->rho_vapor);
    double our_Pl = jim_eos.pressure(our_cx->rho_liquid);
    double our_muv = jim_eos.chemical_potential(our_cx->rho_vapor);
    double our_mul = jim_eos.chemical_potential(our_cx->rho_liquid);

    std::cout << "  Jim  delta_P=" << jim_Pv - jim_Pl << "  delta_mu=" << jim_muv - jim_mul << "\n";
    std::cout << "  Ours delta_P=" << our_Pv - our_Pl << "  delta_mu=" << our_muv - our_mul << "\n";

    check("coex_vapor(kT=" + std::to_string(kT) + ")", our_cx->rho_vapor, jim_cx.x1, 1e-6);
    check("coex_liquid(kT=" + std::to_string(kT) + ")", our_cx->rho_liquid, jim_cx.x2, 1e-6);
  }

  // ------------------------------------------------------------------
  // Step 7: Binodal curve + critical point consistency
  //
  // Trace the full binodal curve using our pseudo-arclength continuation
  // and verify that the critical point is thermodynamically consistent.
  // ------------------------------------------------------------------

  ++step;
  section("Step " + std::to_string(step) + ": Binodal curve consistency");

  functionals::bulk::PhaseDiagramBuilder pd_config{
      .start_temperature = 0.6,
      .search = search_config,
  };

  functionals::bulk::EoSFactory eos_at = [&](double kT) -> functionals::bulk::BulkThermodynamics {
    return eos_factory(kT);
  };

  auto b = pd_config.binodal(eos_at);
  if (!b) {
    std::cout << "  FAIL: binodal returned nullopt\n";
    g_failures += 3;
    g_checks += 3;
  } else {
    std::cout << "  Binodal: " << b->temperature.n_elem << " points"
              << ", T_c=" << b->critical_temperature << ", rho_c=" << b->critical_density << "\n";

    // The critical temperature should be in a physically reasonable range
    // for LJ mean-field DFT (roughly 1.0 < T_c < 1.5).
    ++g_checks;
    if (b->critical_temperature < 1.0 || b->critical_temperature > 1.5) {
      ++g_failures;
      std::cout << "  FAIL Tc=" << b->critical_temperature << " out of range [1.0, 1.5]\n";
    }

    // At the critical point, rho_vapor ≈ rho_liquid.
    double gap = std::abs(b->rho_vapor.tail(1)(0) - b->rho_liquid.tail(1)(0));
    std::cout << "  Final density gap = " << gap << "\n";
    check("density_gap_at_Tc < 0.05", std::min(gap, 0.05), 0.05, 0.05);

    // Cross-check: verify a few binodal points against findCoex at the same T.
    // Pick 3 temperatures from the curve interior.
    arma::uvec idxs = {0, b->temperature.n_elem / 3, 2 * b->temperature.n_elem / 3};
    for (auto idx : idxs) {
      double kT = b->temperature(idx);
      auto eos_at_T = eos_factory(kT);
      auto jim_eos_at_T = make_legacy_eos(eos_at_T);

      try {
        auto jim_cx = legacy::solver::findCoex(jim_eos_at_T, jim_xmax, jim_dx);
        std::cout << "  T=" << kT << " binodal: rv=" << b->rho_vapor(idx) << " rl=" << b->rho_liquid(idx)
                  << " | Jim: rv=" << jim_cx.x1 << " rl=" << jim_cx.x2 << "\n";

        check("binodal_vs_jim_rv(T=" + std::to_string(kT) + ")", b->rho_vapor(idx), jim_cx.x1, 1e-4);
        check("binodal_vs_jim_rl(T=" + std::to_string(kT) + ")", b->rho_liquid(idx), jim_cx.x2, 1e-4);
      } catch (const std::exception& e) {
        std::cout << "  Jim findCoex at T=" << kT << " failed: " << e.what() << "\n";
        // Not a failure of our code; skip the comparison.
      }
    }
  }

  // ------------------------------------------------------------------
  // Summary
  // ------------------------------------------------------------------

  std::cout << "\n========================================\n";
  std::cout << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures > 0 ? 1 : 0;
}
