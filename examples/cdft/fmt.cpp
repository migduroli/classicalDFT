// ── FMT functional example (modern cdft API) ────────────────────────────────
//
// Demonstrates:
//   - FMT model structs as std::variant (Rosenfeld, RSLT, WBI, WBII)
//   - Factor functions: f1, f2, f3 and their autodiff derivatives
//   - Measures from uniform density
//   - Phi evaluation and bulk thermodynamics
//   - Free-function dispatch through FMTModel variant

#include <cdft.hpp>

#include <iomanip>
#include <iostream>

int main() {
  using namespace cdft::functional;
  using namespace cdft::physics;

  // ── FMT factor functions across models ──────────────────────────────────

  std::cout << std::fixed << std::setprecision(8);
  std::cout << "FMT factor functions at eta = 0.3\n";
  std::cout << std::string(72, '-') << "\n";
  std::cout << std::setw(14) << "Model" << std::setw(16) << "f1(eta)" << std::setw(16) << "f2(eta)"
            << std::setw(16) << "f3(eta)\n";

  constexpr double eta = 0.3;

  std::cout << std::setw(14) << "Rosenfeld"
            << std::setw(16) << Rosenfeld::ideal_factor(eta)
            << std::setw(16) << Rosenfeld::pair_factor(eta)
            << std::setw(16) << Rosenfeld::triplet_factor(eta) << "\n";

  std::cout << std::setw(14) << "RSLT"
            << std::setw(16) << RSLT::ideal_factor(eta)
            << std::setw(16) << RSLT::pair_factor(eta)
            << std::setw(16) << RSLT::triplet_factor(eta) << "\n";

  std::cout << std::setw(14) << "WhiteBearI"
            << std::setw(16) << WhiteBearI::ideal_factor(eta)
            << std::setw(16) << WhiteBearI::pair_factor(eta)
            << std::setw(16) << WhiteBearI::triplet_factor(eta) << "\n";

  std::cout << std::setw(14) << "WhiteBearII"
            << std::setw(16) << WhiteBearII::ideal_factor(eta)
            << std::setw(16) << WhiteBearII::pair_factor(eta)
            << std::setw(16) << WhiteBearII::triplet_factor(eta) << "\n";

  // ── Autodiff on FMT factors ─────────────────────────────────────────────

  std::cout << "\n\nAutodiff derivatives of WhiteBearII factors at eta = 0.3\n";
  std::cout << std::string(60, '-') << "\n";

  auto [f1, df1, d2f1, d3f1] = cdft::derivatives_up_to_3(
      [](cdft::dual3rd e) { return WhiteBearII::ideal_factor(e); }, eta);
  auto [f2, df2, d2f2, d3f2] = cdft::derivatives_up_to_3(
      [](cdft::dual3rd e) { return WhiteBearII::pair_factor(e); }, eta);
  auto [f3, df3, d2f3, d3f3] = cdft::derivatives_up_to_3(
      [](cdft::dual3rd e) { return WhiteBearII::triplet_factor(e); }, eta);

  std::cout << std::setw(12) << "Factor" << std::setw(14) << "f" << std::setw(14) << "f'"
            << std::setw(14) << "f''" << std::setw(14) << "f'''\n";
  std::cout << std::setw(12) << "f1" << std::setw(14) << f1 << std::setw(14) << df1
            << std::setw(14) << d2f1 << std::setw(14) << d3f1 << "\n";
  std::cout << std::setw(12) << "f2" << std::setw(14) << f2 << std::setw(14) << df2
            << std::setw(14) << d2f2 << std::setw(14) << d3f2 << "\n";
  std::cout << std::setw(12) << "f3" << std::setw(14) << f3 << std::setw(14) << df3
            << std::setw(14) << d2f3 << std::setw(14) << d3f3 << "\n";

  // ── Variant dispatch ────────────────────────────────────────────────────

  std::cout << "\n\nVariant dispatch: bulk free-energy density at rho=0.5, d=1.0\n";
  std::cout << std::string(60, '-') << "\n";
  std::cout << std::setw(14) << "Model" << std::setw(20) << "f_bulk(rho)"
            << std::setw(20) << "mu_exc(rho)\n";

  constexpr double rho = 0.5;
  constexpr double d = 1.0;

  for (const FMTModel& model : {FMTModel{Rosenfeld{}}, FMTModel{RSLT{}},
                                 FMTModel{WhiteBearI{}}, FMTModel{WhiteBearII{}}}) {
    std::cout << std::setw(14) << fmt_name(model)
              << std::setw(20) << fmt_bulk_free_energy_density(model, rho, d)
              << std::setw(20) << fmt_bulk_excess_chemical_potential(model, rho, d) << "\n";
  }

  // ── Measures for a uniform fluid ────────────────────────────────────────

  std::cout << "\n\nFMT measures for uniform fluid: rho = 0.5, d = 1.0\n";
  std::cout << std::string(40, '-') << "\n";

  auto m = Measures<>::uniform(rho, d);

  std::cout << "  eta    = " << m.eta << "\n";
  std::cout << "  n0     = " << m.n0 << "\n";
  std::cout << "  n1     = " << m.n1 << "\n";
  std::cout << "  n2     = " << m.n2 << "\n";
  std::cout << "  |v2|^2 = " << m.contractions.norm_v2_squared << "\n";
  std::cout << "  Tr(T)  = " << arma::trace(m.T) << "\n";

  // ── Phi evaluation ──────────────────────────────────────────────────────

  std::cout << "\n\nPhi(measures) at uniform rho = 0.5, d = 1.0\n";
  std::cout << std::string(40, '-') << "\n";

  for (const FMTModel& model : {FMTModel{Rosenfeld{}}, FMTModel{RSLT{}},
                                 FMTModel{WhiteBearI{}}, FMTModel{WhiteBearII{}}}) {
    std::cout << "  " << std::setw(12) << fmt_name(model) << " : Phi = " << fmt_phi(model, m) << "\n";
  }

  // ── EOS cross-validation: CS vs FMT bulk ────────────────────────────────

  std::cout << "\n\nCross-validation: Carnahan-Starling vs FMT bulk free energy\n";
  std::cout << std::string(60, '-') << "\n";
  std::cout << std::setw(8) << "rho" << std::setw(18) << "CS f_exc*rho"
            << std::setw(18) << "WBI f_bulk" << std::setw(14) << "diff\n";

  auto wbi = FMTModel{WhiteBearI{}};

  for (double density : {0.1, 0.2, 0.3, 0.4, 0.5}) {
    double eta_val = packing_fraction(density);
    double cs_bulk = density * CarnahanStarling::excess_free_energy(eta_val);
    double fmt_bulk = fmt_bulk_free_energy_density(wbi, density, d);

    std::cout << std::setw(8) << density
              << std::setw(18) << cs_bulk
              << std::setw(18) << fmt_bulk
              << std::setw(14) << std::abs(cs_bulk - fmt_bulk) << "\n";
  }

  return 0;
}
