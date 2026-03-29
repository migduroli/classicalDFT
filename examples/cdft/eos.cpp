// ── Equation-of-state example (modern cdft API) ─────────────────────────────
//
// Demonstrates:
//   - Hard-sphere models as std::variant (CarnahanStarling, PY variants)
//   - Full EOS models (LennardJonesJZG, LennardJonesMecke)
//   - Automatic differentiation: derivatives from a single templated function
//   - Transport coefficients via Enskog theory

#include <cdft.hpp>

#include <iomanip>
#include <iostream>

int main() {
  using namespace cdft::physics;

  // ── Hard-sphere models ──────────────────────────────────────────────────

  auto cs = CarnahanStarling{};
  auto pyv = PercusYevickVirial{};
  auto pyc = PercusYevickCompressibility{};

  std::cout << std::fixed << std::setprecision(8);
  std::cout << "Hard-sphere excess free energy at eta = 0.3\n\n";
  std::cout << std::setw(30) << "Model" << std::setw(18) << "f_exc(eta)\n";
  std::cout << std::string(48, '-') << "\n";

  for (double eta : {0.1, 0.2, 0.3, 0.4}) {
    std::cout << "\n  eta = " << eta << "\n";
    std::cout << std::setw(30) << "Carnahan-Starling" << std::setw(18) << cs.excess_free_energy(eta) << "\n";
    std::cout << std::setw(30) << "PY (virial)" << std::setw(18) << pyv.excess_free_energy(eta) << "\n";
    std::cout << std::setw(30) << "PY (compressibility)" << std::setw(18) << pyc.excess_free_energy(eta) << "\n";
  }

  // ── Variant dispatch ────────────────────────────────────────────────────

  std::cout << "\n\nVariant dispatch: pressure and chemical potential\n";
  std::cout << std::string(64, '-') << "\n";
  std::cout << std::setw(30) << "Model" << std::setw(18) << "P(eta=0.3)" << std::setw(18) << "mu(rho=0.5)\n";

  for (const HardSphereModel& model : {HardSphereModel{cs}, HardSphereModel{pyv}, HardSphereModel{pyc}}) {
    auto name = std::visit([](const auto& m) { return m.name(); }, model);
    std::cout << std::setw(30) << name
              << std::setw(18) << hs_pressure(model, 0.3)
              << std::setw(18) << hs_chemical_potential(model, 0.5) << "\n";
  }

  // ── Autodiff: derivatives from a single function ────────────────────────

  std::cout << "\n\nAutodiff derivatives of Carnahan-Starling at eta = 0.3\n";
  std::cout << std::string(48, '-') << "\n";

  auto [f, df, d2f, d3f] = cdft::derivatives_up_to_3(
      [](cdft::dual3rd eta) { return CarnahanStarling::excess_free_energy(eta); },
      0.3
  );

  std::cout << "  f(0.3)    = " << f << "\n";
  std::cout << "  f'(0.3)   = " << df << "\n";
  std::cout << "  f''(0.3)  = " << d2f << "\n";
  std::cout << "  f'''(0.3) = " << d3f << "\n";

  // ── Full EOS: Lennard-Jones models ──────────────────────────────────────

  constexpr double kT = 1.2;
  auto jzg = LennardJonesJZG(kT);
  auto mecke = LennardJonesMecke(kT);

  std::cout << "\n\nLennard-Jones EOS at kT = " << kT << "\n";
  std::cout << std::string(64, '-') << "\n";
  std::cout << std::setw(10) << "rho" << std::setw(18) << "JZG f_exc/N" << std::setw(18) << "Mecke f_exc/N"
            << std::setw(18) << "JZG P" << "\n";

  for (double rho : {0.1, 0.3, 0.5, 0.7, 0.85}) {
    auto eos_jzg = EquationOfState{jzg};
    std::cout << std::setw(10) << rho
              << std::setw(18) << jzg.excess_free_energy_per_particle(rho)
              << std::setw(18) << mecke.excess_free_energy_per_particle(rho)
              << std::setw(18) << eos_pressure(eos_jzg, rho) << "\n";
  }

  // ── Transport coefficients ──────────────────────────────────────────────

  std::cout << "\n\nTransport coefficients (Enskog, d = kT = 1)\n";
  std::cout << std::string(72, '-') << "\n";
  std::cout << std::setw(10) << "rho" << std::setw(12) << "chi"
            << std::setw(14) << "bulk_visc" << std::setw(14) << "shear_visc"
            << std::setw(14) << "kappa" << std::setw(14) << "Gamma_s\n";

  for (double rho : {0.1, 0.3, 0.5, 0.7}) {
    double eta = packing_fraction(rho);
    double chi = contact_value(eta);
    std::cout << std::setw(10) << rho
              << std::setw(12) << chi
              << std::setw(14) << transport::bulk_viscosity(rho, chi)
              << std::setw(14) << transport::shear_viscosity(rho, chi)
              << std::setw(14) << transport::thermal_conductivity(rho, chi)
              << std::setw(14) << transport::sound_damping(rho, chi) << "\n";
  }

  return 0;
}
