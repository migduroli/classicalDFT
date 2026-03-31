#include "dft.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft::thermodynamics;

int main(int argc, char* argv[]) {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  std::string config_path = (argc > 1) ? argv[1] : "config.ini";
  auto cfg = dft::config::ConfigParser(config_path);

  // ── Hard-sphere thermodynamics ──────────────────────────────────────────

  std::cout << "=== Hard-sphere fluid models ===\n\n";

  CarnahanStarling cs;
  PercusYevickVirial py_virial;
  PercusYevickCompressibility py_comp;

  HardSphereModel cs_model = cs;
  HardSphereModel pyv_model = py_virial;
  HardSphereModel pyc_model = py_comp;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "P/rho*kT (CS)"
            << std::setw(16) << "P/rho*kT (PYv)"
            << std::setw(16) << "P/rho*kT (PYc)"
            << std::setw(16) << "chi (CS)"
            << "\n";

  std::vector<double> eta_values;
  std::vector<double> p_cs, p_pyv, p_pyc, chi_cs;

  for (double eta = cfg.get<double>("hard_sphere.eta_min"); eta <= cfg.get<double>("hard_sphere.eta_max");
       eta += cfg.get<double>("hard_sphere.eta_step")) {
    eta_values.push_back(eta);
    p_cs.push_back(hs_pressure(cs_model, eta));
    p_pyv.push_back(hs_pressure(pyv_model, eta));
    p_pyc.push_back(hs_pressure(pyc_model, eta));
    chi_cs.push_back(contact_value(eta));

    std::cout << std::setw(8) << eta
              << std::setw(16) << p_cs.back()
              << std::setw(16) << p_pyv.back()
              << std::setw(16) << p_pyc.back()
              << std::setw(16) << chi_cs.back()
              << "\n";
  }

  // ── Chemical potential and free energy ─────────────────────────────────

  std::cout << "\n=== Thermodynamic consistency check (CS, eta=0.3) ===\n";
  double rho = density_from_eta(0.3);
  double eta = 0.3;
  double mu = hs_chemical_potential(cs_model, rho);
  double f = hs_free_energy(cs_model, rho);
  double p = hs_pressure(cs_model, eta);
  std::cout << "  mu/kT         = " << mu << "\n";
  std::cout << "  f/kT          = " << f << "\n";
  std::cout << "  P/(rho*kT)    = " << p << "\n";
  std::cout << "  mu - f - P    = " << mu - f - p
            << "  (should be 0)\n";

  // ── Transport coefficients ────────────────────────────────────────────

  std::cout << "\n=== Enskog transport coefficients (d=kT=1) ===\n\n";
  std::cout << std::setw(8) << "rho"
            << std::setw(14) << "eta_shear"
            << std::setw(14) << "eta_bulk"
            << std::setw(14) << "lambda"
            << std::setw(14) << "Gamma"
            << "\n";

  std::vector<double> rho_values;
  std::vector<double> shear_v, bulk_v, thermal_v, damping_v;

  for (double density = cfg.get<double>("transport.rho_min"); density <= cfg.get<double>("transport.rho_max");
       density += cfg.get<double>("transport.rho_step")) {
    double chi = contact_value(packing_fraction(density));
    rho_values.push_back(density);
    shear_v.push_back(transport::shear_viscosity(density, chi));
    bulk_v.push_back(transport::bulk_viscosity(density, chi));
    thermal_v.push_back(transport::thermal_conductivity(density, chi));
    damping_v.push_back(transport::sound_damping(density, chi));

    std::cout << std::setw(8) << density
              << std::setw(14) << shear_v.back()
              << std::setw(14) << bulk_v.back()
              << std::setw(14) << thermal_v.back()
              << std::setw(14) << damping_v.back()
              << "\n";
  }

  // ── Equations of state ────────────────────────────────────────────────

  std::cout << "\n=== Equations of state ===\n\n";

  double kT = cfg.get<double>("thermodynamics.kT");
  eos::EosModel ideal = eos::IdealGas(kT);
  eos::EosModel py_eos = eos::PercusYevick(kT);
  eos::EosModel jzg = eos::LennardJonesJZG(kT);
  eos::EosModel jzg_cut = eos::LennardJonesJZG(kT, 2.5);
  eos::EosModel mecke = eos::LennardJonesMecke(kT);

  std::cout << "kT = " << kT << "\n\n";
  std::cout << std::setw(8) << "rho"
            << std::setw(14) << eos::eos_name(ideal)
            << std::setw(14) << eos::eos_name(py_eos)
            << std::setw(14) << "LJ-JZG"
            << std::setw(14) << "JZG(rc=2.5)"
            << std::setw(14) << eos::eos_name(mecke)
            << "\n";

  std::vector<double> eos_rho;
  std::vector<double> p_ideal, p_py, p_jzg, p_jzg_cut, p_mecke;

  for (double density = cfg.get<double>("eos.rho_min"); density <= cfg.get<double>("eos.rho_max");
       density += cfg.get<double>("eos.rho_step")) {
    eos_rho.push_back(density);
    p_ideal.push_back(eos::eos_pressure(ideal, density));
    p_py.push_back(eos::eos_pressure(py_eos, density));
    p_jzg.push_back(eos::eos_pressure(jzg, density));
    p_jzg_cut.push_back(eos::eos_pressure(jzg_cut, density));
    p_mecke.push_back(eos::eos_pressure(mecke, density));

    std::cout << std::setw(8) << density
              << std::setw(14) << p_ideal.back()
              << std::setw(14) << p_py.back()
              << std::setw(14) << p_jzg.back()
              << std::setw(14) << p_jzg_cut.back()
              << std::setw(14) << p_mecke.back()
              << "\n";
  }

  // ── Plots ──────────────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;
  plt::backend("Agg");

  // ── Plot 1: hard-sphere pressure ──────────────────────────────────────
  {
    const int N = 100;
    std::vector<double> eta_fine(N), cs_fine(N), pyv_fine(N), pyc_fine(N);
    for (int i = 0; i < N; ++i) {
      double e = 0.01 + i * 0.48 / (N - 1);
      eta_fine[i] = e;
      cs_fine[i] = hs_pressure(cs_model, e);
      pyv_fine[i] = hs_pressure(pyv_model, e);
      pyc_fine[i] = hs_pressure(pyc_model, e);
    }

    plt::figure_size(800, 550);
    plt::named_plot("Carnahan\u2013Starling", eta_fine, cs_fine, "k-");
    plt::named_plot("PY (virial)", eta_fine, pyv_fine, "r--");
    plt::named_plot("PY (compressibility)", eta_fine, pyc_fine, "b:");
    plt::xlim(0.0, 0.5);
    plt::ylim(0.0, 30.0);
    plt::xlabel(R"($\eta$)");
    plt::ylabel(R"($P / \rho k_BT$)");
    plt::title("Hard-sphere compressibility factor");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/hs_pressure.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/hs_pressure.png") << std::endl;
  }

  // ── Plot 2: contact value ─────────────────────────────────────────────
  {
    const int N = 100;
    std::vector<double> eta_fine(N), chi_cs_fine(N), chi_pyv_fine(N), chi_pyc_fine(N);
    for (int i = 0; i < N; ++i) {
      double e = 0.01 + i * 0.48 / (N - 1);
      eta_fine[i] = e;
      chi_cs_fine[i] = (hs_pressure(cs_model, e) - 1.0) / (4.0 * e);
      chi_pyv_fine[i] = (hs_pressure(pyv_model, e) - 1.0) / (4.0 * e);
      chi_pyc_fine[i] = (hs_pressure(pyc_model, e) - 1.0) / (4.0 * e);
    }

    plt::figure_size(800, 550);
    plt::named_plot("Carnahan\u2013Starling", eta_fine, chi_cs_fine, "k-");
    plt::named_plot("PY (virial)", eta_fine, chi_pyv_fine, "r--");
    plt::named_plot("PY (compressibility)", eta_fine, chi_pyc_fine, "b:");
    plt::xlim(0.0, 0.5);
    plt::ylim(0.0, 20.0);
    plt::xlabel(R"($\eta$)");
    plt::ylabel(R"($\chi(\eta)$)");
    plt::title("Pair correlation at contact");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/contact_value.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/contact_value.png") << std::endl;
  }

  // ── Plot 3: transport coefficients ────────────────────────────────────
  {
    const int N = 80;
    std::vector<double> rho_fine(N), sh(N), bk(N), th(N), sd(N);
    for (int i = 0; i < N; ++i) {
      double d = 0.01 + i * 0.79 / (N - 1);
      double chi = contact_value(packing_fraction(d));
      rho_fine[i] = d;
      sh[i] = transport::shear_viscosity(d, chi);
      bk[i] = transport::bulk_viscosity(d, chi);
      th[i] = transport::thermal_conductivity(d, chi);
      sd[i] = transport::sound_damping(d, chi);
    }

    plt::figure_size(800, 550);
    plt::named_plot(R"($\eta_\mathrm{shear}$)", rho_fine, sh, "k-");
    plt::named_plot(R"($\eta_\mathrm{bulk}$)", rho_fine, bk, "r--");
    plt::xlim(0.0, 0.8);
    plt::xlabel(R"($\rho\sigma^3$)");
    plt::ylabel(R"(Viscosity / $(m k_BT)^{1/2} \sigma^{-2}$)");
    plt::title(R"(Enskog viscosities ($d = k_BT = 1$))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/transport_viscosity.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/transport_viscosity.png") << std::endl;

    plt::figure_size(800, 550);
    plt::named_plot(R"($\lambda$ (thermal conductivity))", rho_fine, th, "b-");
    plt::named_plot(R"($\Gamma$ (sound damping))", rho_fine, sd, "g--");
    plt::xlim(0.0, 0.8);
    plt::xlabel(R"($\rho\sigma^3$)");
    plt::ylabel("Transport coefficient");
    plt::title(R"(Thermal conductivity and sound damping ($d = k_BT = 1$))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/transport_thermal.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/transport_thermal.png") << std::endl;
  }

  // ── Plot 4: equations of state comparison ─────────────────────────────
  {
    const int N = 100;
    std::vector<double> rho_fine(N), p_jzg_f(N), p_jzg_cut_f(N), p_mecke_f(N), p_py_f(N);
    for (int i = 0; i < N; ++i) {
      double d = 0.01 + i * 0.84 / (N - 1);
      rho_fine[i] = d;
      p_jzg_f[i] = eos::eos_pressure(jzg, d);
      p_jzg_cut_f[i] = eos::eos_pressure(jzg_cut, d);
      p_mecke_f[i] = eos::eos_pressure(mecke, d);
      p_py_f[i] = eos::eos_pressure(py_eos, d);
    }

    plt::figure_size(800, 550);
    plt::named_plot("LJ-JZG", rho_fine, p_jzg_f, "k-");
    plt::named_plot(R"(JZG ($r_c = 2.5$))", rho_fine, p_jzg_cut_f, "r--");
    plt::named_plot("Mecke", rho_fine, p_mecke_f, "b:");
    plt::named_plot("PY (HS)", rho_fine, p_py_f, "g-.");
    plt::xlim(0.0, 0.85);
    plt::xlabel(R"($\rho$)");
    plt::ylabel(R"($P / \rho k_BT$)");
    plt::title(R"(Equations of state ($k_BT = 1.3$))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/eos_comparison.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/eos_comparison.png") << std::endl;
  }
#endif
}
