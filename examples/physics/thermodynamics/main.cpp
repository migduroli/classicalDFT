#include <classicaldft>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numbers>

using namespace dft_core::physics::thermodynamics;

int main() {
  // ── Hard-sphere thermodynamics ──────────────────────────────────────────

  std::cout << "=== Hard-sphere fluid models ===\n\n";

  CarnahanStarling cs;
  PercusYevick py_virial(PercusYevick::Route::Virial);
  PercusYevick py_comp(PercusYevick::Route::Compressibility);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "P/rho*kT (CS)"
            << std::setw(16) << "P/rho*kT (PYv)"
            << std::setw(16) << "P/rho*kT (PYc)"
            << std::setw(16) << "chi (CS)"
            << "\n";

  std::vector<double> eta_values;
  std::vector<double> p_cs, p_pyv, p_pyc, chi_cs;

  for (double eta = 0.05; eta <= 0.49; eta += 0.05) {
    eta_values.push_back(eta);
    p_cs.push_back(cs.pressure(eta));
    p_pyv.push_back(py_virial.pressure(eta));
    p_pyc.push_back(py_comp.pressure(eta));
    chi_cs.push_back(cs.contact_value(eta));

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
  double mu = cs.chemical_potential(rho);
  double f = cs.free_energy(rho);
  double p = cs.pressure(eta);
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

  for (double density = 0.1; density <= 0.8; density += 0.1) {
    double chi = cs.contact_value(packing_fraction(density));
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

  double kT = 1.3;
  eos::IdealGas ideal(kT);
  eos::PercusYevick py_eos(kT);
  eos::LennardJonesJZG jzg(kT);
  eos::LennardJonesJZG jzg_cut(kT, 2.5);
  eos::LennardJonesMecke mecke(kT);

  std::cout << "kT = " << kT << "\n\n";
  std::cout << std::setw(8) << "rho"
            << std::setw(14) << ideal.name()
            << std::setw(14) << py_eos.name()
            << std::setw(14) << "LJ-JZG"
            << std::setw(14) << "JZG(rc=2.5)"
            << std::setw(14) << mecke.name()
            << "\n";

  std::vector<double> eos_rho;
  std::vector<double> p_ideal, p_py, p_jzg, p_jzg_cut, p_mecke;

  for (double density = 0.05; density <= 0.85; density += 0.1) {
    eos_rho.push_back(density);
    p_ideal.push_back(ideal.pressure(density));
    p_py.push_back(py_eos.pressure(density));
    p_jzg.push_back(jzg.pressure(density));
    p_jzg_cut.push_back(jzg_cut.pressure(density));
    p_mecke.push_back(mecke.pressure(density));

    std::cout << std::setw(8) << density
              << std::setw(14) << p_ideal.back()
              << std::setw(14) << p_py.back()
              << std::setw(14) << p_jzg.back()
              << std::setw(14) << p_jzg_cut.back()
              << std::setw(14) << p_mecke.back()
              << "\n";
  }

  // ── Grace plots ───────────────────────────────────────────────────────

#ifdef DFT_HAS_GRACE
  using namespace dft_core::grace_plot;

  // ── Plot 1: hard-sphere pressure ──────────────────────────────────────
  {
    auto g = Grace();
    g.set_title("Hard-sphere compressibility factor");
    g.set_label("\\xh", Axis::X);
    g.set_label("P / \\r kT", Axis::Y);

    const int N = 100;
    std::vector<double> eta_fine(N), cs_fine(N), pyv_fine(N), pyc_fine(N);
    for (int i = 0; i < N; ++i) {
      double e = 0.01 + i * 0.48 / (N - 1);
      eta_fine[i] = e;
      cs_fine[i] = cs.pressure(e);
      pyv_fine[i] = py_virial.pressure(e);
      pyc_fine[i] = py_comp.pressure(e);
    }

    auto ds_cs = g.add_dataset(eta_fine, cs_fine);
    g.set_color(Color::BLACK, ds_cs);
    g.set_legend("Carnahan-Starling", ds_cs);

    auto ds_pyv = g.add_dataset(eta_fine, pyv_fine);
    g.set_color(Color::RED, ds_pyv);
    g.set_line_type(LineStyle::DASHEDLINE_EN, ds_pyv);
    g.set_legend("PY (virial)", ds_pyv);

    auto ds_pyc = g.add_dataset(eta_fine, pyc_fine);
    g.set_color(Color::BLUE, ds_pyc);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_pyc);
    g.set_legend("PY (compressibility)", ds_pyc);

    g.set_x_limits(0.0, 0.5);
    g.set_y_limits(0.0, 30.0);
    g.set_ticks(0.1, 5.0);
    g.print_to_file("exports/hs_pressure.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }

  // ── Plot 2: contact value ─────────────────────────────────────────────
  {
    auto g = Grace();
    g.set_title("Pair correlation at contact");
    g.set_label("\\xh", Axis::X);
    g.set_label("\\xc\\f{}(\\xh\\f{})", Axis::Y);

    const int N = 100;
    std::vector<double> eta_fine(N), chi_cs_fine(N), chi_pyv_fine(N), chi_pyc_fine(N);
    for (int i = 0; i < N; ++i) {
      double e = 0.01 + i * 0.48 / (N - 1);
      eta_fine[i] = e;
      chi_cs_fine[i] = cs.contact_value(e);
      chi_pyv_fine[i] = py_virial.contact_value(e);
      chi_pyc_fine[i] = py_comp.contact_value(e);
    }

    auto ds_cs = g.add_dataset(eta_fine, chi_cs_fine);
    g.set_color(Color::BLACK, ds_cs);
    g.set_legend("Carnahan-Starling", ds_cs);

    auto ds_pyv = g.add_dataset(eta_fine, chi_pyv_fine);
    g.set_color(Color::RED, ds_pyv);
    g.set_line_type(LineStyle::DASHEDLINE_EN, ds_pyv);
    g.set_legend("PY (virial)", ds_pyv);

    auto ds_pyc = g.add_dataset(eta_fine, chi_pyc_fine);
    g.set_color(Color::BLUE, ds_pyc);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_pyc);
    g.set_legend("PY (compressibility)", ds_pyc);

    g.set_x_limits(0.0, 0.5);
    g.set_y_limits(0.0, 20.0);
    g.set_ticks(0.1, 5.0);
    g.print_to_file("exports/contact_value.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }

  // ── Plot 3: transport coefficients ────────────────────────────────────
  {
    auto g = Grace();
    g.set_title("Enskog transport coefficients (d = kT = 1)");
    g.set_label("\\r", Axis::X);
    g.set_label("Transport coefficient", Axis::Y);

    const int N = 80;
    std::vector<double> rho_fine(N), sh(N), bk(N), th(N), sd(N);
    for (int i = 0; i < N; ++i) {
      double d = 0.01 + i * 0.79 / (N - 1);
      double chi = cs.contact_value(packing_fraction(d));
      rho_fine[i] = d;
      sh[i] = transport::shear_viscosity(d, chi);
      bk[i] = transport::bulk_viscosity(d, chi);
      th[i] = transport::thermal_conductivity(d, chi);
      sd[i] = transport::sound_damping(d, chi);
    }

    auto ds_sh = g.add_dataset(rho_fine, sh);
    g.set_color(Color::BLACK, ds_sh);
    g.set_legend("Shear viscosity", ds_sh);

    auto ds_bk = g.add_dataset(rho_fine, bk);
    g.set_color(Color::RED, ds_bk);
    g.set_line_type(LineStyle::DASHEDLINE_EN, ds_bk);
    g.set_legend("Bulk viscosity", ds_bk);

    auto ds_th = g.add_dataset(rho_fine, th);
    g.set_color(Color::BLUE, ds_th);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_th);
    g.set_legend("Thermal conductivity", ds_th);

    auto ds_sd = g.add_dataset(rho_fine, sd);
    g.set_color(Color::DARKGREEN, ds_sd);
    g.set_line_type(LineStyle::DOTTEDDASHEDLINE_EN, ds_sd);
    g.set_legend("Sound damping", ds_sd);

    g.set_x_limits(0.0, 0.8);
    g.set_ticks(0.1, 0.5);
    g.print_to_file("exports/transport.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }

  // ── Plot 4: equations of state comparison ─────────────────────────────
  {
    auto g = Grace();
    g.set_title("Equations of state (kT = 1.3)");
    g.set_label("\\r", Axis::X);
    g.set_label("P / \\r kT", Axis::Y);

    const int N = 100;
    std::vector<double> rho_fine(N), p_jzg_f(N), p_jzg_cut_f(N), p_mecke_f(N), p_py_f(N);
    for (int i = 0; i < N; ++i) {
      double d = 0.01 + i * 0.84 / (N - 1);
      rho_fine[i] = d;
      p_jzg_f[i] = jzg.pressure(d);
      p_jzg_cut_f[i] = jzg_cut.pressure(d);
      p_mecke_f[i] = mecke.pressure(d);
      p_py_f[i] = py_eos.pressure(d);
    }

    auto ds_jzg = g.add_dataset(rho_fine, p_jzg_f);
    g.set_color(Color::BLACK, ds_jzg);
    g.set_legend("LJ-JZG", ds_jzg);

    auto ds_jzg_cut = g.add_dataset(rho_fine, p_jzg_cut_f);
    g.set_color(Color::RED, ds_jzg_cut);
    g.set_line_type(LineStyle::DASHEDLINE_EN, ds_jzg_cut);
    g.set_legend("JZG (r\\sc\\N = 2.5)", ds_jzg_cut);

    auto ds_mecke = g.add_dataset(rho_fine, p_mecke_f);
    g.set_color(Color::BLUE, ds_mecke);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_mecke);
    g.set_legend("Mecke", ds_mecke);

    auto ds_py_eos = g.add_dataset(rho_fine, p_py_f);
    g.set_color(Color::DARKGREEN, ds_py_eos);
    g.set_line_type(LineStyle::DOTTEDDASHEDLINE_EN, ds_py_eos);
    g.set_legend("PY (HS)", ds_py_eos);

    g.set_x_limits(0.0, 0.85);
    g.set_ticks(0.1, 2.0);
    g.print_to_file("exports/eos_comparison.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }
#endif
}
