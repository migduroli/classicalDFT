#include <classicaldft>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

using namespace dft_core::physics::fmt;
using namespace dft_core::physics::density;
using namespace dft_core::physics::thermodynamics;

int main() {
  std::filesystem::create_directories("exports");

  // ── 1. Bulk free energy: FMT models vs known EOS ─────────────────────

  std::cout << "=== Bulk excess free energy per particle: FMT vs EOS ===\n\n";
  std::cout << std::fixed << std::setprecision(8);
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "Rosenfeld"
            << std::setw(16) << "PY(comp)"
            << std::setw(16) << "WhiteBearI"
            << std::setw(16) << "CS"
            << "\n";

  Rosenfeld ros;
  RSLT rslt;
  WhiteBearI wb1;
  WhiteBearII wb2;
  CarnahanStarling cs;
  PercusYevick py_comp(PercusYevick::Route::Compressibility);
  PercusYevick py_vir(PercusYevick::Route::Virial);

  const int N = 200;
  std::vector<double> eta_vec(N), f_ros(N), f_rslt(N), f_wb1(N), f_wb2(N);
  std::vector<double> f_cs(N), f_pyc(N), f_pyv(N);

  for (int i = 0; i < N; ++i) {
    double eta = 0.005 + 0.49 * i / (N - 1);
    double rho = density_from_eta(eta);
    eta_vec[i] = eta;

    f_ros[i] = ros.bulk_free_energy_density(rho, 1.0) / rho;
    f_rslt[i] = rslt.bulk_free_energy_density(rho, 1.0) / rho;
    f_wb1[i] = wb1.bulk_free_energy_density(rho, 1.0) / rho;
    f_wb2[i] = wb2.bulk_free_energy_density(rho, 1.0) / rho;
    f_cs[i] = cs.excess_free_energy(eta);
    f_pyc[i] = py_comp.excess_free_energy(eta);
    f_pyv[i] = py_vir.excess_free_energy(eta);

    if (i % 40 == 0) {
      std::cout << std::setw(8) << eta
                << std::setw(16) << f_ros[i]
                << std::setw(16) << f_pyc[i]
                << std::setw(16) << f_wb1[i]
                << std::setw(16) << f_cs[i]
                << "\n";
    }
  }

  // ── 2. Bulk excess chemical potential ─────────────────────────────────

  std::cout << "\n=== Bulk excess chemical potential ===\n\n";
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "Rosenfeld"
            << std::setw(16) << "WhiteBearI"
            << std::setw(16) << "WhiteBearII"
            << "\n";

  std::vector<double> mu_ros(N), mu_wb1(N), mu_wb2(N);
  for (int i = 0; i < N; ++i) {
    double rho = density_from_eta(eta_vec[i]);
    mu_ros[i] = ros.bulk_excess_chemical_potential(rho, 1.0);
    mu_wb1[i] = wb1.bulk_excess_chemical_potential(rho, 1.0);
    mu_wb2[i] = wb2.bulk_excess_chemical_potential(rho, 1.0);

    if (i % 40 == 0) {
      std::cout << std::setw(8) << eta_vec[i]
                << std::setw(16) << mu_ros[i]
                << std::setw(16) << mu_wb1[i]
                << std::setw(16) << mu_wb2[i]
                << "\n";
    }
  }

  // ── 3. FMT species: free energy on a grid ─────────────────────────────

  std::cout << "\n=== FMT species on 3D grid ===\n";

  double dx = 0.1;
  arma::rowvec3 box = {1.6, 1.6, 1.6};
  double rho0 = 0.5;
  double eta0 = packing_fraction(rho0);

  Density dens(dx, box);
  dens.values().fill(rho0);
  Species sp(std::move(dens), 1.0);

  std::cout << "Grid: " << sp.density().shape()[0] << "^3"
            << ", dx = " << dx
            << ", rho = " << rho0
            << ", eta = " << eta0
            << "\n";

  // Free energy only (no forces)
  double F_ros = sp.compute_free_energy(ros);
  double F_wb1 = sp.compute_free_energy(wb1);

  double V = box(0) * box(1) * box(2);
  std::cout << "F_ex (Rosenfeld):   " << F_ros << "  (bulk: " << ros.bulk_free_energy_density(rho0, 1.0) * V << ")\n";
  std::cout << "F_ex (WhiteBearI):  " << F_wb1 << "  (bulk: " << wb1.bulk_free_energy_density(rho0, 1.0) * V << ")\n";

  // Forces (separate call)
  sp.zero_force();
  double F_from_forces = sp.compute_forces(ros);
  double mean_force = arma::mean(sp.force());
  double mu_ex = ros.bulk_excess_chemical_potential(rho0, 1.0);
  double dV = sp.density().cell_volume();
  std::cout << "Mean force / dV:    " << mean_force / dV << "  (mu_ex: " << mu_ex << ")\n";
  std::cout << "Force uniformity: max|f - mean| = " << arma::max(arma::abs(sp.force() - mean_force)) << "\n";

  // ── 4. Bounded alias demonstration ────────────────────────────────────

  std::cout << "\n=== Bounded alias ===\n";
  double d = 1.0;
  double rho_max = 0.9999 * 6.0 / (std::numbers::pi * d * d * d);
  std::cout << "rho_min = " << dft_core::physics::species::Species::RHO_MIN << "\n";
  std::cout << "rho_max = " << rho_max << " (eta_max = 0.9999)\n";

  arma::vec alias = sp.density_alias();
  sp.set_density_from_alias(alias);
  double roundtrip_err = arma::max(arma::abs(sp.density().values() - rho0));
  std::cout << "Alias round-trip max error: " << roundtrip_err << "\n";

  arma::vec extreme(sp.density().size(), arma::fill::value(1e6));
  sp.set_density_from_alias(extreme);
  double max_rho = arma::max(sp.density().values());
  std::cout << "Extreme alias (y=1e6): rho_max = " << max_rho
            << ", eta = " << packing_fraction(max_rho) << "\n";

  // ── 5. Pressure from FMT: thermodynamic consistency check ─────────

  std::cout << "\n=== Pressure: P/(rho kT) from mu - f ===\n\n";
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "PY(comp)"
            << std::setw(16) << "Rosenfeld"
            << std::setw(16) << "CS"
            << std::setw(16) << "WhiteBearI"
            << "\n";

  std::vector<double> p_ros(N), p_wb1(N), p_cs_vec(N), p_pyc(N);
  for (int i = 0; i < N; ++i) {
    double eta = eta_vec[i];
    double rho = density_from_eta(eta);

    // P / (rho kT) = 1 + rho * (mu_ex - f_ex)
    p_ros[i] = 1.0 + rho * (mu_ros[i] - f_ros[i]);
    p_wb1[i] = 1.0 + rho * (mu_wb1[i] - f_wb1[i]);
    p_cs_vec[i] = cs.pressure(eta);
    p_pyc[i] = py_comp.pressure(eta);

    if (i % 40 == 0) {
      std::cout << std::setw(8) << eta
                << std::setw(16) << p_pyc[i]
                << std::setw(16) << p_ros[i]
                << std::setw(16) << p_cs_vec[i]
                << std::setw(16) << p_wb1[i]
                << "\n";
    }
  }

  // ── Grace plots ───────────────────────────────────────────────────────

#ifdef DFT_HAS_GRACE
  using namespace dft_core::grace_plot;

  // ── Plot 1: excess free energy comparison ─────────────────────────────
  {
    auto g = Grace();
    g.set_title("Excess free energy per particle: FMT models vs EOS");
    g.set_label("\\xh", Axis::X);
    g.set_label("f\\sex\\N / kT", Axis::Y);

    auto ds_ros = g.add_dataset(eta_vec, f_ros);
    g.set_color(Color::BLACK, ds_ros);
    g.set_legend("Rosenfeld (= PY comp.)", ds_ros);

    auto ds_pyv = g.add_dataset(eta_vec, f_pyv);
    g.set_color(Color::RED, ds_pyv);
    g.set_line_type(LineStyle::DASHEDLINE_EN, ds_pyv);
    g.set_legend("PY (virial)", ds_pyv);

    auto ds_wb1 = g.add_dataset(eta_vec, f_wb1);
    g.set_color(Color::BLUE, ds_wb1);
    g.set_legend("White Bear I (= CS)", ds_wb1);

    auto ds_wb2 = g.add_dataset(eta_vec, f_wb2);
    g.set_color(Color::DARKGREEN, ds_wb2);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_wb2);
    g.set_legend("White Bear II", ds_wb2);

    g.set_x_limits(0.0, 0.5);
    g.set_y_limits(0.0, 8.0);
    g.set_ticks(0.1, 1.0);
    g.print_to_file("exports/fmt_free_energy.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }

  // ── Plot 2: pressure comparison ───────────────────────────────────────
  {
    auto g = Grace();
    g.set_title("Compressibility factor: FMT models vs exact EOS");
    g.set_label("\\xh", Axis::X);
    g.set_label("P / (\\xr\\f{} kT)", Axis::Y);

    auto ds_ros = g.add_dataset(eta_vec, p_ros);
    g.set_color(Color::BLACK, ds_ros);
    g.set_legend("Rosenfeld (= PY comp.)", ds_ros);

    auto ds_pyc = g.add_dataset(eta_vec, p_pyc);
    g.set_color(Color::RED, ds_pyc);
    g.set_line_type(LineStyle::DASHEDLINE_EN, ds_pyc);
    g.set_legend("PY (comp.) exact", ds_pyc);

    auto ds_wb1 = g.add_dataset(eta_vec, p_wb1);
    g.set_color(Color::BLUE, ds_wb1);
    g.set_legend("White Bear I (= CS)", ds_wb1);

    auto ds_cs = g.add_dataset(eta_vec, p_cs_vec);
    g.set_color(Color::ORANGE, ds_cs);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_cs);
    g.set_legend("CS exact", ds_cs);

    g.set_x_limits(0.0, 0.5);
    g.set_y_limits(0.0, 25.0);
    g.set_ticks(0.1, 5.0);
    g.print_to_file("exports/fmt_pressure.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }

  // ── Plot 3: chemical potential ────────────────────────────────────────
  {
    auto g = Grace();
    g.set_title("Excess chemical potential: FMT models");
    g.set_label("\\xh", Axis::X);
    g.set_label("\\xm\\f{}\\sex\\N / kT", Axis::Y);

    auto ds_ros = g.add_dataset(eta_vec, mu_ros);
    g.set_color(Color::BLACK, ds_ros);
    g.set_legend("Rosenfeld", ds_ros);

    auto ds_wb1 = g.add_dataset(eta_vec, mu_wb1);
    g.set_color(Color::BLUE, ds_wb1);
    g.set_legend("White Bear I", ds_wb1);

    auto ds_wb2 = g.add_dataset(eta_vec, mu_wb2);
    g.set_color(Color::DARKGREEN, ds_wb2);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_wb2);
    g.set_legend("White Bear II", ds_wb2);

    g.set_x_limits(0.0, 0.5);
    g.set_ticks(0.1, 5.0);
    g.print_to_file("exports/fmt_chemical_potential.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }

  // ── Plot 4: f-functions ───────────────────────────────────────────────
  {
    auto g = Grace();
    g.set_title("FMT f-functions");
    g.set_label("\\xh", Axis::X);
    g.set_label("f(\\xh\\f{})", Axis::Y);

    std::vector<double> f1_r(N), f2_r(N), f3_r(N), f3_rslt(N), f3_wb2_v(N);
    for (int i = 0; i < N; ++i) {
      double eta = eta_vec[i];
      f1_r[i] = ros.f1(eta);
      f2_r[i] = ros.f2(eta);
      f3_r[i] = ros.f3(eta);
      f3_rslt[i] = rslt.f3(eta);
      f3_wb2_v[i] = wb2.f3(eta);
    }

    auto ds_f1 = g.add_dataset(eta_vec, f1_r);
    g.set_color(Color::BLACK, ds_f1);
    g.set_legend("f\\s1\\N = ln(1-\\xh\\f{})", ds_f1);

    auto ds_f2 = g.add_dataset(eta_vec, f2_r);
    g.set_color(Color::RED, ds_f2);
    g.set_legend("f\\s2\\N = 1/(1-\\xh\\f{})", ds_f2);

    auto ds_f3 = g.add_dataset(eta_vec, f3_r);
    g.set_color(Color::BLUE, ds_f3);
    g.set_legend("f\\s3\\N Rosenfeld", ds_f3);

    auto ds_f3r = g.add_dataset(eta_vec, f3_rslt);
    g.set_color(Color::DARKGREEN, ds_f3r);
    g.set_line_type(LineStyle::DASHEDLINE_EN, ds_f3r);
    g.set_legend("f\\s3\\N RSLT / WBI", ds_f3r);

    auto ds_f3w = g.add_dataset(eta_vec, f3_wb2_v);
    g.set_color(Color::ORANGE, ds_f3w);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_f3w);
    g.set_legend("f\\s3\\N WBII", ds_f3w);

    g.set_x_limits(0.0, 0.5);
    g.set_y_limits(-5.0, 30.0);
    g.set_ticks(0.1, 5.0);
    g.print_to_file("exports/fmt_f_functions.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }

  // ── Plot 5: bounded alias mapping ─────────────────────────────────────
  {
    auto g = Grace();
    g.set_title("Bounded alias: \\xr\\f{}(y) = \\xr\\f{}\\smin\\N + c y\\S2\\N/(1+y\\S2\\N)");
    g.set_label("y (alias variable)", Axis::X);
    g.set_label("\\xr\\f{}(y) / \\xh\\f{}(y)", Axis::Y);

    double c = rho_max - dft_core::physics::species::Species::RHO_MIN;
    int M = 300;
    std::vector<double> y_vals(M), rho_vals(M), eta_vals(M), drho_dy(M);
    for (int i = 0; i < M; ++i) {
      double y = 0.01 + 5.0 * i / (M - 1);
      double y2 = y * y;
      y_vals[i] = y;
      rho_vals[i] = dft_core::physics::species::Species::RHO_MIN + c * y2 / (1.0 + y2);
      eta_vals[i] = packing_fraction(rho_vals[i]);
      double denom = (1.0 + y2) * (1.0 + y2);
      drho_dy[i] = c * 2.0 * y / denom;
    }

    auto ds_rho = g.add_dataset(y_vals, rho_vals);
    g.set_color(Color::BLACK, ds_rho);
    g.set_legend("\\xr\\f{}(y)", ds_rho);

    auto ds_eta = g.add_dataset(y_vals, eta_vals);
    g.set_color(Color::RED, ds_eta);
    g.set_line_type(LineStyle::DASHEDLINE_EN, ds_eta);
    g.set_legend("\\xh\\f{}(y)", ds_eta);

    auto ds_drho = g.add_dataset(y_vals, drho_dy);
    g.set_color(Color::BLUE, ds_drho);
    g.set_line_type(LineStyle::DOTTEDLINE, ds_drho);
    g.set_legend("d\\xr\\f{}/dy", ds_drho);

    g.set_x_limits(0.0, 5.0);
    g.set_y_limits(0.0, 2.5);
    g.set_ticks(1.0, 0.5);
    g.print_to_file("exports/fmt_alias.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }
#endif

  std::cout << "\nDone.\n";
  return 0;
}
