#include "dft.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft::functional::fmt;
using namespace dft::density;
using namespace dft::thermodynamics;

int main(int argc, char* argv[]) {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  std::string config_path = (argc > 1) ? argv[1] : "config.ini";
  auto cfg = dft::config::ConfigParser(config_path);

  // ── 1. Bulk free energy: FMT models vs known EOS ─────────────────────

  std::cout << "=== Bulk excess free energy per particle: FMT vs EOS ===\n\n";
  std::cout << std::fixed << std::setprecision(8);
  std::cout << std::setw(8) << "eta"
            << std::setw(16) << "Rosenfeld"
            << std::setw(16) << "PY(comp)"
            << std::setw(16) << "WhiteBearI"
            << std::setw(16) << "CS"
            << "\n";

  FMT ros_m(Rosenfeld{}), rslt_m(RSLT{}), wb1_m(WhiteBearI{}), wb2_m(WhiteBearII{});
  HardSphereModel cs_model = CarnahanStarling{};
  HardSphereModel pyc_model = PercusYevickCompressibility{};
  HardSphereModel pyv_model = PercusYevickVirial{};

  const int N = static_cast<int>(cfg.get<double>("density.n_eta"));
  std::vector<double> eta_vec(N), f_ros(N), f_rslt(N), f_wb1(N), f_wb2(N);
  std::vector<double> f_cs(N), f_pyc(N), f_pyv(N);

  for (int i = 0; i < N; ++i) {
    double eta = 0.005 + 0.49 * i / (N - 1);
    double rho = density_from_eta(eta);
    eta_vec[i] = eta;

    f_ros[i] = ros_m.bulk_free_energy_density(rho, 1.0) / rho;
    f_rslt[i] = rslt_m.bulk_free_energy_density(rho, 1.0) / rho;
    f_wb1[i] = wb1_m.bulk_free_energy_density(rho, 1.0) / rho;
    f_wb2[i] = wb2_m.bulk_free_energy_density(rho, 1.0) / rho;
    f_cs[i] = CarnahanStarling::excess_free_energy(eta);
    f_pyc[i] = PercusYevickCompressibility::excess_free_energy(eta);
    f_pyv[i] = PercusYevickVirial::excess_free_energy(eta);

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
    mu_ros[i] = ros_m.bulk_excess_chemical_potential(rho, 1.0);
    mu_wb1[i] = wb1_m.bulk_excess_chemical_potential(rho, 1.0);
    mu_wb2[i] = wb2_m.bulk_excess_chemical_potential(rho, 1.0);

    if (i % 40 == 0) {
      std::cout << std::setw(8) << eta_vec[i]
                << std::setw(16) << mu_ros[i]
                << std::setw(16) << mu_wb1[i]
                << std::setw(16) << mu_wb2[i]
                << "\n";
    }
  }

  // ── 3. FMT species: free energy on a grid ─────────────────────────────

  std::cout << "\n=== FMT species on 3D grid ===\n\n";

  double dx = cfg.get<double>("grid.dx");
  double box_length = cfg.get<double>("grid.box_length");
  arma::rowvec3 box = {box_length, box_length, box_length};
  double rho0 = cfg.get<double>("density.rho0");
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
  double F_ros = sp.compute_free_energy(ros_m);
  double F_wb1 = sp.compute_free_energy(wb1_m);

  double V = box(0) * box(1) * box(2);
  std::cout << "F_ex (Rosenfeld):   " << F_ros << "  (bulk: " << ros_m.bulk_free_energy_density(rho0, 1.0) * V << ")\n";
  std::cout << "F_ex (WhiteBearI):  " << F_wb1 << "  (bulk: " << wb1_m.bulk_free_energy_density(rho0, 1.0) * V << ")\n";

  // Forces (separate call)
  sp.zero_force();
  double F_from_forces = sp.compute_forces(ros_m);
  double mean_force = arma::mean(sp.force());
  double mu_ex = ros_m.bulk_excess_chemical_potential(rho0, 1.0);
  double dV = sp.density().cell_volume();
  std::cout << "Mean force / dV:    " << mean_force / dV << "  (mu_ex: " << mu_ex << ")\n";
  std::cout << "Force uniformity: max|f - mean| = " << arma::max(arma::abs(sp.force() - mean_force)) << "\n";

  // ── 4. Bounded alias demonstration ────────────────────────────────────

  std::cout << "\n=== Bounded alias ===\n\n";
  double d = 1.0;
  double rho_max = 0.9999 * 6.0 / (std::numbers::pi * d * d * d);
  std::cout << "rho_min = " << dft::species::Species::RHO_MIN << "\n";
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
    p_cs_vec[i] = hs_pressure(cs_model, eta);
    p_pyc[i] = hs_pressure(pyc_model, eta);

    if (i % 40 == 0) {
      std::cout << std::setw(8) << eta
                << std::setw(16) << p_pyc[i]
                << std::setw(16) << p_ros[i]
                << std::setw(16) << p_cs_vec[i]
                << std::setw(16) << p_wb1[i]
                << "\n";
    }
  }
  std::cout << "\n" << std::endl;

  // ── Plots ──────────────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;
  plt::backend("Agg");

  // ── Plot 1: excess free energy comparison ─────────────────────────────
  {
    plt::figure_size(800, 550);
    plt::named_plot("Rosenfeld (= PY comp.)", eta_vec, f_ros, "k-");
    plt::named_plot("PY (virial)", eta_vec, f_pyv, "r--");
    plt::named_plot("White Bear I (= CS)", eta_vec, f_wb1, "b-");
    plt::named_plot("White Bear II", eta_vec, f_wb2, "g:");
    plt::xlim(0.0, 0.5);
    plt::ylim(0.0, 8.0);
    plt::xlabel(R"($\eta$)");
    plt::ylabel(R"($f_\mathrm{ex} / k_BT$)");
    plt::title("Excess free energy per particle: FMT vs EOS");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/fmt_free_energy.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/fmt_free_energy.png") << std::endl;
  }

  // ── Plot 2: pressure comparison ───────────────────────────────────────
  {
    plt::figure_size(800, 550);
    plt::named_plot("Rosenfeld (= PY comp.)", eta_vec, p_ros, "k-");
    plt::named_plot("PY (comp.) exact", eta_vec, p_pyc, "r--");
    plt::named_plot("White Bear I (= CS)", eta_vec, p_wb1, "b-");
    plt::named_plot("CS exact", eta_vec, p_cs_vec, "m:");
    plt::xlim(0.0, 0.5);
    plt::ylim(0.0, 25.0);
    plt::xlabel(R"($\eta$)");
    plt::ylabel(R"($P / (\rho\, k_BT)$)");
    plt::title("Compressibility factor: FMT vs exact EOS");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/fmt_pressure.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/fmt_pressure.png") << std::endl;
  }

  // ── Plot 3: chemical potential ────────────────────────────────────────
  {
    plt::figure_size(800, 550);
    plt::named_plot("Rosenfeld", eta_vec, mu_ros, "k-");
    plt::named_plot("White Bear I", eta_vec, mu_wb1, "b-");
    plt::named_plot("White Bear II", eta_vec, mu_wb2, "g:");
    plt::xlim(0.0, 0.5);
    plt::xlabel(R"($\eta$)");
    plt::ylabel(R"($\mu_\mathrm{ex} / k_BT$)");
    plt::title("Excess chemical potential: FMT models");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/fmt_chemical_potential.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/fmt_chemical_potential.png") << std::endl;
  }

  // ── Plot 4: f-functions ───────────────────────────────────────────────
  {
    std::vector<double> f1_r(N), f2_r(N), f3_r(N), f3_rslt(N), f3_wb2_v(N);
    for (int i = 0; i < N; ++i) {
      double eta = eta_vec[i];
      f1_r[i] = ros_m.ideal_factor(eta);
      f2_r[i] = ros_m.pair_factor(eta);
      f3_r[i] = ros_m.triplet_factor(eta);
      f3_rslt[i] = rslt_m.triplet_factor(eta);
      f3_wb2_v[i] = wb2_m.triplet_factor(eta);
    }

    plt::figure_size(800, 550);
    plt::named_plot(R"($f_1 = \ln(1-\eta)$)", eta_vec, f1_r, "k-");
    plt::named_plot(R"($f_2 = 1/(1-\eta)$)", eta_vec, f2_r, "r-");
    plt::named_plot(R"($f_3$ Rosenfeld)", eta_vec, f3_r, "b-");
    plt::named_plot(R"($f_3$ RSLT / WBI)", eta_vec, f3_rslt, "g--");
    plt::named_plot(R"($f_3$ WBII)", eta_vec, f3_wb2_v, "m:");
    plt::xlim(0.0, 0.5);
    plt::ylim(-5.0, 30.0);
    plt::xlabel(R"($\eta$)");
    plt::ylabel(R"($f(\eta)$)");
    plt::title("FMT f-functions");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/fmt_f_functions.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/fmt_f_functions.png") << std::endl;
  }

  // ── Plot 5: bounded alias mapping ─────────────────────────────────────
  {
    double c = rho_max - dft::species::Species::RHO_MIN;
    int M = 300;
    std::vector<double> y_vals(M), rho_vals(M), eta_vals(M), drho_dy(M);
    for (int i = 0; i < M; ++i) {
      double y = 0.01 + 5.0 * i / (M - 1);
      double y2 = y * y;
      y_vals[i] = y;
      rho_vals[i] = dft::species::Species::RHO_MIN + c * y2 / (1.0 + y2);
      eta_vals[i] = packing_fraction(rho_vals[i]);
      double denom = (1.0 + y2) * (1.0 + y2);
      drho_dy[i] = c * 2.0 * y / denom;
    }

    plt::figure_size(800, 550);
    plt::named_plot(R"($\rho(y)$)", y_vals, rho_vals, "k-");
    plt::named_plot(R"($\eta(y)$)", y_vals, eta_vals, "r--");
    plt::named_plot(R"($d\rho/dy$)", y_vals, drho_dy, "b:");
    plt::xlim(0.0, 5.0);
    plt::ylim(0.0, 2.5);
    plt::xlabel(R"($y$ (alias variable))");
    plt::ylabel(R"($\rho(y)$ / $\eta(y)$)");
    plt::title(R"(Bounded alias: $\rho(y) = \rho_\mathrm{min} + c\,y^2/(1+y^2)$)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/fmt_alias.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/fmt_alias.png") << std::endl;
  }
#endif

  std::cout << "\nDone.\n";
  return 0;
}
