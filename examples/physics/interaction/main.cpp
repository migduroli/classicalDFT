#include <classicaldft>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft_core::physics::interaction;
using namespace dft_core::physics::species;
using namespace dft_core::physics::density;
using namespace dft_core::physics::potentials::intermolecular;

int main() {
  std::filesystem::create_directories("exports");
  std::cout << std::fixed << std::setprecision(6);

  // ── Common parameters ─────────────────────────────────────────────────

  double sigma = 1.0;
  double epsilon = 1.0;
  double r_cutoff = 2.5;
  double dx = 0.25;
  double kT_ref = 1.0;
  double rho0 = 0.5;
  arma::rowvec3 box = {5.0, 5.0, 5.0};

  LennardJones lj(sigma, epsilon, r_cutoff);

  std::cout << "LJ: sigma=" << sigma << ", epsilon=" << epsilon
            << ", r_cutoff=" << r_cutoff << "\n";
  std::cout << "Grid: dx=" << dx << ", box=" << box(0) << "x" << box(1) << "x" << box(2) << "\n";

  // ── 1. Attractive potential profile: WCA vs BH splitting ──────────────

  std::cout << "\n=== Attractive tail: WCA vs BH splitting ===\n\n";

  int n_r = 200;
  std::vector<double> r_vec(n_r), watt_wca(n_r), watt_bh(n_r), v_full(n_r);

  LennardJones lj_wca(sigma, epsilon, r_cutoff);
  LennardJones lj_bh(sigma, epsilon, r_cutoff);
  lj_bh.set_bh_perturbation();

  std::cout << std::setw(8) << "r"
            << std::setw(16) << "v(r)"
            << std::setw(16) << "w_att(WCA)"
            << std::setw(16) << "w_att(BH)"
            << "\n";

  for (int i = 0; i < n_r; ++i) {
    double r = 0.85 + (r_cutoff + 0.1 - 0.85) * i / (n_r - 1);
    r_vec[i] = r;
    v_full[i] = lj_wca.v_potential(r);
    watt_wca[i] = lj_wca.w_attractive(r);
    watt_bh[i] = lj_bh.w_attractive(r);

    if (i % 40 == 0) {
      std::cout << std::setw(8) << r
                << std::setw(16) << v_full[i]
                << std::setw(16) << watt_wca[i]
                << std::setw(16) << watt_bh[i]
                << "\n";
    }
  }

  // ── 2. Van der Waals parameter vs temperature ─────────────────────────

  std::cout << "\n=== Van der Waals parameter: a(kT) ===\n\n";

  int n_temps = 50;
  std::vector<double> kT_vec(n_temps), a_vdw_vec(n_temps);

  std::cout << std::setw(10) << "kT"
            << std::setw(16) << "a_vdw(discrete)"
            << std::setw(16) << "a_vdw(analytic)"
            << "\n";

  for (int i = 0; i < n_temps; ++i) {
    double kT = 0.5 + 4.5 * i / (n_temps - 1);
    kT_vec[i] = kT;

    Species sp(Density(dx, box));
    Interaction inter(sp, sp, lj, kT);
    a_vdw_vec[i] = inter.vdw_parameter();

    if (i % 10 == 0) {
      LennardJones lj_tmp(sigma, epsilon, r_cutoff);
      double a_analytic = 2.0 * lj_tmp.compute_van_der_waals_integral(kT);
      std::cout << std::setw(10) << kT
                << std::setw(16) << a_vdw_vec[i]
                << std::setw(16) << a_analytic
                << "\n";
    }
  }

  // ── 3. Grid convergence: a_vdw vs dx ─────────────────────────────────

  std::cout << "\n=== Grid convergence: a_vdw vs dx (kT=1) ===\n\n";

  LennardJones lj_conv(sigma, epsilon, r_cutoff);
  double a_analytic = 2.0 * lj_conv.compute_van_der_waals_integral(kT_ref);
  std::cout << "Analytic a_vdw = " << a_analytic << "\n\n";

  std::vector<double> dx_vals = {0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1};
  std::vector<double> a_conv(dx_vals.size()), err_conv(dx_vals.size());

  std::cout << std::setw(8) << "dx"
            << std::setw(16) << "a_vdw"
            << std::setw(16) << "rel. error"
            << "\n";

  for (size_t i = 0; i < dx_vals.size(); ++i) {
    double dxi = dx_vals[i];
    double L_raw = std::max(5.0, 2.0 * r_cutoff + 2.0 * dxi);
    double L = std::ceil(L_raw / dxi) * dxi;
    arma::rowvec3 box_i = {L, L, L};
    Species sp(Density(dxi, box_i));
    Interaction inter(sp, sp, lj, kT_ref);
    a_conv[i] = inter.vdw_parameter();
    err_conv[i] = std::abs((a_conv[i] - a_analytic) / a_analytic);

    std::cout << std::setw(8) << dxi
              << std::setw(16) << a_conv[i]
              << std::setw(16) << err_conv[i]
              << "\n";
  }

  // ── 4. Weight scheme comparison ───────────────────────────────────────

  std::cout << "\n=== Weight scheme comparison (kT=1) ===\n\n";

  struct SchemeEntry {
    const char* name;
    WeightScheme scheme;
    int gauss;
  };
  std::vector<SchemeEntry> schemes = {
      {"InterpolationZero", WeightScheme::InterpolationZero, 0},
      {"InterpolationLinearE", WeightScheme::InterpolationLinearE, 0},
      {"InterpolationLinearF", WeightScheme::InterpolationLinearF, 0},
      {"GaussE (order=3)", WeightScheme::GaussE, 3},
      {"GaussE (order=5)", WeightScheme::GaussE, 5},
      {"GaussF (order=5)", WeightScheme::GaussF, 5},
  };
  std::vector<double> scheme_a(schemes.size());

  std::cout << std::setw(25) << "Scheme"
            << std::setw(16) << "a_vdw"
            << std::setw(16) << "rel. error"
            << "\n";

  for (size_t i = 0; i < schemes.size(); ++i) {
    Species sp(Density(dx, box));
    Interaction inter(sp, sp, lj, kT_ref, schemes[i].scheme, schemes[i].gauss);
    scheme_a[i] = inter.vdw_parameter();
    double rel_err = std::abs((scheme_a[i] - a_analytic) / a_analytic);
    std::cout << std::setw(25) << schemes[i].name
              << std::setw(16) << scheme_a[i]
              << std::setw(16) << rel_err
              << "\n";
  }

  // ── 5. Bulk thermodynamics: f(rho) and mu(rho) ───────────────────────

  std::cout << "\n=== Bulk thermodynamics f(rho), mu(rho) at kT=1 ===\n\n";

  Species sp_ref(Density(dx, box));
  Interaction inter_ref(sp_ref, sp_ref, lj, kT_ref);
  double a_ref = inter_ref.vdw_parameter();

  int n_rho = 100;
  std::vector<double> rho_vec(n_rho), f_bulk_vec(n_rho), mu_bulk_vec(n_rho);

  std::cout << std::setw(10) << "rho"
            << std::setw(16) << "f_mf"
            << std::setw(16) << "mu_mf"
            << "\n";

  for (int i = 0; i < n_rho; ++i) {
    double rho = 0.01 + 0.99 * i / (n_rho - 1);
    rho_vec[i] = rho;
    f_bulk_vec[i] = inter_ref.bulk_free_energy_density(rho, rho);
    mu_bulk_vec[i] = inter_ref.bulk_chemical_potential(rho);

    if (i % 20 == 0) {
      std::cout << std::setw(10) << rho
                << std::setw(16) << f_bulk_vec[i]
                << std::setw(16) << mu_bulk_vec[i]
                << "\n";
    }
  }

  // ── 6. Energy and forces on a uniform 3D grid ─────────────────────────

  std::cout << "\n=== Uniform density: energy and force consistency ===\n";

  Species sp_u(Density(dx, box));
  sp_u.density().values().fill(rho0);
  Interaction inter_u(sp_u, sp_u, lj, kT_ref);

  double F_mf = inter_u.compute_free_energy();
  double V = static_cast<double>(sp_u.density().size()) * sp_u.density().cell_volume();
  double F_bulk = inter_u.bulk_free_energy_density(rho0, rho0) * V;
  std::cout << "Uniform rho = " << rho0 << ", a_vdw = " << a_ref << "\n";
  std::cout << "F_mf (computed): " << F_mf << "\n";
  std::cout << "F_mf (bulk):     " << F_bulk << "\n";
  std::cout << "Relative error:  " << std::abs((F_mf - F_bulk) / F_bulk) << "\n";

  sp_u.zero_force();
  inter_u.compute_forces();
  double mean_force = arma::mean(sp_u.force());
  double dV = sp_u.density().cell_volume();
  double mu_expected = inter_u.bulk_chemical_potential(rho0);
  std::cout << "Mean force / dV:    " << mean_force / dV << "\n";
  std::cout << "mu_mf (bulk):       " << mu_expected << "\n";
  std::cout << "Force uniformity:   max|f - mean| = "
            << arma::max(arma::abs(sp_u.force() - mean_force)) << "\n";

  // ── 7. Cross-species interaction ──────────────────────────────────────

  std::cout << "\n=== Cross-species interaction (LJ 1-1 vs LJ 1-2) ===\n";

  Species s1(Density(dx, box));
  Species s2(Density(dx, box));
  s1.density().values().fill(rho0);
  s2.density().values().fill(0.3);

  LennardJones lj_12(1.2, 0.8, 2.5);

  Interaction self_11(s1, s1, lj, kT_ref);
  Interaction cross_12(s1, s2, lj_12, kT_ref);

  double F_11 = self_11.compute_free_energy();
  double F_12 = cross_12.compute_free_energy();

  std::cout << "s1: rho=" << rho0 << ", s2: rho=0.3\n";
  std::cout << "LJ(1-1): sigma=1.0, eps=1.0 -> a_vdw = " << self_11.vdw_parameter() << "\n";
  std::cout << "LJ(1-2): sigma=1.2, eps=0.8 -> a_vdw = " << cross_12.vdw_parameter() << "\n";
  std::cout << "F_self(1-1)  = " << F_11 << "\n";
  std::cout << "F_cross(1-2) = " << F_12 << "\n";

  s1.zero_force();
  s2.zero_force();
  cross_12.compute_forces();
  double f1_mean = arma::mean(s1.force()) / dV;
  double f2_mean = arma::mean(s2.force()) / dV;
  std::cout << "Cross force on s1 / dV = " << f1_mean
            << " (expected: " << 0.5 * cross_12.vdw_parameter() * 0.3 << ")\n";
  std::cout << "Cross force on s2 / dV = " << f2_mean
            << " (expected: " << 0.5 * cross_12.vdw_parameter() * rho0 << ")\n";

  // ── 8. BH perturbation effect on a_vdw ────────────────────────────────

  std::cout << "\n=== BH perturbation: WCA vs BH splitting ===\n";

  LennardJones lj_bh_inter(sigma, epsilon, r_cutoff);
  lj_bh_inter.set_bh_perturbation();

  Species sp_wca(Density(dx, box));
  Species sp_bh(Density(dx, box));
  Interaction inter_wca(sp_wca, sp_wca, lj, kT_ref);
  Interaction inter_bh(sp_bh, sp_bh, lj_bh_inter, kT_ref);

  std::cout << "a_vdw (WCA split): " << inter_wca.vdw_parameter() << "\n";
  std::cout << "a_vdw (BH split):  " << inter_bh.vdw_parameter() << "\n";
  std::cout << "Difference:        " << inter_wca.vdw_parameter() - inter_bh.vdw_parameter() << "\n";

  // ── 9. Sinusoidal density: force profile along x ──────────────────────

  std::cout << "\n=== Sinusoidal density: force profile ===\n";

  double dx_fine = 0.2;
  arma::rowvec3 box_fine = {6.0, 6.0, 6.0};
  Species sp_sin(Density(dx_fine, box_fine));
  auto& shape = sp_sin.density().shape();
  long nx = shape[0];
  long ny = shape[1];
  long nz = shape[2];

  double rho_avg = 0.5;
  double rho_amp = 0.2;
  double Lx = box_fine(0);

  for (long ix = 0; ix < nx; ++ix) {
    double x = ix * dx_fine;
    double rho_x = rho_avg + rho_amp * std::sin(2.0 * std::numbers::pi * x / Lx);
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        auto idx = sp_sin.density().flat_index(ix, iy, iz);
        sp_sin.density().values()(idx) = rho_x;
      }
    }
  }

  Interaction inter_sin(sp_sin, sp_sin, lj, kT_ref);
  double F_sin = inter_sin.compute_free_energy();
  sp_sin.zero_force();
  inter_sin.compute_forces();

  int n_profile = nx;
  std::vector<double> x_profile(n_profile), rho_profile(n_profile), force_profile(n_profile);

  for (long ix = 0; ix < nx; ++ix) {
    x_profile[ix] = ix * dx_fine;
    auto idx = sp_sin.density().flat_index(ix, 0, 0);
    rho_profile[ix] = sp_sin.density().values()(idx);
    force_profile[ix] = sp_sin.force()(idx) / sp_sin.density().cell_volume();
  }

  std::cout << "rho(x) = " << rho_avg << " + " << rho_amp << " sin(2 pi x / " << Lx << ")\n";
  std::cout << "F_mf = " << F_sin << "\n";
  std::cout << std::setw(8) << "x"
            << std::setw(14) << "rho(x)"
            << std::setw(14) << "force/dV"
            << "\n";

  for (int i = 0; i < n_profile; i += std::max(1, n_profile / 8)) {
    std::cout << std::setw(8) << x_profile[i]
              << std::setw(14) << rho_profile[i]
              << std::setw(14) << force_profile[i]
              << "\n";
  }

  // ── 10. Energy-force consistency: numerical derivative ────────────────

  std::cout << "\n=== Energy-force consistency (numerical derivative) ===\n";

  Species sp_num(Density(dx, box));
  sp_num.density().values().fill(rho0);
  long test_site = sp_num.density().size() / 2;
  double delta = 1e-5;

  // F(rho + delta) at test_site
  sp_num.density().values()(test_site) = rho0 + delta;
  Interaction inter_plus(sp_num, sp_num, lj, kT_ref);
  double F_plus = inter_plus.compute_free_energy();

  // F(rho - delta) at test_site
  sp_num.density().values()(test_site) = rho0 - delta;
  Interaction inter_minus(sp_num, sp_num, lj, kT_ref);
  double F_minus = inter_minus.compute_free_energy();

  double dFdrho_numerical = (F_plus - F_minus) / (2.0 * delta);

  // Analytic force
  sp_num.density().values()(test_site) = rho0;
  Interaction inter_force(sp_num, sp_num, lj, kT_ref);
  sp_num.zero_force();
  inter_force.compute_forces();
  double dFdrho_analytic = sp_num.force()(test_site);

  std::cout << "dF/drho (numerical):  " << dFdrho_numerical << "\n";
  std::cout << "dF/drho (analytic):   " << dFdrho_analytic << "\n";
  std::cout << "Relative error:       " << std::abs((dFdrho_numerical - dFdrho_analytic) / dFdrho_analytic) << "\n";

  // ── matplotlib plots ───────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;
  plt::backend("Agg");

  std::cout << "\nGenerating matplotlib plots...\n";

  // ── Plot 1: Attractive tail decomposition ─────────────────────────────
  plt::figure_size(800, 500);
  plt::named_plot("v(r) (full, cut+shifted)", r_vec, v_full, "k-");
  plt::named_plot("w_att (WCA: flat inside r_min)", r_vec, watt_wca, "b-");
  plt::named_plot("w_att (BH: zero inside r0)", r_vec, watt_bh, "r--");
  plt::xlabel(R"(r / $\sigma$)");
  plt::ylabel(R"(v(r), $w_{\mathrm{att}}(r)$ / $\epsilon$)");
  plt::title(R"(LJ potential: WCA vs BH attractive tail ($\sigma$=1, $\epsilon$=1))");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/interaction_potential_decomposition.png");
  plt::close();
  std::cout << "Plot saved: " << std::filesystem::absolute("exports/interaction_potential_decomposition.png") << std::endl;

  // ── Plot 2: a_vdw vs temperature ──────────────────────────────────────
  {
    plt::figure_size(800, 500);
    plt::named_plot("a(kT) discrete (dx=0.25)", kT_vec, a_vdw_vec, "b-o");

    std::vector<double> a_analytic_vec(n_temps);
    for (int i = 0; i < n_temps; ++i) {
      LennardJones lj_tmp(sigma, epsilon, r_cutoff);
      a_analytic_vec[i] = 2.0 * lj_tmp.compute_van_der_waals_integral(kT_vec[i]);
    }
    plt::named_plot("a(kT) analytic", kT_vec, a_analytic_vec, "r--");

    plt::xlabel(R"(kT / $\epsilon$)");
    plt::ylabel(R"($a_{\mathrm{vdw}}$ / ($\epsilon \sigma^3$))");
    plt::title("Van der Waals parameter vs temperature (LJ, r_c=2.5)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/interaction_avdw_vs_temperature.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/interaction_avdw_vs_temperature.png") << std::endl;
  }

  // ── Plot 3: Grid convergence of a_vdw ─────────────────────────────────
  plt::figure_size(800, 500);
  plt::semilogy(dx_vals, err_conv, "bo");
  plt::xlabel(R"(dx / $\sigma$)");
  plt::ylabel(R"($|a_{\mathrm{discrete}} - a_{\mathrm{analytic}}|$ / $|a_{\mathrm{analytic}}|$)");
  plt::title(R"(Grid convergence of $a_{\mathrm{vdw}}$ (LJ, kT=1))");
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/interaction_grid_convergence.png");
  plt::close();
  std::cout << "Plot saved: " << std::filesystem::absolute("exports/interaction_grid_convergence.png") << std::endl;

  // ── Plot 4: Bulk thermodynamics ───────────────────────────────────────
  plt::figure_size(800, 500);
  plt::named_plot(R"($f_{\mathrm{mf}} = a \rho^2 / 2$)", rho_vec, f_bulk_vec, "b-");
  plt::named_plot(R"($\mu_{\mathrm{mf}} = a \rho$)", rho_vec, mu_bulk_vec, "r-");
  plt::xlabel(R"($\rho \sigma^3$)");
  plt::ylabel("f / kT , mu / kT");
  plt::title("Mean-field bulk thermodynamics (LJ, kT=1)");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/interaction_bulk_thermodynamics.png");
  plt::close();
  std::cout << "Plot saved: " << std::filesystem::absolute("exports/interaction_bulk_thermodynamics.png") << std::endl;

  // ── Plot 5: Scheme comparison (bar chart) ─────────────────────────────
  {
    std::vector<double> x_idx(schemes.size());
    std::vector<std::string> scheme_labels(schemes.size());
    for (size_t i = 0; i < schemes.size(); ++i) {
      x_idx[i] = static_cast<double>(i);
      scheme_labels[i] = schemes[i].name;
    }

    plt::figure_size(800, 500);
    plt::bar(x_idx, scheme_a);
    plt::axhline(a_analytic, 0.0, 1.0,
                 {{"color", "red"}, {"linestyle", "--"}, {"label", "Analytic a_vdw"}});
    plt::xticks(x_idx, scheme_labels);
    plt::ylabel(R"($a_{\mathrm{vdw}}$)");
    plt::title(R"(Weight scheme comparison: $a_{\mathrm{vdw}}$ at kT=1 (dx=0.25))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/interaction_scheme_comparison.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/interaction_scheme_comparison.png") << std::endl;
  }

  // ── Plot 6: Sinusoidal density and force profile ──────────────────────
  plt::figure_size(800, 500);
  plt::named_plot(R"($\rho(x)$)", x_profile, rho_profile, "b-");
  plt::named_plot(R"($dF/d\rho(x)$ / dV)", x_profile, force_profile, "r-");
  plt::xlabel(R"(x / $\sigma$)");
  plt::ylabel(R"($\rho \sigma^3$ , force density)");
  plt::title(R"(Sinusoidal density: $\rho(x)$ and force profile)");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/interaction_sinusoidal_profile.png");
  plt::close();
  std::cout << "Plot saved: " << std::filesystem::absolute("exports/interaction_sinusoidal_profile.png") << std::endl;

  auto export_dir = std::filesystem::absolute("exports");
  std::cout << "Plots saved to " << export_dir.string() << ":\n";
  for (auto& entry : std::filesystem::directory_iterator(export_dir)) {
    std::cout << "  " << entry.path().filename().string()
              << " (" << std::filesystem::file_size(entry) << " bytes)\n";
  }
#endif

  std::cout << "\nDone.\n";
  return 0;
}
