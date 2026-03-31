#include "dft.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

using namespace dft;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Build a single-component LJ solver at temperature kT.
static Solver make_lj_solver(double dx, const arma::rowvec3& box, double diameter,
                              const potentials::LennardJones& lj, double kT,
                              functional::fmt::FMTModel model) {
  Solver s;
  auto dens = density::Density(dx, box);
  dens.values().fill(0.1);
  auto sp = std::make_unique<functional::fmt::Species>(std::move(dens), diameter);
  auto& sp_ref = *sp;
  s.add_species(std::move(sp));
  s.add_interaction(
      std::make_unique<functional::interaction::Interaction>(sp_ref, sp_ref, lj, kT));
  s.set_fmt(std::make_unique<functional::fmt::FMT>(std::move(model)));
  return s;
}

/// Extract the spherically-averaged radial density profile.
static void extract_radial_profile(const Solver& solver, std::vector<double>& r_out,
                                    std::vector<double>& rho_out) {
  const auto& dens = solver.density(0);
  const auto& shape = dens.shape();
  double dx = dens.dx();
  arma::rowvec3 center = dens.box_size() / 2.0;

  double r_max = center.min();
  int n_bins = static_cast<int>(r_max / dx);
  std::vector<double> sum(static_cast<size_t>(n_bins), 0.0);
  std::vector<int> count(static_cast<size_t>(n_bins), 0);

  for (long ix = 0; ix < shape[0]; ++ix) {
    double x = ix * dx - center(0);
    for (long iy = 0; iy < shape[1]; ++iy) {
      double y = iy * dx - center(1);
      for (long iz = 0; iz < shape[2]; ++iz) {
        double z = iz * dx - center(2);
        double r = std::sqrt(x * x + y * y + z * z);
        int bin = static_cast<int>(r / dx);
        if (bin < n_bins) {
          sum[static_cast<size_t>(bin)] += dens.values()(dens.flat_index(ix, iy, iz));
          count[static_cast<size_t>(bin)] += 1;
        }
      }
    }
  }

  r_out.clear();
  rho_out.clear();
  for (int i = 0; i < n_bins; ++i) {
    if (count[static_cast<size_t>(i)] > 0) {
      r_out.push_back((i + 0.5) * dx);
      rho_out.push_back(sum[static_cast<size_t>(i)] / count[static_cast<size_t>(i)]);
    }
  }
}

/// Set the solver density to a spherical tanh-profile droplet.
static void init_droplet(Solver& solver, double rho_v, double rho_l, double r_droplet,
                          double interface_width) {
  auto& dens = solver.species(0).density();
  const auto& shape = dens.shape();
  double dx = dens.dx();
  arma::rowvec3 center = dens.box_size() / 2.0;
  arma::vec rho(dens.size());

  double mid = 0.5 * (rho_l + rho_v);
  double amp = 0.5 * (rho_l - rho_v);

  for (long ix = 0; ix < shape[0]; ++ix) {
    double x = ix * dx - center(0);
    for (long iy = 0; iy < shape[1]; ++iy) {
      double y = iy * dx - center(1);
      for (long iz = 0; iz < shape[2]; ++iz) {
        double z = iz * dx - center(2);
        double r = std::sqrt(x * x + y * y + z * z);
        rho(dens.flat_index(ix, iy, iz)) = mid - amp * std::tanh((r - r_droplet) / interface_width);
      }
    }
  }
  dens.set(rho);
}

/// Radial density snapshot at a given time.
struct Snapshot {
  double time;
  std::vector<double> r;
  std::vector<double> rho;
};

/// Capture the current radial profile as a Snapshot.
static Snapshot take_snapshot(const Solver& solver, double time) {
  Snapshot snap;
  snap.time = time;
  extract_radial_profile(solver, snap.r, snap.rho);
  return snap;
}

/// Spline-interpolate a snapshot's radial profile onto a dense grid.
/// Returns (r_dense, rho_dense) with n_pts uniformly spaced points.
static std::pair<std::vector<double>, std::vector<double>>
spline_profile(const std::vector<double>& r, const std::vector<double>& rho, int n_pts = 200) {
  std::pair<std::vector<double>, std::vector<double>> result;
  if (r.size() < 4) {
    result = {r, rho};
    return result;
  }
  math::spline::CubicSpline spl(r, rho);
  auto& [r_out, rho_out] = result;
  double lo = r.front();
  double hi = r.back();
  for (int i = 0; i <= n_pts; ++i) {
    double x = lo + (hi - lo) * i / n_pts;
    r_out.push_back(x);
    rho_out.push_back(spl(x));
  }
  return result;
}

/// Generate a hex colour string by linearly interpolating between two RGB values.
static std::string lerp_color(int r0, int g0, int b0, int r1, int g1, int b1, double frac) {
  auto lerp = [](int a, int b, double t) { return static_cast<int>(a + t * (b - a)); };
  char buf[8];
  std::snprintf(buf, sizeof(buf), "#%02X%02X%02X", lerp(r0, r1, frac), lerp(g0, g1, frac),
                lerp(b0, b1, frac));
  return buf;
}

// ── Main ─────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  std::string config_path = (argc > 1) ? argv[1] : "config.ini";
  auto cfg = config::ConfigParser(config_path);

  // ── Configuration ──────────────────────────────────────────────────────

  double sigma = cfg.get<double>("potential.sigma");
  double epsilon = cfg.get<double>("potential.epsilon");
  double r_cutoff = cfg.get<double>("potential.r_cutoff");
  double dx = cfg.get<double>("grid.dx");
  double box_length = cfg.get<double>("grid.box_length");
  double kT = cfg.get<double>("thermodynamics.kT");
  double delta_mu = cfg.get<double>("thermodynamics.delta_mu");
  double interface_w = cfg.get<double>("droplet.interface_width");
  double dR = cfg.get<double>("droplet.delta_R");
  double R_min = cfg.get<double>("barrier_scan.R_min");
  double R_max = cfg.get<double>("barrier_scan.R_max");
  double R_step = cfg.get<double>("barrier_scan.R_step");
  int steps_per_snap = static_cast<int>(cfg.get<double>("dynamics.steps_per_snapshot"));
  int n_snaps = static_cast<int>(cfg.get<double>("dynamics.n_snapshots"));

  dynamics::IntegratorConfig iconf{
      .scheme = dynamics::IntegrationScheme::SplitOperator,
      .dt = cfg.get<double>("dynamics.dt"),
      .diffusion_coefficient = cfg.get<double>("dynamics.diffusion_coefficient"),
      .force_limit = cfg.get<double>("dynamics.force_limit"),
      .min_density = cfg.get<double>("dynamics.min_density"),
  };

  arma::rowvec3 box = {box_length, box_length, box_length};

  potentials::LennardJones lj(sigma, epsilon, r_cutoff);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "=== Droplet nucleation: critical cluster & dynamics ===\n"
            << "  LJ: sigma=" << sigma << " epsilon=" << epsilon << " r_c=" << r_cutoff << "\n"
            << "  Grid: " << box(0) << "^3, dx=" << dx << ", T*=" << kT << "\n\n";

  auto solver = make_lj_solver(dx, box, sigma, lj, kT, functional::fmt::WhiteBearII{});

  // ── 1. Coexistence ────────────────────────────────────────────────────

  double rho_v = 0.0;
  double rho_l = 0.0;
  solver.find_coexistence(1.1, 0.005, rho_v, rho_l, 1e-8);
  double mu_coex = solver.chemical_potential(rho_v);

  // Set supersaturated chemical potential (drives nucleation).
  double mu = mu_coex + delta_mu;
  solver.species(0).set_chemical_potential(mu);

  std::cout << "rho_vapor  = " << rho_v << "\n"
            << "rho_liquid = " << rho_l << "\n"
            << "mu_coex    = " << mu_coex << "\n"
            << "delta_mu   = " << delta_mu << "\n"
            << "mu         = " << mu << "\n\n";

  // ── 2. Grand-potential barrier Omega(R) ───────────────────────────────
  // For each trial radius R, initialise a tanh droplet and evaluate the
  // total grand potential Omega[rho_R].  The barrier maximum locates the
  // critical nucleus.

  // Reference: Omega for uniform vapor at this chemical potential.
  {
    arma::vec rho_uni(solver.density(0).values().n_elem);
    rho_uni.fill(rho_v);
    solver.species(0).density().set(rho_uni);
  }
  double omega_ref = solver.compute_free_energy_and_forces();

  std::cout << "=== Scanning Omega(R) ===\n"
            << std::setw(10) << "R/sigma" << std::setw(18) << "Delta_Omega\n"
            << std::string(28, '-') << "\n";

  std::vector<double> R_scan;
  std::vector<double> dOmega_scan;

  for (double R = R_min; R <= R_max; R += R_step) {
    init_droplet(solver, rho_v, rho_l, R, interface_w);
    double omega = solver.compute_free_energy_and_forces();
    double dw = omega - omega_ref;
    R_scan.push_back(R);
    dOmega_scan.push_back(dw);
    std::cout << std::fixed << std::setprecision(2) << std::setw(10) << R << std::scientific
              << std::setprecision(6) << std::setw(18) << dw << "\n";
  }

  // Locate R* = argmax Delta_Omega.
  size_t i_max = 0;
  for (size_t i = 1; i < dOmega_scan.size(); ++i) {
    if (dOmega_scan[i] > dOmega_scan[i_max]) {
      i_max = i;
    }
  }
  double R_star = R_scan[i_max];
  double barrier = dOmega_scan[i_max];

  std::cout << "\nCritical nucleus: R* ~ " << std::fixed << std::setprecision(2) << R_star
            << " sigma, Delta_Omega* = " << std::scientific << barrier << "\n";

  // ── 3. Critical cluster profile ───────────────────────────────────────

  init_droplet(solver, rho_v, rho_l, R_star, interface_w);
  Snapshot critical = take_snapshot(solver, 0.0);

  // ── 4. DDFT dynamics: sub-critical (dissolution) ──────────────────────

  double R_sub = R_star - dR;

  std::cout << "\n=== DDFT dissolution: R0 = " << std::fixed << std::setprecision(2) << R_sub
            << " sigma (< R*) ===\n";

  init_droplet(solver, rho_v, rho_l, R_sub, interface_w);
  std::vector<Snapshot> sub_snaps;
  sub_snaps.push_back(take_snapshot(solver, 0.0));

  {
    dynamics::Integrator integrator(solver, iconf);
    for (int s = 0; s < n_snaps; ++s) {
      (void)integrator.resume(steps_per_snap);
      double t = (s + 1) * steps_per_snap * iconf.dt;
      sub_snaps.push_back(take_snapshot(solver, t));
      std::cout << "  t = " << std::fixed << std::setprecision(4) << t
                << "  Omega = " << std::scientific << integrator.energy() << "\n";
    }
  }

  // ── 5. DDFT dynamics: super-critical (growth, small overshoot) ────────

  double R_sup = R_star + dR;

  std::cout << "\n=== DDFT growth (small): R0 = " << std::fixed << std::setprecision(2) << R_sup
            << " sigma (> R*) ===\n";

  init_droplet(solver, rho_v, rho_l, R_sup, interface_w);
  std::vector<Snapshot> sup_snaps;
  sup_snaps.push_back(take_snapshot(solver, 0.0));

  {
    dynamics::Integrator integrator(solver, iconf);
    for (int s = 0; s < n_snaps; ++s) {
      (void)integrator.resume(steps_per_snap);
      double t = (s + 1) * steps_per_snap * iconf.dt;
      sup_snaps.push_back(take_snapshot(solver, t));
      std::cout << "  t = " << std::fixed << std::setprecision(4) << t
                << "  Omega = " << std::scientific << integrator.energy() << "\n";
    }
  }

  // ── 6. DDFT dynamics: super-critical (growth, large overshoot) ────────

  double R_sup2 = R_star + 2.0 * dR;

  std::cout << "\n=== DDFT growth (large): R0 = " << std::fixed << std::setprecision(2) << R_sup2
            << " sigma (>> R*) ===\n";

  init_droplet(solver, rho_v, rho_l, R_sup2, interface_w);
  std::vector<Snapshot> sup2_snaps;
  sup2_snaps.push_back(take_snapshot(solver, 0.0));

  {
    dynamics::Integrator integrator(solver, iconf);
    for (int s = 0; s < n_snaps; ++s) {
      (void)integrator.resume(steps_per_snap);
      double t = (s + 1) * steps_per_snap * iconf.dt;
      sup2_snaps.push_back(take_snapshot(solver, t));
      std::cout << "  t = " << std::fixed << std::setprecision(4) << t
                << "  Omega = " << std::scientific << integrator.energy() << "\n";
    }
  }

  // ── 7. Plots ──────────────────────────────────────────────────────────

  // Lambda: plot a set of snapshots with spline-smooth curves + raw markers.
  auto plot_snapshots = [&](const std::vector<Snapshot>& snaps,
                            int r0, int g0, int b0, int r1, int g1, int b1) {
    auto n = static_cast<int>(snaps.size());
    for (int i = 0; i < n; ++i) {
      double frac = static_cast<double>(i) / std::max(n - 1, 1);
      std::string color = lerp_color(r0, g0, b0, r1, g1, b1, frac);
      const auto& s = snaps[static_cast<size_t>(i)];
      char label[32];
      std::snprintf(label, sizeof(label), "t = %.3f", s.time);

      // Spline-interpolated smooth curve.
      auto [rs, rhos] = spline_profile(s.r, s.rho);
      plt::plot(rs, rhos,
                {{"color", color}, {"linewidth", "1.5"}, {"label", label}});

      // Raw data markers.
      plt::plot(s.r, s.rho,
                {{"color", color}, {"marker", "o"}, {"markersize", "2.5"},
                 {"linestyle", "none"}});
    }
  };

  auto plot_critical = [&]() {
    auto [rc, rhoc] = spline_profile(critical.r, critical.rho);
    plt::named_plot(R"($\rho^*(r)$ critical)", rc, rhoc, "k--");
    plt::plot(critical.r, critical.rho,
              {{"color", "black"}, {"marker", "s"}, {"markersize", "3"},
               {"linestyle", "none"}});
  };

  auto plot_reference_lines = [&]() {
    plt::plot({0.0, critical.r.back()}, {rho_v, rho_v},
              {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_v$)"}});
    plt::plot({0.0, critical.r.back()}, {rho_l, rho_l},
              {{"color", "gray"}, {"linestyle", ":"}, {"linewidth", "0.8"}, {"label", R"($\rho_l$)"}});
  };

#ifdef DFT_HAS_MATPLOTLIB

  // Plot 1: free-energy barrier Delta_Omega(R).
  {
    plt::figure_size(800, 500);
    plt::plot(R_scan, dOmega_scan, {{"color", "black"}, {"marker", "o"}, {"markersize", "4"}});
    plt::plot({R_star, R_star}, {0.0, barrier},
              {{"color", "red"}, {"linestyle", "--"}, {"linewidth", "0.8"}});
    plt::plot({R_star}, {barrier}, {{"color", "red"}, {"marker", "o"}, {"markersize", "7"}});

    // Mark sub-critical and super-critical initial radii.
    plt::plot({R_sub}, {0.0}, {{
        {"color", "#0055CC"}, {"marker", "v"}, {"markersize", "8"}, {"label", "$R_0 < R^*$"}
    }});
    plt::plot({R_sup}, {0.0}, {{
        {"color", "#CC2200"}, {"marker", "^"}, {"markersize", "8"}, {"label", "$R_0 > R^*$"}
    }});

    plt::xlabel(R"($R / \sigma$)");
    plt::ylabel(R"($\Delta\Omega\; [k_BT]$)");
    plt::title(R"(Nucleation barrier $\Delta\Omega(R)$ at $T^*=1.00$)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/nucleation_barrier.png");
    plt::close();
    std::cout << "\nPlot: " << std::filesystem::absolute("exports/nucleation_barrier.png") << "\n";
  }

  // Plot 2: dissolution dynamics (sub-critical, blue shades).
  {
    plt::figure_size(900, 550);
    plot_critical();
    plot_snapshots(sub_snaps, 0x00, 0x33, 0xAA, 0x88, 0xCC, 0xFF);
    plot_reference_lines();
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($\rho \sigma^3$)");
    plt::ylim(0.0, rho_l * 1.15);
    plt::title(R"(Dissolution: sub-critical droplet ($R_0 < R^*$))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/dissolution.png");
    plt::close();
    std::cout << "Plot: " << std::filesystem::absolute("exports/dissolution.png") << "\n";
  }

  // Plot 3: growth dynamics — small overshoot (red shades).
  {
    plt::figure_size(900, 550);
    plot_critical();
    plot_snapshots(sup_snaps, 0xAA, 0x00, 0x00, 0xFF, 0xAA, 0x77);
    plot_reference_lines();
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($\rho \sigma^3$)");
    plt::ylim(0.0, rho_l * 1.15);
    char title3[128];
    std::snprintf(title3, sizeof(title3),
                  R"(Growth: $R_0 = %.1f\sigma > R^*$)", R_sup);
    plt::title(title3);
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/growth.png");
    plt::close();
    std::cout << "Plot: " << std::filesystem::absolute("exports/growth.png") << "\n";
  }

  // Plot 4: growth dynamics — large overshoot (orange shades).
  {
    plt::figure_size(900, 550);
    plot_critical();
    plot_snapshots(sup2_snaps, 0x99, 0x33, 0x00, 0xFF, 0xBB, 0x44);
    plot_reference_lines();
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($\rho \sigma^3$)");
    plt::ylim(0.0, rho_l * 1.15);
    char title4[128];
    std::snprintf(title4, sizeof(title4),
                  R"(Growth: $R_0 = %.1f\sigma \gg R^*$)", R_sup2);
    plt::title(title4);
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/growth_large.png");
    plt::close();
    std::cout << "Plot: " << std::filesystem::absolute("exports/growth_large.png") << "\n";
  }

#endif

  std::cout << "\nDone.\n";
  return 0;
}
