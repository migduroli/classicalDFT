#include "dft.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

using namespace dft;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Create a 1D-like solver (N x 1 x 1) for clear visualisation.
static Solver make_1d_solver(double dx, double length, double rho0) {
  auto d = density::Density(dx, {length, dx, dx});
  d.values().fill(rho0);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));
  return solver;
}

/// Create a 1D solver with a sinusoidal density perturbation.
static Solver make_perturbed_solver(double dx, double length, double rho0, double amplitude) {
  auto d = density::Density(dx, {length, dx, dx});
  long nx = d.shape()[0];
  arma::vec rho(d.size());
  for (arma::uword i = 0; i < d.size(); ++i) {
    long ix = static_cast<long>(i) / (d.shape()[1] * d.shape()[2]);
    double x = ix * dx;
    rho(i) = rho0 + amplitude * std::sin(2.0 * std::numbers::pi * x / length);
  }
  d.set(rho);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));
  return solver;
}

/// Extract the 1D density profile (x-direction) from a solver.
static std::vector<double> extract_profile(const Solver& solver) {
  const auto& rho = solver.density(0).values();
  long nx = solver.density(0).shape()[0];
  long ny = solver.density(0).shape()[1];
  long nz = solver.density(0).shape()[2];
  std::vector<double> profile(static_cast<size_t>(nx));
  for (long ix = 0; ix < nx; ++ix) {
    // Average over y and z (should be trivial for 1D-like grids)
    double sum = 0.0;
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        sum += rho(static_cast<arma::uword>(ix * ny * nz + iy * nz + iz));
      }
    }
    profile[static_cast<size_t>(ix)] = sum / static_cast<double>(ny * nz);
  }
  return profile;
}

/// Extract the x-coordinate grid.
static std::vector<double> extract_x_grid(const Solver& solver) {
  double dx = solver.density(0).dx();
  long nx = solver.density(0).shape()[0];
  std::vector<double> x(static_cast<size_t>(nx));
  for (long i = 0; i < nx; ++i) {
    x[static_cast<size_t>(i)] = i * dx;
  }
  return x;
}

// ── Demo 1: FIRE2 minimisation ───────────────────────────────────────────

static void demo_fire2() {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  FIRE2 minimisation: ideal gas relaxation to equilibrium\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double dx = 0.5;
  double length = 20.0;
  double rho_initial = 0.3;
  double rho_target = 0.7;

  auto solver = make_1d_solver(dx, length, rho_initial);
  solver.species(0).set_chemical_potential(std::log(rho_target));

  dynamics::Fire2Config config{
      .dt = 1e-3, .dt_max = 0.1, .force_limit = 1e-8, .min_density = 1e-30};
  dynamics::Fire2Minimizer fire(solver, config);

  // Track convergence
  std::vector<double> steps, energies, forces;
  fire.set_step_callback([&](long step, double energy, double max_force) {
    if (step % 10 == 0 || step <= 5) {
      steps.push_back(static_cast<double>(step));
      energies.push_back(energy);
      forces.push_back(max_force);
    }
    return true;
  });

  bool converged = fire.run(500);

  // Print convergence table
  std::cout << std::setw(8) << "Step" << std::setw(16) << "Energy" << std::setw(16) << "Max force\n";
  std::cout << std::string(40, '-') << "\n";
  for (size_t i = 0; i < steps.size(); ++i) {
    std::cout << std::setw(8) << static_cast<int>(steps[i]) << std::setw(16) << std::scientific
              << std::setprecision(6) << energies[i] << std::setw(16) << forces[i] << "\n";
  }
  std::cout << "\nConverged: " << std::boolalpha << converged << " after " << fire.step_count() << " steps\n";

  // Check final density
  const arma::vec& rho = solver.density(0).values();
  double mean_rho = arma::mean(rho);
  std::cout << "Target density: " << rho_target << ", final mean density: " << std::fixed
            << std::setprecision(6) << mean_rho << "\n";

#ifdef DFT_HAS_MATPLOTLIB
  // Plot energy convergence
  plt::figure_size(700, 500);
  plt::named_plot("Energy", steps, energies, "b-");
  plt::xlabel("Step");
  plt::ylabel("Free energy");
  plt::title("FIRE2: energy convergence");
  plt::legend();
  plt::tight_layout();
  plt::save("exports/fire2_energy.png");
  std::cout << "Plot saved: " << std::filesystem::absolute("exports/fire2_energy.png") << "\n";
  plt::close();

  // Plot force convergence
  plt::figure_size(700, 500);
  plt::named_plot("Max force", steps, forces, "r-");
  plt::xlabel("Step");
  plt::ylabel("Max |force|");
  plt::title("FIRE2: force convergence");
  plt::legend();
  plt::tight_layout();
  plt::save("exports/fire2_force.png");
  std::cout << "Plot saved: " << std::filesystem::absolute("exports/fire2_force.png") << "\n";
  plt::close();
#endif
}

// ── Demo 2: Split-operator density dynamics ──────────────────────────────

static void demo_split_operator() {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  Split-operator integrator: density relaxation dynamics\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double dx = 0.5;
  double length = 10.0;
  double rho0 = 0.5;
  double amplitude = 0.2;

  auto solver = make_perturbed_solver(dx, length, rho0, amplitude);
  solver.species(0).set_chemical_potential(std::log(rho0));

  auto x = extract_x_grid(solver);
  auto rho_initial = extract_profile(solver);

  dynamics::IntegratorConfig config{
      .scheme = dynamics::IntegrationScheme::SplitOperator,
      .dt = 5e-3,
      .diffusion_coefficient = 1.0,
      .force_limit = 1e-12,
  };
  dynamics::Integrator integrator(solver, config);

  // Capture snapshots at various times: t = 0, 2, 4, 6, 8, 10
  struct Snapshot {
    double time;
    std::vector<double> profile;
  };
  std::vector<Snapshot> snapshots;
  snapshots.push_back({0.0, rho_initial});

  double total_time = 10.0;
  int n_snapshots = 10;
  double snapshot_interval = total_time / n_snapshots;
  int steps_per_snapshot = static_cast<int>(snapshot_interval / config.dt);

  for (int snap = 0; snap < n_snapshots; ++snap) {
    (void)integrator.resume(steps_per_snapshot);
    double time = static_cast<double>((snap + 1) * steps_per_snapshot) * config.dt;
    snapshots.push_back({time, extract_profile(solver)});
  }

  // Print variance decay
  std::cout << std::setw(10) << "Time" << std::setw(16) << "Variance\n";
  std::cout << std::string(26, '-') << "\n";
  for (const auto& snap : snapshots) {
    double mean = 0.0;
    for (double v : snap.profile) mean += v;
    mean /= static_cast<double>(snap.profile.size());
    double var = 0.0;
    for (double v : snap.profile) var += (v - mean) * (v - mean);
    var /= static_cast<double>(snap.profile.size());
    std::cout << std::setw(10) << std::scientific << std::setprecision(3) << snap.time << std::setw(16)
              << var << "\n";
  }

#ifdef DFT_HAS_MATPLOTLIB
  plt::figure_size(700, 500);
  std::vector<std::string> colors = {"k-", "b--", "g--", "r--", "m--"};
  for (size_t i = 0; i < snapshots.size(); ++i) {
    std::string label = (i == 0) ? "t = 0" : "t = " + std::to_string(snapshots[i].time);
    plt::named_plot(label, x, snapshots[i].profile, colors[i % colors.size()]);
  }
  plt::xlabel("x");
  plt::ylabel("rho(x)");
  plt::title("Split-operator: density relaxation");
  plt::legend();
  plt::tight_layout();
  plt::save("exports/split_operator_dynamics.png");
  std::cout << "\nPlot saved: " << std::filesystem::absolute("exports/split_operator_dynamics.png") << "\n";
  plt::close();
#endif
}

// ── Demo 3: Crank-Nicholson density dynamics ─────────────────────────────

static void demo_crank_nicholson() {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  Crank-Nicholson integrator: density relaxation dynamics\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double dx = 0.5;
  double length = 10.0;
  double rho0 = 0.5;
  double amplitude = 0.2;

  auto solver = make_perturbed_solver(dx, length, rho0, amplitude);
  solver.species(0).set_chemical_potential(std::log(rho0));

  auto x = extract_x_grid(solver);
  auto rho_initial = extract_profile(solver);

  dynamics::IntegratorConfig config{
      .scheme = dynamics::IntegrationScheme::CrankNicholson,
      .dt = 2.5e-2,
      .diffusion_coefficient = 1.0,
      .force_limit = 1e-12,
      .crank_nicholson_iterations = 5,
      .cn_tolerance = 1e-10,
  };
  dynamics::Integrator integrator(solver, config);

  struct Snapshot {
    double time;
    std::vector<double> profile;
  };
  std::vector<Snapshot> snapshots;
  snapshots.push_back({0.0, rho_initial});

  double total_time = 10.0;
  int n_snapshots = 10;
  double snapshot_interval = total_time / n_snapshots;
  int steps_per_snapshot = static_cast<int>(snapshot_interval / config.dt);

  for (int snap = 0; snap < n_snapshots; ++snap) {
    (void)integrator.resume(steps_per_snapshot);
    double time = static_cast<double>((snap + 1) * steps_per_snapshot) * config.dt;
    snapshots.push_back({time, extract_profile(solver)});
  }

  // Print variance decay
  std::cout << std::setw(10) << "Time" << std::setw(16) << "Variance\n";
  std::cout << std::string(26, '-') << "\n";
  for (const auto& snap : snapshots) {
    double mean = 0.0;
    for (double v : snap.profile) mean += v;
    mean /= static_cast<double>(snap.profile.size());
    double var = 0.0;
    for (double v : snap.profile) var += (v - mean) * (v - mean);
    var /= static_cast<double>(snap.profile.size());
    std::cout << std::setw(10) << std::scientific << std::setprecision(3) << snap.time << std::setw(16)
              << var << "\n";
  }

  std::cout << "\nCrank-Nicholson allows larger timestep (dt = " << config.dt << ") compared to\n"
            << "split-operator, while maintaining stability and second-order accuracy.\n";

#ifdef DFT_HAS_MATPLOTLIB
  plt::figure_size(700, 500);
  std::vector<std::string> colors = {"k-", "b--", "g--", "r--", "m--"};
  for (size_t i = 0; i < snapshots.size(); ++i) {
    std::string label = (i == 0) ? "t = 0" : "t = " + std::to_string(snapshots[i].time);
    plt::named_plot(label, x, snapshots[i].profile, colors[i % colors.size()]);
  }
  plt::xlabel("x");
  plt::ylabel("rho(x)");
  plt::title("Crank-Nicholson: density relaxation");
  plt::legend();
  plt::tight_layout();
  plt::save("exports/crank_nicholson_dynamics.png");
  std::cout << "\nPlot saved: " << std::filesystem::absolute("exports/crank_nicholson_dynamics.png") << "\n";
  plt::close();
#endif
}

// ── Demo 4: Comparing both schemes ──────────────────────────────────────

static void demo_comparison() {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  Comparison: split-operator vs Crank-Nicholson\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double dx = 0.5;
  double length = 10.0;
  double rho0 = 0.5;
  double amplitude = 0.2;
  double total_time = 10.0;

  // Run split-operator
  auto solver_so = make_perturbed_solver(dx, length, rho0, amplitude);
  solver_so.species(0).set_chemical_potential(std::log(rho0));

  dynamics::IntegratorConfig config_so{
      .scheme = dynamics::IntegrationScheme::SplitOperator,
      .dt = 5e-3,
      .diffusion_coefficient = 1.0,
      .force_limit = 1e-12,
  };
  dynamics::Integrator integrator_so(solver_so, config_so);

  int so_total_steps = static_cast<int>(total_time / config_so.dt);
  int so_sample_every = so_total_steps / 40;
  std::vector<double> so_steps, so_variance;
  for (int step = 0; step <= so_total_steps; step += so_sample_every) {
    if (step > 0) (void)integrator_so.resume(so_sample_every);
    double var = arma::var(solver_so.density(0).values());
    so_steps.push_back(static_cast<double>(step) * config_so.dt);
    so_variance.push_back(var);
  }

  // Run Crank-Nicholson with larger timestep
  auto solver_cn = make_perturbed_solver(dx, length, rho0, amplitude);
  solver_cn.species(0).set_chemical_potential(std::log(rho0));

  dynamics::IntegratorConfig config_cn{
      .scheme = dynamics::IntegrationScheme::CrankNicholson,
      .dt = 2.5e-2,
      .diffusion_coefficient = 1.0,
      .force_limit = 1e-12,
      .crank_nicholson_iterations = 5,
      .cn_tolerance = 1e-10,
  };
  dynamics::Integrator integrator_cn(solver_cn, config_cn);

  int cn_total_steps = static_cast<int>(total_time / config_cn.dt);
  int cn_sample_every = std::max(cn_total_steps / 40, 1);
  std::vector<double> cn_steps, cn_variance;
  for (int step = 0; step <= cn_total_steps; step += cn_sample_every) {
    if (step > 0) (void)integrator_cn.resume(cn_sample_every);
    double var = arma::var(solver_cn.density(0).values());
    cn_steps.push_back(static_cast<double>(step) * config_cn.dt);
    cn_variance.push_back(var);
  }

  // Print comparison
  std::cout << "Split-operator (dt = " << config_so.dt << "): " << so_total_steps << " steps\n"
            << "  Final variance: " << std::scientific << so_variance.back() << "\n\n"
            << "Crank-Nicholson (dt = " << config_cn.dt << "): " << cn_total_steps << " steps\n"
            << "  Final variance: " << std::scientific << cn_variance.back() << "\n\n"
            << "Both schemes reach comparable results. Crank-Nicholson uses 5x\n"
            << "fewer steps by tolerating a larger timestep.\n";

#ifdef DFT_HAS_MATPLOTLIB
  plt::figure_size(700, 500);
  plt::named_plot("Split-operator", so_steps, so_variance, "b-");
  plt::named_plot("Crank-Nicholson", cn_steps, cn_variance, "r--");
  plt::xlabel("Time");
  plt::ylabel("Density variance");
  plt::title("Variance decay: split-operator vs Crank-Nicholson");
  plt::legend();
  plt::tight_layout();
  plt::save("exports/scheme_comparison.png");
  std::cout << "\nPlot saved: " << std::filesystem::absolute("exports/scheme_comparison.png") << "\n";
  plt::close();
#endif
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  demo_fire2();
  demo_split_operator();
  demo_crank_nicholson();
  demo_comparison();

  std::cout << "\nAll dynamics demos completed.\n";
  return 0;
}
