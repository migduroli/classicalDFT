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
  const auto& dens = solver.density(0);
  const auto& shape = dens.shape();
  std::vector<double> profile(static_cast<size_t>(shape[0]));
  for (long ix = 0; ix < shape[0]; ++ix) {
    double sum = 0.0;
    for (long iy = 0; iy < shape[1]; ++iy) {
      for (long iz = 0; iz < shape[2]; ++iz) {
        sum += dens.values()(dens.flat_index(ix, iy, iz));
      }
    }
    profile[static_cast<size_t>(ix)] = sum / static_cast<double>(shape[1] * shape[2]);
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

static void demo_fire2(const config::ConfigParser& cfg) {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  FIRE2 minimisation: ideal gas relaxation to equilibrium\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double dx = cfg.get<double>("grid.dx");
  double length = cfg.get<double>("fire2.length");
  double rho_initial = cfg.get<double>("fire2.rho_initial");
  double rho_target = cfg.get<double>("fire2.rho_target");
  int max_steps = static_cast<int>(cfg.get<double>("fire2.max_steps"));

  auto solver = make_1d_solver(dx, length, rho_initial);
  solver.species(0).set_chemical_potential(std::log(rho_target));

  dynamics::Fire2Config fconf{
      .dt = cfg.get<double>("fire2.dt"),
      .dt_max = cfg.get<double>("fire2.dt_max"),
      .force_limit = cfg.get<double>("fire2.force_limit"),
      .min_density = cfg.get<double>("fire2.min_density"),
  };
  dynamics::Fire2Minimizer fire(solver, fconf);

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

  bool converged = fire.run(max_steps);

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

static void demo_split_operator(const config::ConfigParser& cfg) {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  Split-operator integrator: density relaxation dynamics\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double dx = cfg.get<double>("grid.dx");
  double length = cfg.get<double>("split_operator.length");
  double rho0 = cfg.get<double>("split_operator.rho0");
  double amplitude = cfg.get<double>("split_operator.amplitude");

  auto solver = make_perturbed_solver(dx, length, rho0, amplitude);
  solver.species(0).set_chemical_potential(std::log(rho0));

  auto x = extract_x_grid(solver);
  auto rho_initial = extract_profile(solver);

  dynamics::IntegratorConfig iconf{
      .scheme = dynamics::IntegrationScheme::SplitOperator,
      .dt = cfg.get<double>("split_operator.dt"),
      .diffusion_coefficient = cfg.get<double>("split_operator.diffusion_coefficient"),
      .force_limit = cfg.get<double>("split_operator.force_limit"),
  };
  dynamics::Integrator integrator(solver, iconf);

  struct Snapshot {
    double time;
    std::vector<double> profile;
  };
  std::vector<Snapshot> snapshots;
  snapshots.push_back({0.0, rho_initial});

  double total_time = cfg.get<double>("split_operator.total_time");
  int n_snapshots = static_cast<int>(cfg.get<double>("split_operator.n_snapshots"));
  double snapshot_interval = total_time / n_snapshots;
  int steps_per_snapshot = static_cast<int>(snapshot_interval / iconf.dt);

  for (int snap = 0; snap < n_snapshots; ++snap) {
    (void)integrator.resume(steps_per_snapshot);
    double time = static_cast<double>((snap + 1) * steps_per_snapshot) * iconf.dt;
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

static void demo_crank_nicholson(const config::ConfigParser& cfg) {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  Crank-Nicholson integrator: density relaxation dynamics\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double dx = cfg.get<double>("grid.dx");
  double length = cfg.get<double>("crank_nicholson.length");
  double rho0 = cfg.get<double>("crank_nicholson.rho0");
  double amplitude = cfg.get<double>("crank_nicholson.amplitude");

  auto solver = make_perturbed_solver(dx, length, rho0, amplitude);
  solver.species(0).set_chemical_potential(std::log(rho0));

  auto x = extract_x_grid(solver);
  auto rho_initial = extract_profile(solver);

  dynamics::IntegratorConfig iconf{
      .scheme = dynamics::IntegrationScheme::CrankNicholson,
      .dt = cfg.get<double>("crank_nicholson.dt"),
      .diffusion_coefficient = cfg.get<double>("crank_nicholson.diffusion_coefficient"),
      .force_limit = cfg.get<double>("crank_nicholson.force_limit"),
      .crank_nicholson_iterations = static_cast<int>(cfg.get<double>("crank_nicholson.cn_iterations")),
      .cn_tolerance = cfg.get<double>("crank_nicholson.cn_tolerance"),
  };
  dynamics::Integrator integrator(solver, iconf);

  struct Snapshot {
    double time;
    std::vector<double> profile;
  };
  std::vector<Snapshot> snapshots;
  snapshots.push_back({0.0, rho_initial});

  double total_time = cfg.get<double>("crank_nicholson.total_time");
  int n_snapshots = static_cast<int>(cfg.get<double>("crank_nicholson.n_snapshots"));
  double snapshot_interval = total_time / n_snapshots;
  int steps_per_snapshot = static_cast<int>(snapshot_interval / iconf.dt);

  for (int snap = 0; snap < n_snapshots; ++snap) {
    (void)integrator.resume(steps_per_snapshot);
    double time = static_cast<double>((snap + 1) * steps_per_snapshot) * iconf.dt;
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

  std::cout << "\nCrank-Nicholson allows larger timestep (dt = " << iconf.dt << ") compared to\n"
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

static void demo_comparison(const config::ConfigParser& cfg) {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  Comparison: split-operator vs Crank-Nicholson\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double dx = cfg.get<double>("grid.dx");
  double length = cfg.get<double>("comparison.length");
  double rho0 = cfg.get<double>("comparison.rho0");
  double amplitude = cfg.get<double>("comparison.amplitude");
  double total_time = cfg.get<double>("comparison.total_time");

  // Run split-operator
  auto solver_so = make_perturbed_solver(dx, length, rho0, amplitude);
  solver_so.species(0).set_chemical_potential(std::log(rho0));

  dynamics::IntegratorConfig iconf_so{
      .scheme = dynamics::IntegrationScheme::SplitOperator,
      .dt = cfg.get<double>("split_operator.dt"),
      .diffusion_coefficient = cfg.get<double>("split_operator.diffusion_coefficient"),
      .force_limit = cfg.get<double>("split_operator.force_limit"),
  };
  dynamics::Integrator integrator_so(solver_so, iconf_so);

  int so_total_steps = static_cast<int>(total_time / iconf_so.dt);
  int so_sample_every = so_total_steps / 40;
  std::vector<double> so_steps, so_variance;
  for (int step = 0; step <= so_total_steps; step += so_sample_every) {
    if (step > 0) (void)integrator_so.resume(so_sample_every);
    double var = arma::var(solver_so.density(0).values());
    so_steps.push_back(static_cast<double>(step) * iconf_so.dt);
    so_variance.push_back(var);
  }

  // Run Crank-Nicholson with larger timestep
  auto solver_cn = make_perturbed_solver(dx, length, rho0, amplitude);
  solver_cn.species(0).set_chemical_potential(std::log(rho0));

  dynamics::IntegratorConfig iconf_cn{
      .scheme = dynamics::IntegrationScheme::CrankNicholson,
      .dt = cfg.get<double>("crank_nicholson.dt"),
      .diffusion_coefficient = cfg.get<double>("crank_nicholson.diffusion_coefficient"),
      .force_limit = cfg.get<double>("crank_nicholson.force_limit"),
      .crank_nicholson_iterations = static_cast<int>(cfg.get<double>("crank_nicholson.cn_iterations")),
      .cn_tolerance = cfg.get<double>("crank_nicholson.cn_tolerance"),
  };
  dynamics::Integrator integrator_cn(solver_cn, iconf_cn);

  int cn_total_steps = static_cast<int>(total_time / iconf_cn.dt);
  int cn_sample_every = std::max(cn_total_steps / 40, 1);
  std::vector<double> cn_steps, cn_variance;
  for (int step = 0; step <= cn_total_steps; step += cn_sample_every) {
    if (step > 0) (void)integrator_cn.resume(cn_sample_every);
    double var = arma::var(solver_cn.density(0).values());
    cn_steps.push_back(static_cast<double>(step) * iconf_cn.dt);
    cn_variance.push_back(var);
  }

  // Print comparison
  std::cout << "Split-operator (dt = " << iconf_so.dt << "): " << so_total_steps << " steps\n"
            << "  Final variance: " << std::scientific << so_variance.back() << "\n\n"
            << "Crank-Nicholson (dt = " << iconf_cn.dt << "): " << cn_total_steps << " steps\n"
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

int main(int argc, char* argv[]) {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  std::string config_path = (argc > 1) ? argv[1] : "config.ini";
  auto cfg = config::ConfigParser(config_path);

  demo_fire2(cfg);
  demo_split_operator(cfg);
  demo_crank_nicholson(cfg);
  demo_comparison(cfg);

  std::cout << "\nAll dynamics demos completed.\n";
  return 0;
}
