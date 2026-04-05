#include "dft.hpp"
#include "plot.hpp"
#include "utils.hpp"

#include <filesystem>
#include <iostream>
#include <numbers>
#include <print>
#include <vector>

using namespace dft;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  auto cfg = config::parse_config("config.ini", config::FileType::INI);

  // FIRE2 minimisation of a 2D quadratic.

  std::println(std::cout, "\n=== FIRE2: 2D quadratic minimisation ===\n");

  double fire_dt = config::get<double>(cfg, "fire2.dt");
  double fire_dt_max = config::get<double>(cfg, "fire2.dt_max");
  double fire_tol = config::get<double>(cfg, "fire2.force_tolerance");
  int fire_max_steps = config::get<int>(cfg, "fire2.max_steps");
  double fire_x0 = config::get<double>(cfg, "fire2.x0");
  double fire_y0 = config::get<double>(cfg, "fire2.y0");
  int fire_log_interval = config::get<int>(cfg, "fire2.log_interval");

  auto fire_force_fn = [](const std::vector<arma::vec>& x)
      -> std::pair<double, std::vector<arma::vec>> {
    double xv = x[0](0);
    double yv = x[0](1);
    double energy = (xv - 1.0) * (xv - 1.0) + 4.0 * (yv + 2.0) * (yv + 2.0);
    arma::vec force = {-2.0 * (xv - 1.0), -8.0 * (yv + 2.0)};
    return {energy, {force}};
  };

  algorithms::fire::Fire fire_config{
      .dt = fire_dt,
      .dt_max = fire_dt_max,
      .force_tolerance = fire_tol,
      .max_steps = fire_max_steps,
  };

  auto fire_result = fire_config.minimize({arma::vec{fire_x0, fire_y0}}, fire_force_fn);

  std::println(std::cout, "  Converged: {} after {} steps",
               fire_result.converged ? "true" : "false", fire_result.iteration);
  std::println(std::cout, "  Solution: ({}, {})", fire_result.x[0](0), fire_result.x[0](1));
  std::println(std::cout, "  Energy: {}", fire_result.energy);
  std::println(std::cout, "  RMS force: {}", fire_result.rms_force);

  // Step-by-step iteration for convergence logging.

  auto state = fire_config.initialize({arma::vec{fire_x0, fire_y0}}, fire_force_fn);
  auto [init_energy, forces] = fire_force_fn(state.x);

  std::vector<double> fire_steps, fire_energies;
  while (!state.converged && state.iteration < fire_config.max_steps) {
    auto [new_state, new_forces] = fire_config.step(std::move(state), forces, fire_force_fn);
    state = std::move(new_state);
    forces = std::move(new_forces);
    if (state.iteration % fire_log_interval == 0 || state.converged) {
      fire_steps.push_back(static_cast<double>(state.iteration));
      fire_energies.push_back(state.energy);
    }
  }

  // Split-operator DDFT: ideal gas relaxation to equilibrium.

  std::println(std::cout, "\n=== Split-operator DDFT: ideal gas relaxation ===\n");

  double ddft_dx = config::get<double>(cfg, "ddft.dx");
  double box_size = config::get<double>(cfg, "ddft.box_size");
  double temperature = config::get<double>(cfg, "ddft.temperature");
  double rho0 = config::get<double>(cfg, "ddft.rho0");
  double amplitude = config::get<double>(cfg, "ddft.amplitude");
  double ddft_dt = config::get<double>(cfg, "ddft.dt");
  double D = config::get<double>(cfg, "ddft.diffusion_coefficient");
  int n_steps = config::get<int>(cfg, "ddft.n_steps");
  int snapshot_interval = config::get<int>(cfg, "ddft.snapshot_interval");

  physics::Model model{
      .grid = make_grid(ddft_dx, {box_size, box_size, box_size}),
      .species = {Species{.name = "ideal", .hard_sphere_diameter = 0.0}},
      .interactions = {},
      .temperature = temperature,
  };

  long nx = model.grid.shape[0];
  long ny = model.grid.shape[1];
  long nz = model.grid.shape[2];

  arma::vec z_vals = arma::linspace(0.0, (nz - 1) * model.grid.dx, nz);
  arma::vec z_profile = rho0 + amplitude * arma::sin(2.0 * std::numbers::pi * z_vals / model.grid.box_size[2]);
  arma::vec rho = arma::repmat(z_profile, nx * ny, 1);

  auto ddft_state = init::from_profile(model, rho);

  auto ddft_force_fn = [&](const std::vector<arma::vec>& densities)
      -> std::pair<double, std::vector<arma::vec>> {
    double dv = model.grid.cell_volume();
    arma::vec rho_safe = arma::clamp(densities[0], 1e-18, arma::datum::inf);
    double energy = dv * arma::dot(rho_safe, arma::log(rho_safe) - 1.0);
    arma::vec force = dv * arma::log(rho_safe);
    return {energy, {force}};
  };

  algorithms::dynamics::Simulation sim_config{
      .step = {.dt = ddft_dt, .diffusion_coefficient = D, .min_density = 1e-18},
      .n_steps = n_steps,
      .snapshot_interval = snapshot_interval,
      .log_interval = snapshot_interval,
  };

  auto sim = sim_config.run({ddft_state.species[0].density.values}, model.grid, ddft_force_fn);

  // Collect z-profile snapshots.

  auto z_coords = arma::conv_to<std::vector<double>>::from(z_vals);
  std::vector<std::vector<double>> profile_snapshots;
  std::vector<double> snapshot_times;

  for (const auto& snap : sim.snapshots) {
    profile_snapshots.push_back(utils::extract_z_profile(snap.densities[0], nx, ny, nz));
    snapshot_times.push_back(snap.time);
  }

  // Variance decay.

  std::vector<double> ddft_times, ddft_variances;
  for (const auto& snap : sim.snapshots) {
    ddft_times.push_back(snap.time);
    ddft_variances.push_back(arma::var(snap.densities[0]));
  }

  std::println(std::cout, "{:>10s}{:>16s}", "Time", "Variance");
  std::println(std::cout, "{}", std::string(26, '-'));
  for (std::size_t i = 0; i < ddft_times.size(); ++i) {
    std::println(std::cout, "{:>10.4f}{:>16.4e}", ddft_times[i], ddft_variances[i]);
  }

  std::println(std::cout, "\n  Mass initial: {:.6f}", sim.mass_initial);
  std::println(std::cout, "  Mass final:   {:.6f}", sim.mass_final);
  std::println(std::cout, "  Rel. error:   {:.6e}",
               std::abs(sim.mass_final - sim.mass_initial) / sim.mass_initial);

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(fire_steps, fire_energies,
                   ddft_times, ddft_variances,
                   z_coords, profile_snapshots, snapshot_times, rho0);
#endif
}
