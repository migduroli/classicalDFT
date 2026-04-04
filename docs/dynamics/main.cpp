#include "dft.hpp"
#include "plot.hpp"
#include "utils.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

using namespace dft;

// FIRE2 minimisation of a 2D quadratic.

void demo_fire2(const nlohmann::json& cfg) {
  std::cout << "\n=== FIRE2: 2D quadratic minimisation ===\n\n";

  double dt = config::get<double>(cfg, "fire2.dt");
  double dt_max = config::get<double>(cfg, "fire2.dt_max");
  double force_tolerance = config::get<double>(cfg, "fire2.force_tolerance");
  int max_steps = config::get<int>(cfg, "fire2.max_steps");
  double x0 = config::get<double>(cfg, "fire2.x0");
  double y0 = config::get<double>(cfg, "fire2.y0");
  int log_interval = config::get<int>(cfg, "fire2.log_interval");

  // Minimize f(x,y) = (x-1)^2 + 4*(y+2)^2.

  auto force_fn = [](const std::vector<arma::vec>& x)
      -> std::pair<double, std::vector<arma::vec>> {
    double xv = x[0](0);
    double yv = x[0](1);
    double energy = (xv - 1.0) * (xv - 1.0) + 4.0 * (yv + 2.0) * (yv + 2.0);
    arma::vec force = {-2.0 * (xv - 1.0), -8.0 * (yv + 2.0)};
    return {energy, {force}};
  };

  algorithms::fire::FireConfig fire_config{
      .dt = dt,
      .dt_max = dt_max,
      .force_tolerance = force_tolerance,
      .max_steps = max_steps,
  };

  auto result = algorithms::fire::minimize({arma::vec{x0, y0}}, force_fn, fire_config);

  std::cout << "  Converged: " << std::boolalpha << result.converged
            << " after " << result.iteration << " steps\n";
  std::cout << "  Solution: (" << result.x[0](0) << ", " << result.x[0](1) << ")\n";
  std::cout << "  Energy: " << result.energy << "\n";
  std::cout << "  RMS force: " << result.rms_force << "\n";

  // Step-by-step iteration for convergence logging.

  auto state = algorithms::fire::initialize({arma::vec{x0, y0}}, force_fn, fire_config);
  auto [init_energy, forces] = force_fn(state.x);

  std::vector<double> steps, energies;
  while (!state.converged && state.iteration < fire_config.max_steps) {
    auto [new_state, new_forces] = algorithms::fire::step(std::move(state), forces, force_fn, fire_config);
    state = std::move(new_state);
    forces = std::move(new_forces);
    if (state.iteration % log_interval == 0 || state.converged) {
      steps.push_back(static_cast<double>(state.iteration));
      energies.push_back(state.energy);
    }
  }

#ifdef DFT_HAS_MATPLOTLIB
  plot::fire2_energy(steps, energies);
#endif
}

// Split-operator DDFT: ideal gas relaxation to equilibrium.

void demo_ddft(const nlohmann::json& cfg) {
  std::cout << "\n=== Split-operator DDFT: ideal gas relaxation ===\n\n";

  double dx = config::get<double>(cfg, "ddft.dx");
  double box_size = config::get<double>(cfg, "ddft.box_size");
  double temperature = config::get<double>(cfg, "ddft.temperature");
  double rho0 = config::get<double>(cfg, "ddft.rho0");
  double amplitude = config::get<double>(cfg, "ddft.amplitude");
  double dt = config::get<double>(cfg, "ddft.dt");
  double D = config::get<double>(cfg, "ddft.diffusion_coefficient");
  int n_steps = config::get<int>(cfg, "ddft.n_steps");
  int snapshot_interval = config::get<int>(cfg, "ddft.snapshot_interval");

  // Define the system as a Model (ideal gas: no interactions).

  physics::Model model{
      .grid = make_grid(dx, {box_size, box_size, box_size}),
      .species = {Species{.name = "ideal", .hard_sphere_diameter = 0.0}},
      .interactions = {},
      .temperature = temperature,
  };

  // Sinusoidal density perturbation along z, built with Armadillo vectorisation.

  long nx = model.grid.shape[0];
  long ny = model.grid.shape[1];
  long nz = model.grid.shape[2];

  arma::vec z_vals = arma::linspace(0.0, (nz - 1) * model.grid.dx, nz);
  arma::vec z_profile = rho0 + amplitude * arma::sin(2.0 * std::numbers::pi * z_vals / model.grid.box_size[2]);
  arma::vec rho = arma::repmat(z_profile, nx * ny, 1);

  auto state = init::from_profile(model, rho);

  // Ideal gas force function.

  auto force_fn = [&](const std::vector<arma::vec>& densities)
      -> std::pair<double, std::vector<arma::vec>> {
    double dv = model.grid.cell_volume();
    arma::vec rho_safe = arma::clamp(densities[0], 1e-18, arma::datum::inf);
    double energy = dv * arma::dot(rho_safe, arma::log(rho_safe) - 1.0);
    arma::vec force = dv * arma::log(rho_safe);
    return {energy, {force}};
  };

  // DDFT relaxation via the simulate API.

  algorithms::ddft::SimulationConfig sim_config{
      .ddft = {.dt = dt, .diffusion_coefficient = D, .min_density = 1e-18},
      .n_steps = n_steps,
      .snapshot_interval = snapshot_interval,
      .log_interval = snapshot_interval,
  };

  auto sim = algorithms::ddft::simulate({state.species[0].density.values}, model.grid, force_fn, sim_config);

  // Collect z-profile snapshots.

  auto z_coords = arma::conv_to<std::vector<double>>::from(z_vals);
  std::vector<std::vector<double>> profile_snapshots;
  std::vector<double> snapshot_times;

  for (const auto& snap : sim.snapshots) {
    profile_snapshots.push_back(utils::extract_z_profile(snap.densities[0], nx, ny, nz));
    snapshot_times.push_back(snap.time);
  }

  // Variance decay.

  std::vector<double> times, variances;
  for (const auto& snap : sim.snapshots) {
    times.push_back(snap.time);
    variances.push_back(arma::var(snap.densities[0]));
  }

  std::cout << std::setw(10) << "Time" << std::setw(16) << "Variance\n";
  std::cout << std::string(26, '-') << "\n";
  for (std::size_t i = 0; i < times.size(); ++i) {
    std::cout << std::setw(10) << std::setprecision(4) << times[i]
              << std::setw(16) << std::scientific << variances[i] << "\n";
  }

  std::cout << "\n  Mass initial: " << std::fixed << sim.mass_initial << "\n";
  std::cout << "  Mass final:   " << sim.mass_final << "\n";
  std::cout << "  Rel. error:   " << std::abs(sim.mass_final - sim.mass_initial) / sim.mass_initial << "\n";

#ifdef DFT_HAS_MATPLOTLIB
  plot::ddft_variance(times, variances);
  plot::ddft_density_profiles(z_coords, profile_snapshots, snapshot_times, rho0);
#endif
}

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  auto cfg = config::parse_config("config.ini", config::FileType::INI);
  demo_fire2(cfg);
  demo_ddft(cfg);
}
