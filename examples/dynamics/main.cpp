#include "dft.hpp"
#include "plot.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

using namespace dft;

// FIRE2 minimisation of a 2D quadratic.

void demo_fire2() {
  std::cout << "\n=== FIRE2: 2D quadratic minimisation ===\n\n";

  // Minimize f(x,y) = (x-1)^2 + 4*(y+2)^2.

  auto force_fn = [](const std::vector<arma::vec>& x)
      -> std::pair<double, std::vector<arma::vec>> {
    double xv = x[0](0);
    double yv = x[0](1);
    double energy = (xv - 1.0) * (xv - 1.0) + 4.0 * (yv + 2.0) * (yv + 2.0);
    arma::vec force = {-2.0 * (xv - 1.0), -8.0 * (yv + 2.0)};
    return {energy, {force}};
  };

  algorithms::fire::FireConfig config{
      .dt = 1e-3,
      .dt_max = 0.1,
      .force_tolerance = 1e-10,
      .max_steps = 5000,
  };

  auto result = algorithms::fire::minimize({arma::vec{5.0, 5.0}}, force_fn, config);

  std::cout << "  Converged: " << std::boolalpha << result.converged
            << " after " << result.iteration << " steps\n";
  std::cout << "  Solution: (" << result.x[0](0) << ", " << result.x[0](1) << ")\n";
  std::cout << "  Energy: " << result.energy << "\n";
  std::cout << "  RMS force: " << result.rms_force << "\n";

  // Step-by-step iteration for convergence logging.

  auto state = algorithms::fire::initialize({arma::vec{5.0, 5.0}}, force_fn, config);
  auto [init_energy, forces] = force_fn(state.x);

  std::vector<double> steps, energies;
  while (!state.converged && state.iteration < config.max_steps) {
    auto [new_state, new_forces] = algorithms::fire::step(std::move(state), forces, force_fn, config);
    state = std::move(new_state);
    forces = std::move(new_forces);
    if (state.iteration % 50 == 0 || state.converged) {
      steps.push_back(static_cast<double>(state.iteration));
      energies.push_back(state.energy);
    }
  }

#ifdef DFT_HAS_MATPLOTLIB
  plot::fire2_energy(steps, energies);
#endif
}

// Split-operator DDFT: ideal gas relaxation to equilibrium.

void demo_ddft() {
  std::cout << "\n=== Split-operator DDFT: ideal gas relaxation ===\n\n";

  // Define the system as a Model (ideal gas: no interactions).

  physics::Model model{
      .grid = make_grid(0.5, {8.0, 8.0, 8.0}),
      .species = {Species{.name = "ideal", .hard_sphere_diameter = 0.0}},
      .interactions = {},
      .temperature = 1.0,
  };

  // Sinusoidal density perturbation along z, built with Armadillo vectorisation.

  double rho0 = 0.5;
  double amplitude = 0.2;
  long nx = model.grid.shape[0];
  long ny = model.grid.shape[1];
  long nz = model.grid.shape[2];

  // 1D z-profile, then replicate across all (x, y) planes.
  // Flat layout is z-fastest: index = iz + nz*(iy + ny*ix).
  // A full z-period repeats every nz entries, so tile it nx*ny times.
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

  algorithms::ddft::DdftConfig dconf{
      .dt = 1e-3,
      .diffusion_coefficient = 1.0,
      .min_density = 1e-18,
  };

  auto k2 = algorithms::ddft::compute_k_squared(model.grid);
  auto prop = algorithms::ddft::diffusion_propagator(k2, dconf.diffusion_coefficient, dconf.dt);

  std::vector<arma::vec> densities = {state.species[0].density.values};
  int n_steps = 500;
  int snapshot_interval = 100;

  std::vector<double> times, variances;
  times.push_back(0.0);
  variances.push_back(arma::var(densities[0]));

  for (int step = 1; step <= n_steps; ++step) {
    auto result = algorithms::ddft::split_operator_step(
        densities, model.grid, k2, prop, force_fn, dconf
    );
    densities = std::move(result.densities);

    if (step % snapshot_interval == 0) {
      times.push_back(step * dconf.dt);
      variances.push_back(arma::var(densities[0]));
    }
  }

  std::cout << std::setw(10) << "Time" << std::setw(16) << "Variance\n";
  std::cout << std::string(26, '-') << "\n";
  for (std::size_t i = 0; i < times.size(); ++i) {
    std::cout << std::setw(10) << std::setprecision(4) << times[i]
              << std::setw(16) << std::scientific << variances[i] << "\n";
  }

  // Mass conservation check.

  double mass_initial = arma::accu(rho) * model.grid.cell_volume();
  double mass_final = arma::accu(densities[0]) * model.grid.cell_volume();
  std::cout << "\n  Mass initial: " << std::fixed << mass_initial << "\n";
  std::cout << "  Mass final:   " << mass_final << "\n";
  std::cout << "  Rel. error:   " << std::abs(mass_final - mass_initial) / mass_initial << "\n";

#ifdef DFT_HAS_MATPLOTLIB
  plot::ddft_variance(times, variances);
#endif
}

int main() {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  demo_fire2();
  demo_ddft();
}
