#include "dft.hpp"
#include "plot.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <vector>

using namespace dft;

int main() {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  std::cout << std::fixed << std::setprecision(6);

  // Read configuration.

  auto cfg = config::parse_config("config.ini", config::FileType::INI);

  double dx = config::get<double>(cfg, "model.dx");
  double box_x = config::get<double>(cfg, "model.box_x");
  double box_y = config::get<double>(cfg, "model.box_y");
  double box_z = config::get<double>(cfg, "model.box_z");
  double temperature = config::get<double>(cfg, "model.temperature");
  double sigma = config::get<double>(cfg, "model.sigma");
  double epsilon_lj = config::get<double>(cfg, "model.epsilon");
  double cutoff = config::get<double>(cfg, "model.cutoff");
  double R_drop = config::get<double>(cfg, "droplet.radius");
  double interface_w = config::get<double>(cfg, "droplet.interface_width");
  double delta_mu = config::get<double>(cfg, "droplet.delta_mu");
  double dt = config::get<double>(cfg, "ddft.dt");
  double D = config::get<double>(cfg, "ddft.diffusion_coefficient");
  int n_steps = config::get<int>(cfg, "ddft.n_steps");
  int snapshot_interval = config::get<int>(cfg, "ddft.snapshot_interval");
  int log_interval = config::get<int>(cfg, "ddft.log_interval");

  // Lennard-Jones model.

  physics::Model model{
      .grid = make_grid(dx, {box_x, box_y, box_z}),
      .species = {Species{.name = "LJ", .hard_sphere_diameter = sigma}},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = physics::potentials::make_lennard_jones(sigma, epsilon_lj, cutoff),
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      }},
      .temperature = temperature,
  };

  auto fmt_model = functionals::fmt::WhiteBearII{};

  std::cout << "=== Nucleation: critical droplet in supersaturated vapor ===\n\n";
  std::cout << "  Grid:        " << model.grid.shape[0] << "^3"
            << " (dx = " << model.grid.dx << ")\n";
  std::cout << "  Temperature: T* = " << model.temperature << "\n\n";

  // Bulk thermodynamics and coexistence.

  auto bulk_weights = functionals::make_bulk_weights(
      fmt_model, model.interactions, model.temperature
  );

  functionals::bulk::PhaseSearchConfig search_config{
      .rho_max = 1.0,
      .rho_scan_step = 0.005,
      .newton = {.max_iterations = 300, .tolerance = 1e-10},
  };

  auto coex = functionals::bulk::find_coexistence(model.species, bulk_weights, search_config);
  if (!coex) {
    std::cerr << "ERROR: coexistence not found\n";
    return 1;
  }

  double mu_coex = functionals::bulk::chemical_potential(
      arma::vec{coex->rho_vapor}, model.species, bulk_weights, 0
  );
  double mu_super = mu_coex + delta_mu;

  std::cout << "  Coexistence:\n";
  std::cout << "    rho_vapor  = " << coex->rho_vapor << "\n";
  std::cout << "    rho_liquid = " << coex->rho_liquid << "\n";
  std::cout << "    mu_coex    = " << mu_coex << "\n\n";
  std::cout << "  Supersaturation:\n";
  std::cout << "    delta_mu   = " << delta_mu << "\n";
  std::cout << "    mu_super   = " << mu_super << "\n\n";

  // Build DFT weights.

  auto weights = functionals::make_weights(fmt_model, model);

  // Construct spherical droplet initial profile.

  long nx = model.grid.shape[0];
  long ny = model.grid.shape[1];
  long nz = model.grid.shape[2];
  double cx = model.grid.box_size[0] / 2.0;
  double cy = model.grid.box_size[1] / 2.0;
  double cz = model.grid.box_size[2] / 2.0;

  auto n_points = static_cast<arma::uword>(model.grid.total_points());
  arma::vec rho_init(n_points);

  for (long ix = 0; ix < nx; ++ix) {
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        double x = ix * dx - cx;
        double y = iy * dx - cy;
        double z = iz * dx - cz;
        double r = std::sqrt(x * x + y * y + z * z);
        double profile = 0.5 * (1.0 - std::tanh((r - R_drop) / interface_w));
        double rho = coex->rho_vapor + (coex->rho_liquid - coex->rho_vapor) * profile;
        long idx = model.grid.flat_index(ix, iy, iz);
        rho_init(static_cast<arma::uword>(idx)) = rho;
      }
    }
  }

  // Extract radial density profile (x-axis slice through center).

  long iy_mid = ny / 2;
  long iz_mid = nz / 2;
  auto extract_radial = [&](const arma::vec& rho_3d) -> std::pair<std::vector<double>, std::vector<double>> {
    std::vector<double> r_vals, rho_vals;
    for (long ix = 0; ix < nx; ++ix) {
      double x = ix * dx - cx;
      r_vals.push_back(x);
      long idx = model.grid.flat_index(ix, iy_mid, iz_mid);
      rho_vals.push_back(rho_3d(static_cast<arma::uword>(idx)));
    }
    return {r_vals, rho_vals};
  };

  auto [r_init, profile_init] = extract_radial(rho_init);

  // Force function with supersaturated chemical potential.

  auto make_state = [&](const arma::vec& rho) -> State {
    auto s = init::from_profile(model, rho);
    s.species[0].chemical_potential = mu_super;
    return s;
  };

  auto force_fn = [&](const std::vector<arma::vec>& densities)
      -> std::pair<double, std::vector<arma::vec>> {
    auto state = make_state(densities[0]);
    auto result = functionals::total(model, state, weights);
    return {result.grand_potential, result.forces};
  };

  // Initial evaluation.

  auto initial_state = make_state(rho_init);
  auto initial_result = functionals::total(model, initial_state, weights);

  std::cout << "=== Initial droplet (R = " << R_drop << " sigma) ===\n\n";
  std::cout << "  Free energy:      " << initial_result.free_energy << "\n";
  std::cout << "  Grand potential:  " << initial_result.grand_potential << "\n";
  std::cout << "  Max |force|:      " << arma::max(arma::abs(initial_result.forces[0])) << "\n";
  std::cout << "  Total mass:       " << arma::accu(rho_init) * model.grid.cell_volume() << "\n\n";

  // DDFT relaxation via the simulate API.

  std::cout << "=== DDFT relaxation ===\n\n";

  algorithms::ddft::SimulationConfig sim_config{
      .ddft = {.dt = dt, .diffusion_coefficient = D, .min_density = 1e-18},
      .n_steps = n_steps,
      .snapshot_interval = snapshot_interval,
      .log_interval = log_interval,
  };

  auto sim = algorithms::ddft::simulate({rho_init}, model.grid, force_fn, sim_config);

  // Collect profile snapshots.

  std::vector<std::vector<double>> profile_snapshots;
  std::vector<double> snapshot_times;
  for (const auto& snap : sim.snapshots) {
    auto [r, prof] = extract_radial(snap.densities[0]);
    profile_snapshots.push_back(prof);
    snapshot_times.push_back(snap.time);
  }
  auto [r_final, profile_final] = extract_radial(sim.densities[0]);

  std::cout << "\n=== Final state ===\n\n";
  std::cout << "  Grand potential:  " << sim.energies.back() << "\n";
  std::cout << "  Mass initial:     " << sim.mass_initial << "\n";
  std::cout << "  Mass final:       " << sim.mass_final << "\n";
  std::cout << "  Rel. error:       " << std::abs(sim.mass_final - sim.mass_initial) / sim.mass_initial << "\n";

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  plot::droplet_evolution(r_init, profile_snapshots, snapshot_times,
                          profile_init, profile_final,
                          coex->rho_vapor, coex->rho_liquid);
  plot::grand_potential(sim.times, sim.energies);
#endif
}
