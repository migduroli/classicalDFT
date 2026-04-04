#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numbers>
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

  std::cout << std::fixed << std::setprecision(6);

  // Read configuration.

  auto cfg = config::parse_config("config.ini", config::FileType::INI);

  double dx = config::get<double>(cfg, "model.dx");
  double box_x = config::get<double>(cfg, "model.box_x");
  double box_y = config::get<double>(cfg, "model.box_y");
  double box_z = config::get<double>(cfg, "model.box_z");
  double temperature = config::get<double>(cfg, "model.temperature");
  double sigma = config::get<double>(cfg, "model.sigma");
  double epsilon = config::get<double>(cfg, "model.epsilon");
  double cutoff = config::get<double>(cfg, "model.cutoff");
  double interface_width = config::get<double>(cfg, "slab.interface_width");
  double dt = config::get<double>(cfg, "ddft.dt");
  double D = config::get<double>(cfg, "ddft.diffusion_coefficient");
  int n_steps = config::get<int>(cfg, "ddft.n_steps");
  int snapshot_interval = config::get<int>(cfg, "ddft.snapshot_interval");
  int log_interval = config::get<int>(cfg, "ddft.log_interval");

  // Define the Lennard-Jones system.

  physics::Model model{
      .grid = make_grid(dx, {box_x, box_y, box_z}),
      .species = {Species{.name = "LJ", .hard_sphere_diameter = sigma}},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = physics::potentials::make_lennard_jones(sigma, epsilon, cutoff),
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      }},
      .temperature = temperature,
  };

  auto fmt_model = functionals::fmt::WhiteBearII{};

  std::cout << "=== DFT + DDFT: LJ liquid slab relaxation ===\n\n";
  std::cout << "  Grid:        " << model.grid.shape[0] << "x"
            << model.grid.shape[1] << "x" << model.grid.shape[2]
            << " (dx = " << model.grid.dx << ")\n";
  std::cout << "  Species:     " << model.species[0].name << "\n";
  std::cout << "  Temperature: T* = " << model.temperature << "\n\n";

  // Bulk thermodynamics.

  auto bulk_weights = functionals::make_bulk_weights(
      fmt_model, model.interactions, model.temperature
  );

  arma::vec rho_grid = arma::linspace(0.01, 1.0, 200);
  arma::vec f_grid(rho_grid.n_elem), mu_grid(rho_grid.n_elem), p_grid(rho_grid.n_elem);
  for (arma::uword i = 0; i < rho_grid.n_elem; ++i) {
    f_grid(i) = functionals::bulk::free_energy_density(arma::vec{rho_grid(i)}, model.species, bulk_weights);
    mu_grid(i) = functionals::bulk::chemical_potential(arma::vec{rho_grid(i)}, model.species, bulk_weights, 0);
    p_grid(i) = functionals::bulk::pressure(arma::vec{rho_grid(i)}, model.species, bulk_weights);
  }

  // Find coexistence.

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
  double p_coex = functionals::bulk::pressure(
      arma::vec{coex->rho_vapor}, model.species, bulk_weights
  );

  std::cout << "  Coexistence:\n";
  std::cout << "    rho_vapor  = " << coex->rho_vapor << "\n";
  std::cout << "    rho_liquid = " << coex->rho_liquid << "\n";
  std::cout << "    mu_coex    = " << mu_coex << "\n";
  std::cout << "    P_coex     = " << p_coex << "\n\n";

  // Build DFT weights (FFT-based, full functional).

  auto weights = functionals::make_weights(fmt_model, model);

  // Construct liquid slab initial profile.

  long nx = model.grid.shape[0];
  long ny = model.grid.shape[1];
  long nz = model.grid.shape[2];
  double center = model.grid.box_size[0] / 2.0;
  double width = model.grid.box_size[0] / 4.0;

  arma::vec x_vals = arma::linspace(0.0, (nx - 1) * model.grid.dx, nx);
  arma::vec profile_1d = coex->rho_vapor + 0.5 * (coex->rho_liquid - coex->rho_vapor) *
      (arma::tanh((x_vals - center + width) / interface_width) -
       arma::tanh((x_vals - center - width) / interface_width));

  arma::vec slab_rho = arma::repelem(profile_1d, ny * nz, 1);

  // Extract 1D profile helper.

  auto extract_profile = [&](const arma::vec& rho_3d) -> std::vector<double> {
    arma::mat rho_mat = arma::reshape(rho_3d, ny * nz, nx);
    arma::vec profile_avg = arma::mean(rho_mat, 0).as_col();
    return arma::conv_to<std::vector<double>>::from(profile_avg);
  };

  auto x_coords = arma::conv_to<std::vector<double>>::from(x_vals);

  // Force function wrapping the full DFT functional.

  auto make_state = [&](const arma::vec& rho) -> State {
    auto s = init::from_profile(model, rho);
    s.species[0].chemical_potential = mu_coex;
    return s;
  };

  auto force_fn = [&](const std::vector<arma::vec>& densities)
      -> std::pair<double, std::vector<arma::vec>> {
    auto state = make_state(densities[0]);
    auto result = functionals::total(model, state, weights);
    return {result.grand_potential, result.forces};
  };

  // Evaluate the initial functional.

  auto initial_state = make_state(slab_rho);
  auto initial_result = functionals::total(model, initial_state, weights);

  std::cout << "=== Initial slab (tanh profile) ===\n\n";
  std::cout << "  Free energy:      " << initial_result.free_energy << "\n";
  std::cout << "  Grand potential:  " << initial_result.grand_potential << "\n";
  std::cout << "  Max |force|:      " << arma::max(arma::abs(initial_result.forces[0])) << "\n\n";

  // DDFT relaxation via the simulate API.

  std::cout << "=== DDFT relaxation (split-operator) ===\n\n";

  algorithms::ddft::SimulationConfig sim_config{
      .ddft = {.dt = dt, .diffusion_coefficient = D, .min_density = 1e-18},
      .n_steps = n_steps,
      .snapshot_interval = snapshot_interval,
      .log_interval = log_interval,
  };

  auto sim = algorithms::ddft::simulate({slab_rho}, model.grid, force_fn, sim_config);

  // Collect profile snapshots from the simulation.

  auto initial_profile = extract_profile(slab_rho);
  std::vector<std::vector<double>> profile_snapshots;
  std::vector<double> snapshot_times;
  for (const auto& snap : sim.snapshots) {
    profile_snapshots.push_back(extract_profile(snap.densities[0]));
    snapshot_times.push_back(snap.time);
  }
  auto final_profile = extract_profile(sim.densities[0]);

  std::cout << "\n=== Final relaxed state ===\n\n";
  std::cout << "  Grand potential:  " << sim.energies.back() << "\n";
  std::cout << "  Mass initial:     " << sim.mass_initial << "\n";
  std::cout << "  Mass final:       " << sim.mass_final << "\n";
  std::cout << "  Rel. error:       " << std::abs(sim.mass_final - sim.mass_initial) / sim.mass_initial << "\n";

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  auto rho_range = arma::conv_to<std::vector<double>>::from(rho_grid);
  auto p_range = arma::conv_to<std::vector<double>>::from(p_grid);
  auto f_range = arma::conv_to<std::vector<double>>::from(f_grid);
  auto mu_range = arma::conv_to<std::vector<double>>::from(mu_grid);
  double f_v = functionals::bulk::free_energy_density(arma::vec{coex->rho_vapor}, model.species, bulk_weights);
  double f_l = functionals::bulk::free_energy_density(arma::vec{coex->rho_liquid}, model.species, bulk_weights);

  plot::pressure_isotherm(rho_range, p_range, coex->rho_vapor, coex->rho_liquid, p_coex, model.temperature);
  plot::free_energy(rho_range, f_range, coex->rho_vapor, coex->rho_liquid, f_v, f_l, model.temperature);
  plot::chemical_potential(rho_range, mu_range, coex->rho_vapor, coex->rho_liquid, mu_coex, model.temperature);
  plot::density_evolution(x_coords, profile_snapshots, snapshot_times,
                          initial_profile, final_profile,
                          coex->rho_vapor, coex->rho_liquid);
  plot::grand_potential(sim.times, sim.energies);
#endif
}
