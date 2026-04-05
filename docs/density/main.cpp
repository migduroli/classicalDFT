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
    .grid = make_grid(dx, { box_x, box_y, box_z }),
    .species = { Species{ .name = "LJ", .hard_sphere_diameter = sigma } },
    .interactions = { {
        .species_i = 0,
        .species_j = 0,
        .potential = physics::potentials::make_lennard_jones(sigma, epsilon, cutoff),
        .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
    } },
    .temperature = temperature,
  };

  auto func = functionals::make_functional(functionals::fmt::WhiteBearII{}, model);

  std::println(std::cout, "=== DFT + DDFT: LJ liquid slab relaxation ===\n");
  std::println(
      std::cout,
      "  Grid:        {}x{}x{} (dx = {})",
      func.model.grid.shape[0],
      func.model.grid.shape[1],
      func.model.grid.shape[2],
      func.model.grid.dx
  );
  std::println(std::cout, "  Species:     {}", func.model.species[0].name);
  std::println(std::cout, "  Temperature: T* = {}\n", func.model.temperature);

  // Bulk thermodynamics.

  auto eos = func.bulk();

  arma::vec rho_grid = arma::linspace(0.01, 1.0, 200);
  arma::vec f_grid(rho_grid.n_elem), mu_grid(rho_grid.n_elem), p_grid(rho_grid.n_elem);
  for (arma::uword i = 0; i < rho_grid.n_elem; ++i) {
    f_grid(i) = eos.free_energy_density(arma::vec{ rho_grid(i) });
    mu_grid(i) = eos.chemical_potential(arma::vec{ rho_grid(i) }, 0);
    p_grid(i) = eos.pressure(arma::vec{ rho_grid(i) });
  }

  // Find coexistence.

  functionals::bulk::PhaseSearch search_config{
    .rho_max = 1.0,
    .rho_scan_step = 0.005,
    .newton = { .max_iterations = 300, .tolerance = 1e-10 },
  };

  auto coex = search_config.find_coexistence(eos);
  if (!coex) {
    std::cerr << "ERROR: coexistence not found\n";
    return 1;
  }

  double mu_coex = eos.chemical_potential(arma::vec{ coex->rho_vapor }, 0);
  double p_coex = eos.pressure(arma::vec{ coex->rho_vapor });

  std::println(std::cout, "  Coexistence:");
  std::println(std::cout, "    rho_vapor  = {:.6f}", coex->rho_vapor);
  std::println(std::cout, "    rho_liquid = {:.6f}", coex->rho_liquid);
  std::println(std::cout, "    mu_coex    = {:.6f}", mu_coex);
  std::println(std::cout, "    P_coex     = {:.6f}\n", p_coex);

  // Build DFT weights (FFT-based, full functional).

  // Construct liquid slab initial profile.

  long nx = func.model.grid.shape[0];
  long ny = func.model.grid.shape[1];
  long nz = func.model.grid.shape[2];
  double center = func.model.grid.box_size[0] / 2.0;
  double width = func.model.grid.box_size[0] / 4.0;

  arma::vec x_vals = arma::linspace(0.0, (nx - 1) * func.model.grid.dx, nx);
  arma::vec profile_1d = coex->rho_vapor
      + 0.5 * (coex->rho_liquid - coex->rho_vapor)
          * (arma::tanh((x_vals - center + width) / interface_width)
             - arma::tanh((x_vals - center - width) / interface_width));

  arma::vec slab_rho = arma::repelem(profile_1d, ny * nz, 1);

  auto x_coords = arma::conv_to<std::vector<double>>::from(x_vals);
  long nxl = nx, nyl = ny, nzl = nz;

  // Force function wrapping the full DFT functional.

  auto force_fn = func.grand_potential_callback(mu_coex);

  // Evaluate the initial functional.

  auto initial_result = func.evaluate(slab_rho, mu_coex);

  std::println(std::cout, "=== Initial slab (tanh profile) ===\n");
  std::println(std::cout, "  Free energy:      {:.6f}", initial_result.free_energy);
  std::println(std::cout, "  Grand potential:  {:.6f}", initial_result.grand_potential);
  std::println(std::cout, "  Max |force|:      {:.6f}\n", arma::max(arma::abs(initial_result.forces[0])));

  // DDFT relaxation via the simulate API.

  std::println(std::cout, "=== DDFT relaxation (split-operator) ===\n");

  algorithms::dynamics::Simulation sim_config{
    .step = { .dt = dt, .diffusion_coefficient = D, .min_density = 1e-18 },
    .n_steps = n_steps,
    .snapshot_interval = snapshot_interval,
    .log_interval = log_interval,
  };

  auto sim = sim_config.run({ slab_rho }, func.model.grid, force_fn);

  // Collect profile snapshots from the simulation.

  auto initial_profile = utils::extract_profile(slab_rho, nxl, nyl, nzl);
  std::vector<std::vector<double>> profile_snapshots;
  std::vector<double> snapshot_times;
  for (const auto& snap : sim.snapshots) {
    profile_snapshots.push_back(utils::extract_profile(snap.densities[0], nxl, nyl, nzl));
    snapshot_times.push_back(snap.time);
  }
  auto final_profile = utils::extract_profile(sim.densities[0], nxl, nyl, nzl);

  std::println(std::cout, "\n=== Final relaxed state ===\n");
  std::println(std::cout, "  Grand potential:  {:.6f}", sim.energies.back());
  std::println(std::cout, "  Mass initial:     {:.6f}", sim.mass_initial);
  std::println(std::cout, "  Mass final:       {:.6f}", sim.mass_final);
  std::println(std::cout, "  Rel. error:       {:.6e}", std::abs(sim.mass_final - sim.mass_initial) / sim.mass_initial);

  // Plots.

  auto rho_range = arma::conv_to<std::vector<double>>::from(rho_grid);
  auto p_range = arma::conv_to<std::vector<double>>::from(p_grid);
  auto f_range = arma::conv_to<std::vector<double>>::from(f_grid);
  auto mu_range = arma::conv_to<std::vector<double>>::from(mu_grid);
  double f_v = eos.free_energy_density(arma::vec{ coex->rho_vapor });
  double f_l = eos.free_energy_density(arma::vec{ coex->rho_liquid });

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(
      rho_range,
      p_range,
      f_range,
      mu_range,
      coex->rho_vapor,
      coex->rho_liquid,
      p_coex,
      f_v,
      f_l,
      mu_coex,
      model.temperature,
      x_coords,
      profile_snapshots,
      snapshot_times,
      initial_profile,
      final_profile,
      sim.times,
      sim.energies
  );
#endif
}
