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

  // Lennard-Jones model (same as the density example).

  physics::Model model{
      .grid = make_grid(0.4, {12.0, 12.0, 12.0}),
      .species = {Species{.name = "LJ", .hard_sphere_diameter = 1.0}},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = physics::potentials::make_lennard_jones(1.0, 1.0, 2.5),
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      }},
      .temperature = 0.7,
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

  std::cout << "  Coexistence:\n";
  std::cout << "    rho_vapor  = " << coex->rho_vapor << "\n";
  std::cout << "    rho_liquid = " << coex->rho_liquid << "\n";
  std::cout << "    mu_coex    = " << mu_coex << "\n\n";

  // Supersaturation: increase chemical potential above coexistence.
  // At mu > mu_coex, the liquid phase is thermodynamically stable,
  // while the vapor is metastable. A sufficiently large liquid
  // droplet (above the critical size) will grow.

  double delta_mu = 0.1;
  double mu_super = mu_coex + delta_mu;

  std::cout << "  Supersaturation:\n";
  std::cout << "    delta_mu   = " << delta_mu << "\n";
  std::cout << "    mu_super   = " << mu_super << "\n\n";

  // Build DFT weights.

  auto weights = functionals::make_weights(fmt_model, model);

  // Construct spherical droplet initial profile.
  // A liquid droplet of radius R_drop in a uniform metastable vapor.

  long nx = model.grid.shape[0];
  long ny = model.grid.shape[1];
  long nz = model.grid.shape[2];
  double dx = model.grid.dx;
  double cx = model.grid.box_size[0] / 2.0;
  double cy = model.grid.box_size[1] / 2.0;
  double cz = model.grid.box_size[2] / 2.0;

  double R_drop = 3.0;       // droplet radius (sigma units)
  double interface_w = 1.0;   // interface width

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

  // State factory with the supersaturated chemical potential.

  auto make_state = [&](const arma::vec& rho) -> State {
    auto s = init::from_profile(model, rho);
    s.species[0].chemical_potential = mu_super;
    return s;
  };

  // Evaluate the initial functional.

  auto initial_state = make_state(rho_init);
  auto initial_result = functionals::total(model, initial_state, weights);

  std::cout << "=== Initial droplet (R = " << R_drop << " sigma) ===\n\n";
  std::cout << "  Free energy:      " << initial_result.free_energy << "\n";
  std::cout << "  Grand potential:  " << initial_result.grand_potential << "\n";
  std::cout << "  Max |force|:      " << arma::max(arma::abs(initial_result.forces[0])) << "\n";
  std::cout << "  Total mass:       " << arma::accu(rho_init) * model.grid.cell_volume() << "\n\n";

  // DDFT relaxation.

  std::cout << "=== DDFT relaxation ===\n\n";

  auto force_fn = [&](const std::vector<arma::vec>& densities)
      -> std::pair<double, std::vector<arma::vec>> {
    auto state = make_state(densities[0]);
    auto result = functionals::total(model, state, weights);
    return {result.grand_potential, result.forces};
  };

  algorithms::ddft::DdftConfig dconf{
      .dt = 5e-4,
      .diffusion_coefficient = 1.0,
      .min_density = 1e-18,
  };

  auto k2 = algorithms::ddft::compute_k_squared(model.grid);
  auto prop = algorithms::ddft::diffusion_propagator(k2, dconf.diffusion_coefficient, dconf.dt);

  std::vector<arma::vec> densities = {rho_init};
  int n_steps = 300;
  int snapshot_interval = 60;
  int log_interval = 30;

  std::vector<double> times, omegas;
  std::vector<std::vector<double>> profile_snapshots;
  std::vector<double> snapshot_times;

  times.push_back(0.0);
  omegas.push_back(initial_result.grand_potential);
  profile_snapshots.push_back(profile_init);
  snapshot_times.push_back(0.0);

  std::cout << std::setw(8) << "Step"
            << std::setw(14) << "Time"
            << std::setw(18) << "Grand pot."
            << std::setw(16) << "Max |force|" << "\n";
  std::cout << std::string(56, '-') << "\n";
  std::cout << std::setw(8) << 0
            << std::setw(14) << 0.0
            << std::setw(18) << initial_result.grand_potential
            << std::setw(16) << arma::max(arma::abs(initial_result.forces[0])) << "\n";

  for (int step = 1; step <= n_steps; ++step) {
    auto result = algorithms::ddft::split_operator_step(
        densities, model.grid, k2, prop, force_fn, dconf
    );
    densities = std::move(result.densities);

    if (step % log_interval == 0 || step == n_steps) {
      auto state = make_state(densities[0]);
      auto eval = functionals::total(model, state, weights);
      double t = step * dconf.dt;
      times.push_back(t);
      omegas.push_back(eval.grand_potential);

      std::cout << std::setw(8) << step
                << std::setw(14) << t
                << std::setw(18) << eval.grand_potential
                << std::setw(16) << arma::max(arma::abs(eval.forces[0])) << "\n";
    }

    if (step % snapshot_interval == 0) {
      auto [r, prof] = extract_radial(densities[0]);
      profile_snapshots.push_back(prof);
      snapshot_times.push_back(step * dconf.dt);
    }
  }

  // Final state.

  auto final_state = make_state(densities[0]);
  auto final_result = functionals::total(model, final_state, weights);
  auto [r_final, profile_final] = extract_radial(densities[0]);

  double mass_initial = arma::accu(rho_init) * model.grid.cell_volume();
  double mass_final = arma::accu(densities[0]) * model.grid.cell_volume();

  std::cout << "\n=== Final state ===\n\n";
  std::cout << "  Grand potential:  " << final_result.grand_potential << "\n";
  std::cout << "  Max |force|:      " << arma::max(arma::abs(final_result.forces[0])) << "\n";
  std::cout << "  Mass initial:     " << mass_initial << "\n";
  std::cout << "  Mass final:       " << mass_final << "\n";
  std::cout << "  Rel. error:       " << std::abs(mass_final - mass_initial) / mass_initial << "\n";

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  plot::droplet_evolution(r_init, profile_snapshots, snapshot_times,
                          profile_init, profile_final,
                          coex->rho_vapor, coex->rho_liquid);
  plot::grand_potential(times, omegas);
#endif
}
