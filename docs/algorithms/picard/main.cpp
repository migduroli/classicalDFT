#include "dft.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>

using namespace dft;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  std::cout << std::fixed << std::setprecision(6);

  // Define a hard-sphere fluid at moderate packing fraction.

  console::info("Picard iteration: hard-sphere density equilibration");

  double dx = 0.1;
  double box_x = 10.0;
  double box_yz = 2.0;
  double temperature = 1.0;
  double sigma = 1.0;
  double rho_bulk = 0.6;

  physics::Model model{
      .grid = make_grid(dx, {box_x, box_yz, box_yz}),
      .species = {Species{.name = "HS", .hard_sphere_diameter = sigma}},
      .interactions = {},
      .temperature = temperature,
  };

  auto fmt_model = functionals::fmt::WhiteBearII{};
  auto weights = functionals::make_weights(fmt_model, model);

  long nx = model.grid.shape[0];
  long ny = model.grid.shape[1];
  long nz = model.grid.shape[2];

  std::cout << "  Grid:        " << nx << "x" << ny << "x" << nz
            << " (dx = " << model.grid.dx << ")\n";
  std::cout << "  rho_bulk:    " << rho_bulk << "\n";
  std::cout << "  eta_bulk:    " << physics::hard_spheres::packing_fraction(rho_bulk) << "\n\n";

  // Bulk chemical potential at the target density.

  auto bulk_weights = functionals::make_bulk_weights(
      fmt_model, model.interactions, model.temperature
  );
  double mu_bulk = functionals::bulk::chemical_potential(
      arma::vec{rho_bulk}, model.species, bulk_weights, 0
  );
  double p_bulk = functionals::bulk::pressure(
      arma::vec{rho_bulk}, model.species, bulk_weights
  );

  std::cout << "  mu_bulk:     " << mu_bulk << "\n";
  std::cout << "  P_bulk:      " << p_bulk << "\n\n";

  // Initial density: uniform with a small sinusoidal perturbation.

  arma::vec x_vals = arma::linspace(0.0, (nx - 1) * model.grid.dx, nx);
  arma::vec profile_1d = rho_bulk * (1.0 + 0.05 * arma::sin(
      2.0 * arma::datum::pi * x_vals / model.grid.box_size[0]
  ));

  arma::vec rho_init = arma::repelem(profile_1d, ny * nz, 1);

  // Force function.

  auto make_state = [&](const arma::vec& rho) -> State {
    auto s = init::from_profile(model, rho);
    s.species[0].chemical_potential = mu_bulk;
    return s;
  };

  auto force_fn = [&](const std::vector<arma::vec>& densities)
      -> std::pair<double, std::vector<arma::vec>> {
    auto state = make_state(densities[0]);
    auto result = functionals::total(model, state, weights);
    return {result.grand_potential, result.forces};
  };

  // Evaluate the initial state.

  auto initial_state = make_state(rho_init);
  auto initial_result = functionals::total(model, initial_state, weights);

  std::cout << "=== Initial state (perturbed uniform) ===\n\n";
  std::cout << "  Grand potential:  " << initial_result.grand_potential << "\n";
  std::cout << "  Max |force|:      " << arma::max(arma::abs(initial_result.forces[0])) << "\n\n";

  // Picard iteration.

  console::info("Running Picard iteration");

  algorithms::picard::PicardConfig picard_config{
      .mixing = 0.005,
      .min_density = 1e-30,
      .tolerance = 1e-8,
      .max_iterations = 5000,
      .log_interval = 500,
  };

  auto picard_result = algorithms::picard::solve(
      {rho_init}, force_fn, model.grid.cell_volume(), picard_config
  );

  std::cout << "\n=== Picard result ===\n\n";
  std::cout << "  Converged:        " << (picard_result.converged ? "true" : "false") << "\n";
  std::cout << "  Iterations:       " << picard_result.iterations << "\n";
  std::cout << "  Grand potential:  " << picard_result.grand_potential << "\n";
  std::cout << "  Residual:         " << std::scientific << picard_result.residual << "\n\n";

  // Verify that the converged density is uniform.

  std::cout << std::fixed;
  double rho_min = picard_result.densities[0].min();
  double rho_max = picard_result.densities[0].max();
  double rho_mean = arma::mean(picard_result.densities[0]);
  std::cout << "  rho_min:          " << rho_min << "\n";
  std::cout << "  rho_max:          " << rho_max << "\n";
  std::cout << "  rho_mean:         " << rho_mean << "\n";
  std::cout << "  variation:        " << std::scientific << (rho_max - rho_min) << "\n\n";

  // Verify that Omega/V = -P_bulk.

  std::cout << std::fixed;
  double volume = model.grid.cell_volume()
                  * static_cast<double>(model.grid.total_points());
  double omega_per_vol = picard_result.grand_potential / volume;
  std::cout << "  Omega/V:          " << omega_per_vol << "\n";
  std::cout << "  -P_bulk:          " << -p_bulk << "\n";
  std::cout << "  relative error:   " << std::scientific
            << std::abs(omega_per_vol + p_bulk) / p_bulk << "\n";
}
