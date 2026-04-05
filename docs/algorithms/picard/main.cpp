#include "plot.hpp"
#include "utils.hpp"

#include <dftlib>
#include <filesystem>
#include <iostream>
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

  auto func = functionals::make_functional(functionals::fmt::WhiteBearII{}, model);

  long nx = func.model.grid.shape[0];
  long ny = func.model.grid.shape[1];
  long nz = func.model.grid.shape[2];

  std::println(std::cout, "  Grid:        {}x{}x{} (dx = {})", nx, ny, nz, func.model.grid.dx);
  std::println(std::cout, "  rho_bulk:    {:.6f}", rho_bulk);
  std::println(std::cout, "  eta_bulk:    {:.6f}\n", physics::hard_spheres::packing_fraction(rho_bulk));

  // Bulk chemical potential at the target density.

  auto eos = func.bulk();
  double mu_bulk = eos.chemical_potential(arma::vec{rho_bulk}, 0);
  double p_bulk = eos.pressure(arma::vec{rho_bulk});

  std::println(std::cout, "  mu_bulk:     {:.6f}", mu_bulk);
  std::println(std::cout, "  P_bulk:      {:.6f}\n", p_bulk);

  // Initial density: uniform with a small sinusoidal perturbation.

  arma::vec x_vals = arma::linspace(0.0, (nx - 1) * func.model.grid.dx, nx);
  arma::vec profile_1d = rho_bulk
      * (1.0 + 0.05 * arma::sin(2.0 * arma::datum::pi * x_vals / func.model.grid.box_size[0]));

  arma::vec rho_init = arma::repelem(profile_1d, ny * nz, 1);

  // Force function.

  auto force_fn = func.grand_potential_callback(mu_bulk);

  // Evaluate the initial state.

  auto initial_result = func.evaluate(rho_init, mu_bulk);

  std::println(std::cout, "=== Initial state (perturbed uniform) ===\n");
  std::println(std::cout, "  Grand potential:  {:.6f}", initial_result.grand_potential);
  std::println(std::cout, "  Max |force|:      {:.6f}\n", arma::max(arma::abs(initial_result.forces[0])));

  // Picard iteration.

  console::info("Running Picard iteration");

  algorithms::picard::Picard picard_config{
      .mixing = 0.005,
      .min_density = 1e-30,
      .tolerance = 1e-8,
      .max_iterations = 5000,
      .log_interval = 500,
  };

  auto picard_result = picard_config.solve({rho_init}, force_fn, func.model.grid.cell_volume());

  std::println(std::cout, "\n=== Picard result ===\n");
  std::println(std::cout, "  Converged:        {}", picard_result.converged ? "true" : "false");
  std::println(std::cout, "  Iterations:       {}", picard_result.iterations);
  std::println(std::cout, "  Grand potential:  {:.6f}", picard_result.grand_potential);
  std::println(std::cout, "  Residual:         {:.6e}\n", picard_result.residual);

  // Verify that the converged density is uniform.

  double rho_min = picard_result.densities[0].min();
  double rho_max = picard_result.densities[0].max();
  double rho_mean = arma::mean(picard_result.densities[0]);
  std::println(std::cout, "  rho_min:          {:.6f}", rho_min);
  std::println(std::cout, "  rho_max:          {:.6f}", rho_max);
  std::println(std::cout, "  rho_mean:         {:.6f}", rho_mean);
  std::println(std::cout, "  variation:        {:.6e}\n", rho_max - rho_min);

  // Verify that Omega/V = -P_bulk.

  double volume = func.model.grid.cell_volume() * static_cast<double>(func.model.grid.total_points());
  double omega_per_vol = picard_result.grand_potential / volume;
  std::println(std::cout, "  Omega/V:          {:.6f}", omega_per_vol);
  std::println(std::cout, "  -P_bulk:          {:.6f}", -p_bulk);
  std::println(std::cout, "  relative error:   {:.6e}", std::abs(omega_per_vol + p_bulk) / p_bulk);

  // Collect plot data.

  auto extract_x_profile = [&](const arma::vec& rho_3d) -> std::vector<double> {
    arma::mat rho_mat = arma::reshape(rho_3d, ny * nz, nx);
    arma::vec profile_avg = arma::mean(rho_mat, 0).as_col();
    return arma::conv_to<std::vector<double>>::from(profile_avg);
  };
  auto x_coords = arma::conv_to<std::vector<double>>::from(x_vals);
  auto rho_init_profile = extract_x_profile(rho_init);
  auto rho_final_profile = extract_x_profile(picard_result.densities[0]);

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(x_coords, rho_init_profile, rho_final_profile, rho_bulk);
#endif
}
