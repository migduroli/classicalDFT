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

  // Define the Lennard-Jones system.

  physics::Model model{
      .grid = make_grid(0.1, {10.0, 10.0, 10.0}),
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

  std::cout << "=== DFT density calculation: LJ fluid ===\n\n";
  std::cout << "  Grid:        " << model.grid.shape[0] << "^3"
            << " (dx = " << model.grid.dx << ")\n";
  std::cout << "  Species:     " << model.species[0].name << "\n";
  std::cout << "  Temperature: T* = " << model.temperature << "\n\n";

  // Bulk thermodynamics at this temperature.

  auto bulk_weights = functionals::make_bulk_weights(
      fmt_model, model.interactions, model.temperature
  );

  std::cout << "=== Bulk properties at T* = " << model.temperature << " ===\n\n";

  arma::vec rho_grid = arma::linspace(0.01, 1.0, 200);
  arma::vec f_grid(rho_grid.n_elem), mu_grid(rho_grid.n_elem), p_grid(rho_grid.n_elem);
  for (arma::uword i = 0; i < rho_grid.n_elem; ++i) {
    f_grid(i) = functionals::bulk::free_energy_density(arma::vec{rho_grid(i)}, model.species, bulk_weights);
    mu_grid(i) = functionals::bulk::chemical_potential(arma::vec{rho_grid(i)}, model.species, bulk_weights, 0);
    p_grid(i) = functionals::bulk::pressure(arma::vec{rho_grid(i)}, model.species, bulk_weights);
  }
  auto rho_range = arma::conv_to<std::vector<double>>::from(rho_grid);
  auto f_range = arma::conv_to<std::vector<double>>::from(f_grid);
  auto mu_range = arma::conv_to<std::vector<double>>::from(mu_grid);
  auto p_range = arma::conv_to<std::vector<double>>::from(p_grid);

  // Find coexistence.

  functionals::bulk::PhaseSearchConfig search_config{
      .rho_max = 1.0,
      .rho_scan_step = 0.005,
      .newton = {.max_iterations = 300, .tolerance = 1e-10},
  };

  auto coex = functionals::bulk::find_coexistence(model.species, bulk_weights, search_config);

  if (coex) {
    std::cout << "  Coexistence densities:\n";
    std::cout << "    rho_vapor  = " << coex->rho_vapor << "\n";
    std::cout << "    rho_liquid = " << coex->rho_liquid << "\n";

    double mu_coex = functionals::bulk::chemical_potential(
        arma::vec{coex->rho_vapor}, model.species, bulk_weights, 0
    );
    double p_coex = functionals::bulk::pressure(
        arma::vec{coex->rho_vapor}, model.species, bulk_weights
    );
    std::cout << "    mu_coex    = " << mu_coex << "\n";
    std::cout << "    P_coex     = " << p_coex << "\n";
  }

  // Create a homogeneous state and evaluate the functional.

  double rho0 = 0.5;
  auto state = init::homogeneous(model, rho0);

  std::cout << "\n=== Functional evaluation at rho = " << rho0 << " ===\n\n";

  auto weights = functionals::make_weights(fmt_model, model);
  auto result = functionals::total(model, state, weights);

  std::cout << "  Free energy:      " << result.free_energy << "\n";
  std::cout << "  Grand potential:  " << result.grand_potential << "\n";
  std::cout << "  Max |force|:      " << arma::max(arma::abs(result.forces[0])) << "\n";

  // Create an inhomogeneous state (liquid slab) and evaluate.

  std::cout << "\n=== Inhomogeneous state: liquid slab ===\n\n";

  if (coex) {
    long nx = model.grid.shape[0];
    long ny = model.grid.shape[1];
    long nz = model.grid.shape[2];
    double center = model.grid.box_size[0] / 2.0;
    double width = model.grid.box_size[0] / 4.0;
    double interface_width = 1.0;

    // Build 1D x-profile using Armadillo vectorized ops.
    arma::vec x_vals = arma::linspace(0.0, (nx - 1) * model.grid.dx, nx);
    arma::vec profile_1d = coex->rho_vapor + 0.5 * (coex->rho_liquid - coex->rho_vapor) *
        (arma::tanh((x_vals - center + width) / interface_width) -
         arma::tanh((x_vals - center - width) / interface_width));

    // Replicate into 3D: each x-slice is constant across (y, z) planes.
    // Flat layout is z-fastest: index = iz + nz*(iy + ny*ix).
    arma::vec slab_rho = arma::repelem(profile_1d, ny * nz, 1);

    auto slab_state = init::from_profile(model, slab_rho);
    auto slab_result = functionals::total(model, slab_state, weights);

    std::cout << "  Free energy:      " << slab_result.free_energy << "\n";
    std::cout << "  Grand potential:  " << slab_result.grand_potential << "\n";
    std::cout << "  Max |force|:      " << arma::max(arma::abs(slab_result.forces[0])) << "\n";

    // Extract 1D density profile along x (averaged over y, z).
    // Reshape the flat vector into (ny*nz, nx) matrix, then take mean of each column.
    arma::mat rho_mat = arma::reshape(slab_rho, ny * nz, nx);
    arma::vec profile_avg = arma::mean(rho_mat, 0).as_col();
    auto x_coords = arma::conv_to<std::vector<double>>::from(x_vals);
    auto profile = arma::conv_to<std::vector<double>>::from(profile_avg);

    std::cout << "\n  Density profile (x cross-section):\n";
    for (std::size_t i = 0; i < x_coords.size(); i += 10) {
      std::cout << "    x = " << std::setw(6) << x_coords[i]
                << "  rho = " << profile[i] << "\n";
    }

    // Plots.

#ifdef DFT_HAS_MATPLOTLIB
    double p_coex = functionals::bulk::pressure(
        arma::vec{coex->rho_vapor}, model.species, bulk_weights
    );
    plot::pressure_isotherm(rho_range, p_range, coex->rho_vapor, coex->rho_liquid, p_coex, model.temperature);
    plot::density_profile(x_coords, profile, coex->rho_vapor, coex->rho_liquid);
    plot::free_energy(rho_range, f_range, model.temperature);
#endif
  }
}
