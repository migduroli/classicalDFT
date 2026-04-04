#include "dft.hpp"
#include "dft/algorithms/saddle_point.hpp"
#include "plot.hpp"
#include "utils.hpp"

#include <filesystem>
#include <iostream>

using namespace dft;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");
  std::cout << std::unitbuf;

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  auto cfg = config::parse_config("config.ini", config::FileType::INI);

  // Model parameters.

  double sigma = config::get<double>(cfg, "model.sigma");
  double epsilon = config::get<double>(cfg, "model.epsilon");
  double rcut = config::get<double>(cfg, "model.cutoff");
  double kT = config::get<double>(cfg, "model.temperature");
  double dx = config::get<double>(cfg, "model.dx");
  double box_length = config::get<double>(cfg, "model.box_length");
  double R0 = config::get<double>(cfg, "droplet.radius");

  physics::Model model{
      .grid = make_grid(dx, {box_length, box_length, box_length}),
      .species = {Species{.name = "LJ",
                          .hard_sphere_diameter = physics::potentials::hard_sphere_diameter(
                              physics::potentials::Potential{physics::potentials::make_lennard_jones(sigma, epsilon, rcut)},
                              kT, physics::potentials::SplitScheme::WeeksChandlerAndersen)}},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = physics::potentials::make_lennard_jones(sigma, epsilon, rcut),
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
          .weight_scheme = physics::WeightScheme::InterpolationQuadraticF,
      }},
      .temperature = kT,
  };

  auto fmt_model = functionals::fmt::RSLT{};
  auto weights = functionals::make_weights(fmt_model, model);

  // Bulk thermodynamics (use grid a_vdw for consistency at coarse dx).

  auto bulk_weights = functionals::make_bulk_weights(fmt_model, model.interactions, kT);
  bulk_weights.mean_field.interactions[0].a_vdw = weights.mean_field.interactions[0].a_vdw;

  auto coex = functionals::bulk::find_coexistence(
      model.species, bulk_weights,
      {.rho_max = 1.0, .rho_scan_step = 0.005,
       .newton = {.max_iterations = 300, .tolerance = 1e-10}});

  if (!coex) {
    std::println(std::cout, "Coexistence not found. Exiting.");
    return 1;
  }

  double rho_v = coex->rho_vapor;
  double rho_l = coex->rho_liquid;
  double rho_out = config::get<double>(cfg, "droplet.density_outside");

  // Ensure supersaturation (rho_out > rho_v).
  double S = rho_out / rho_v;
  if (S < 1.0) {
    rho_out = 1.1 * rho_v;
    S = 1.1;
  }

  double mu_out = functionals::bulk::chemical_potential(
      arma::vec{rho_out}, model.species, bulk_weights, 0);

  std::println(std::cout, "Coexistence: rho_v={:.6f}  rho_l={:.6f}", rho_v, rho_l);
  std::println(std::cout, "Supersaturation: S={:.4f}  rho_out={:.6f}  mu={:.6f}", S, rho_out, mu_out);

  // Phase 1: find critical cluster via fixed-mass FIRE.

  auto r = nucleation::radial_distances(model.grid);
  arma::vec rho0 = nucleation::step_function(r, R0, rho_l, rho_out);
  double target_mass = arma::accu(rho0) * model.grid.cell_volume();

  auto cluster = algorithms::minimization::fixed_mass::minimize(
      model, weights, rho0, mu_out, target_mass,
      {.fire = {
           .dt = config::get<double>(cfg, "fire.dt"),
           .dt_max = config::get<double>(cfg, "fire.dt_max"),
           .alpha_start = config::get<double>(cfg, "fire.alpha_start"),
           .f_alpha = config::get<double>(cfg, "fire.alpha_fac"),
           .force_tolerance = config::get<double>(cfg, "fire.force_tolerance"),
           .max_steps = config::get<int>(cfg, "fire.max_steps"),
       },
       .param = algorithms::minimization::Unbounded{.rho_min = 1e-99},
       .homogeneous_boundary = true,
       .log_interval = config::get<int>(cfg, "fire.log_interval")});

  // Background density from boundary.

  arma::uvec bdry = boundary_mask(model.grid);
  double background = 0.0;
  arma::uword n_bdry = 0;
  for (arma::uword i = 0; i < cluster.densities[0].n_elem; ++i) {
    if (bdry(i)) { background += cluster.densities[0](i); ++n_bdry; }
  }
  if (n_bdry > 0) background /= static_cast<double>(n_bdry);

  double mu_bg = functionals::bulk::chemical_potential(
      arma::vec{background}, model.species, bulk_weights, 0);

  // Grand potential and barrier height.

  auto eval_omega = [&](const arma::vec& rho) {
    auto state = init::from_profile(model, rho);
    state.species[0].chemical_potential = mu_bg;
    return functionals::total(model, state, weights).grand_potential;
  };

  double omega_cluster = eval_omega(cluster.densities[0]);
  double omega_bg = eval_omega(arma::vec(rho0.n_elem, arma::fill::value(background)));
  double delta_omega = omega_cluster - omega_bg;

  double delta_rho = rho_l - rho_v;
  double R_eff = nucleation::effective_radius(
      cluster.densities[0], background, delta_rho, model.grid.cell_volume());

  std::println(std::cout, "\nCritical cluster:");
  std::println(std::cout, "  converged={} ({} iters)", cluster.converged, cluster.iterations);
  std::println(std::cout, "  R_eff={:.4f}  background={:.6f}", R_eff, background);
  std::println(std::cout, "  Delta_Omega={:.6f}  (barrier height)", delta_omega);

  if (R_eff < 0.5 || cluster.densities[0].max() < 2.0 * background) {
    std::println(std::cout, "Cluster evaporated. Exiting.");
    return 0;
  }

  // Phase 2: eigenvalue (confirm saddle point).

  auto eig_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
    auto state = init::from_profile(model, rho);
    state.species[0].chemical_potential = mu_bg;
    auto result = functionals::total(model, state, weights);
    arma::vec grad = result.forces[0];
    grad = fixed_boundary(grad, bdry);
    return {result.grand_potential, grad};
  };

  auto eig = algorithms::saddle_point::smallest_eigenvalue(
      eig_force_fn, cluster.densities[0],
      {.tolerance = config::get<double>(cfg, "eigen.tolerance"),
       .max_iterations = config::get<int>(cfg, "eigen.max_iterations"),
       .hessian_eps = config::get<double>(cfg, "eigen.hessian_eps"),
       .log_interval = config::get<int>(cfg, "eigen.log_interval")});

  std::println(std::cout, "\nEigenvalue: {:.6f}  converged={}", eig.eigenvalue, eig.converged);

  // Phase 3: DDFT dynamics.

  auto ddft_force_fn = [&](const std::vector<arma::vec>& densities)
      -> std::pair<double, std::vector<arma::vec>> {
    auto state = init::from_profile(model, densities[0]);
    state.species[0].chemical_potential = mu_bg;
    auto result = functionals::total(model, state, weights);
    return {result.grand_potential, result.forces};
  };

  double perturb_scale = config::get<double>(cfg, "ddft.perturb_scale");
  int n_steps = config::get<int>(cfg, "ddft.n_steps");

  algorithms::dynamics::SimulationConfig sim_cfg{
      .step = {.dt = config::get<double>(cfg, "ddft.dt"),
               .diffusion_coefficient = 1.0,
               .min_density = 1e-18,
               .dt_max = config::get<double>(cfg, "ddft.dt_max"),
               .fp_tolerance = config::get<double>(cfg, "ddft.fp_tolerance"),
               .fp_max_iterations = config::get<int>(cfg, "ddft.fp_max_iterations")},
      .n_steps = n_steps,
      .snapshot_interval = config::get<int>(cfg, "ddft.snapshot_interval"),
      .log_interval = config::get<int>(cfg, "ddft.log_interval"),
      .energy_offset = omega_bg,
  };

  // Determine eigenvector sign: positive perturbation should grow the cluster.
  // If adding the eigenvector increases the total mass, it points "outward"
  // (growth direction); otherwise we flip it.

  arma::vec ev = eig.eigenvector;
  double delta_mass = arma::accu(ev) * model.grid.cell_volume();
  if (delta_mass < 0.0) ev = -ev;

  std::println(std::cout, "\nGrowth (perturb along +eigenvector):");
  arma::vec rho_grow = cluster.densities[0] + perturb_scale * ev;
  rho_grow = arma::clamp(rho_grow, 1e-18, arma::datum::inf);
  auto sim_grow = algorithms::dynamics::simulate({rho_grow}, model.grid, ddft_force_fn, sim_cfg);

  std::println(std::cout, "\nDissolution (perturb along -eigenvector):");
  arma::vec rho_shrink = cluster.densities[0] - perturb_scale * ev;
  rho_shrink = arma::clamp(rho_shrink, 1e-18, arma::datum::inf);
  auto sim_shrink = algorithms::dynamics::simulate({rho_shrink}, model.grid, ddft_force_fn, sim_cfg);

#ifdef DFT_HAS_MATPLOTLIB
  std::println(std::cout, "\nGenerating plots...");
  double rho_center_critical = nucleation::center_density(cluster.densities[0], model.grid);
  plot::make_plots(
      nucleation::extract_x_slice(cluster.densities[0], model.grid),
      nucleation::extract_x_slice(rho0, model.grid),
      nucleation::extract_dynamics(sim_shrink, model.grid, r, background, delta_rho),
      nucleation::extract_dynamics(sim_grow, model.grid, r, background, delta_rho),
      {.radius = R_eff, .energy = omega_cluster, .rho_center = rho_center_critical},
      omega_bg, rho_v, rho_l);
#endif

  std::println(std::cout, "\nDone.");
}
