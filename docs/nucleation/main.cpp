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
                          .hard_sphere_diameter =
                              physics::potentials::make_lennard_jones(sigma, epsilon, rcut)
                                  .hard_sphere_diameter(kT, physics::potentials::SplitScheme::WeeksChandlerAndersen)}},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = physics::potentials::make_lennard_jones(sigma, epsilon, rcut),
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
          .weight_scheme = physics::WeightScheme::InterpolationQuadraticF,
      }},
      .temperature = kT,
  };

  auto func = functionals::make_functional(functionals::fmt::RSLT{}, model);
  auto eos = func.bulk();

  auto coex = functionals::bulk::PhaseSearch{
      .rho_max = 1.0, .rho_scan_step = 0.005,
      .newton = {.max_iterations = 300, .tolerance = 1e-10}
  }.find_coexistence(eos);

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

  double mu_out = eos.chemical_potential(
      arma::vec{rho_out}, 0);

  std::println(std::cout, "Coexistence: rho_v={:.6f}  rho_l={:.6f}", rho_v, rho_l);
  std::println(std::cout, "Supersaturation: S={:.4f}  rho_out={:.6f}  mu={:.6f}", S, rho_out, mu_out);

  // Phase 1: find critical cluster via fixed-mass FIRE.

  auto r = nucleation::radial_distances(func.model.grid);
  arma::vec rho0 = nucleation::step_function(r, R0, rho_l, rho_out);
  double target_mass = arma::accu(rho0) * func.model.grid.cell_volume();

  auto cluster = algorithms::minimization::Minimizer{
      .fire = {
           .dt = config::get<double>(cfg, "fire.dt"),
           .dt_max = config::get<double>(cfg, "fire.dt_max"),
           .alpha_start = config::get<double>(cfg, "fire.alpha_start"),
           .f_alpha = config::get<double>(cfg, "fire.alpha_fac"),
           .force_tolerance = config::get<double>(cfg, "fire.force_tolerance"),
           .max_steps = config::get<int>(cfg, "fire.max_steps"),
       },
       .param = algorithms::minimization::Unbounded{.rho_min = 1e-99},
       .use_homogeneous_boundary = true,
       .log_interval = config::get<int>(cfg, "fire.log_interval"),
  }.fixed_mass(func.model, func.weights, rho0, mu_out, target_mass);

  // Background density from boundary.

  arma::uvec bdry = func.model.grid.boundary_mask();
  double background = 0.0;
  arma::uword n_bdry = 0;
  for (arma::uword i = 0; i < cluster.densities[0].n_elem; ++i) {
    if (bdry(i)) { background += cluster.densities[0](i); ++n_bdry; }
  }
  if (n_bdry > 0) background /= static_cast<double>(n_bdry);

  double mu_bg = eos.chemical_potential(
      arma::vec{background}, 0);

  // Grand potential and barrier height.

  double omega_cluster = func.grand_potential(cluster.densities[0], mu_bg);
  double omega_bg = func.grand_potential(
      arma::vec(rho0.n_elem, arma::fill::value(background)), mu_bg);

  double delta_omega = omega_cluster - omega_bg;

  double R_eff = nucleation::equimolar_radius(cluster.densities[0], func.model.grid, r);

  // Metastable vapor density at mu_bg (dissolution endpoint).
  auto rho_v_meta_opt = functionals::bulk::density_from_chemical_potential(
      mu_bg, rho_v * 0.9, eos);
  double rho_v_meta = rho_v_meta_opt.value_or(background);

  std::println(std::cout, "\nCritical cluster:");
  std::println(std::cout, "  converged={} ({} iters)", cluster.converged, cluster.iterations);
  std::println(std::cout, "  R_eff={:.4f}  background={:.6f}", R_eff, background);
  std::println(std::cout, "  rho_v(mu)={:.6f}  (metastable vapor at mu_bg)", rho_v_meta);
  std::println(std::cout, "  Delta_Omega={:.6f}  (barrier height)", delta_omega);

  if (R_eff < 0.5 || cluster.densities[0].max() < 2.0 * background) {
    std::println(std::cout, "Cluster evaporated. Exiting.");
    return 0;
  }

  // Phase 2: eigenvalue (confirm saddle point).

  auto eig_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
    auto result = func.evaluate(rho, mu_bg);
    return {result.grand_potential, fixed_boundary(result.forces[0], bdry)};
  };

  auto eig = algorithms::saddle_point::EigenvalueSolver{
      .tolerance = config::get<double>(cfg, "eigen.tolerance"),
      .max_iterations = config::get<int>(cfg, "eigen.max_iterations"),
      .hessian_eps = config::get<double>(cfg, "eigen.hessian_eps"),
      .log_interval = config::get<int>(cfg, "eigen.log_interval"),
  }.solve(eig_force_fn, cluster.densities[0]);

  std::println(std::cout, "\nEigenvalue: {:.6f}  converged={}", eig.eigenvalue, eig.converged);

  // Phase 3: grand-canonical DDFT with reservoir boundary.
  // Standard conserved DDFT with Dirichlet BCs at the box faces:
  // boundary cells are reset to background (reservoir) density after
  // each step, allowing mass to flow in/out as in Lutsko's scheme.

  auto gc_force_fn = func.grand_potential_callback(mu_bg);

  double perturb_scale = config::get<double>(cfg, "ddft.perturb_scale");

  algorithms::dynamics::Simulation ddft_cfg{
      .step = {
          .dt = config::get<double>(cfg, "ddft.dt"),
          .diffusion_coefficient = config::get<double>(cfg, "ddft.diffusion_coefficient"),
          .min_density = 1e-18,
          .dt_max = config::get<double>(cfg, "ddft.dt_max"),
          .fp_tolerance = config::get<double>(cfg, "ddft.fp_tolerance"),
          .fp_max_iterations = config::get<int>(cfg, "ddft.fp_max_iterations"),
      },
      .n_steps = config::get<int>(cfg, "ddft.n_steps"),
      .snapshot_interval = config::get<int>(cfg, "ddft.snapshot_interval"),
      .log_interval = config::get<int>(cfg, "ddft.log_interval"),
      .energy_offset = omega_bg,
      .boundary = algorithms::dynamics::reservoir_boundary(bdry, background),
  };

  // Determine eigenvector sign: positive perturbation should grow the cluster.

  arma::vec ev = eig.eigenvector;
  double delta_mass = arma::accu(ev) * func.model.grid.cell_volume();
  if (delta_mass < 0.0) ev = -ev;

  std::println(std::cout, "\nGrowth (perturb along +eigenvector):");
  arma::vec rho_grow = cluster.densities[0] + perturb_scale * ev;
  rho_grow = arma::clamp(rho_grow, 1e-18, arma::datum::inf);
  auto sim_grow = ddft_cfg.run(
      {rho_grow}, func.model.grid, gc_force_fn);

  std::println(std::cout, "\nDissolution (perturb along -eigenvector):");
  arma::vec rho_shrink = cluster.densities[0] - perturb_scale * ev;
  rho_shrink = arma::clamp(rho_shrink, 1e-18, arma::datum::inf);
  auto sim_shrink = ddft_cfg.run(
      {rho_shrink}, func.model.grid, gc_force_fn);

  auto dyn_grow = nucleation::extract_dynamics(sim_grow, func.model.grid, r);
  auto dyn_shrink = nucleation::extract_dynamics(sim_shrink, func.model.grid, r);

#ifdef DFT_HAS_MATPLOTLIB
  std::println(std::cout, "\nGenerating plots...");
  double rho_center_critical = nucleation::center_density(cluster.densities[0], func.model.grid);
  plot::make_plots(
      nucleation::extract_x_slice(cluster.densities[0], func.model.grid),
      nucleation::extract_x_slice(rho0, func.model.grid),
      dyn_shrink, dyn_grow,
      {.radius = R_eff, .energy = omega_cluster, .rho_center = rho_center_critical},
      omega_bg, rho_v, rho_l, rho_v_meta);
#endif

  std::println(std::cout, "\nDone.");
}
