#include "plot.hpp"
#include "utils.hpp"

#include <dftlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace dft;

int main(int argc, char* argv[]) {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");
  std::cout << std::unitbuf;

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  std::filesystem::path export_dir;

  std::string config_file;
  bool stop_after_critical_cluster = false;

  for (int argi = 1; argi < argc; ++argi) {
    std::string_view arg = argv[argi];
    if ((arg == "--config" || arg == "-c") && argi + 1 < argc) {
      config_file = argv[++argi];
    } else if (arg == "--stop-after-critical-cluster") {
      stop_after_critical_cluster = true;
    } else if (arg.ends_with(".toml")) {
      config_file = arg;
    }
  }

  if (config_file.empty()) {
    std::println(std::cerr, "Usage: doc_nucleation --config <file.toml> [--stop-after-critical-cluster]");
    return 1;
  }

  auto cfg = nucleation::read_config(config_file);

  // Model setup

  auto potential = cfg.model.potential.build();
  auto interaction_split = nucleation::split_scheme(cfg.model.split.interaction);
  auto hsd_split = nucleation::split_scheme(cfg.model.split.hard_sphere_diameter);
  double kT = cfg.model.functional.temperature;

  std::println(
      std::cout,
      "Potential: {}  split: {}  hsd_scheme: {}  kT={:.4f}",
      potential.name(),
      cfg.model.split.interaction,
      cfg.model.split.hard_sphere_diameter,
      kT
  );

  bool has_wall = cfg.wall.is_active();

  physics::Model model{
      .grid = make_grid(cfg.model.grid.dx, cfg.model.grid.box_size, {true, true, !has_wall}),
      .species = {Species{
          .name = std::string(potential.name()),
          .hard_sphere_diameter = potential.hard_sphere_diameter(kT, hsd_split),
      }},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = potential,
          .split = interaction_split,
          .weight_scheme = physics::WeightScheme::InterpolationQuadraticF,
      }},
      .temperature = kT,
  };

  auto func = functionals::make_functional(functionals::fmt::FMTModel::from_name(cfg.model.functional.name), model);
  auto eos = func.bulk();

  // Thermodynamics: coexistence and spinodal

  auto phase_search = functionals::bulk::PhaseSearch{
      .rho_max = 1.0,
      .rho_scan_step = 0.005,
      .newton = {.max_iterations = 300, .tolerance = 1e-10}
  };

  auto coex = phase_search.find_coexistence(eos);
  if (!coex) {
    std::println(std::cout, "Coexistence not found. Exiting.");
    return 1;
  }

  double rho_v = coex->rho_vapor;
  double rho_l = coex->rho_liquid;

  auto spinodal = phase_search.find_spinodal(eos);
  double rho_spinodal_v = spinodal ? spinodal->rho_low : rho_l;
  double S_max = rho_spinodal_v / rho_v;

  std::println(std::cout, "Coexistence: rho_v={:.6f}  rho_l={:.6f}", rho_v, rho_l);
  std::println(std::cout, "Spinodal (vapor): rho_sp={:.6f}  S_max={:.2f}", rho_spinodal_v, S_max);

  double rho_out = cfg.droplet.supersaturation * rho_v;
  double mu_out = eos.chemical_potential(arma::vec{rho_out}, 0);

  std::println(
      std::cout,
      "Supersaturation: S={:.4f}  rho_out={:.6f}  mu={:.6f}",
      cfg.droplet.supersaturation,
      rho_out,
      mu_out
  );
  std::println(std::cout, "Initial seed: {}", cfg.seed.kind);

  // Phase 1: critical cluster via constrained FIRE

  auto wall_potential = cfg.wall.build();
  auto wall_field = nucleation::build_external_field(cfg, func.model.grid, wall_potential);

  auto seed_origin = nucleation::seed_center(func.model.grid, cfg);
  auto r = func.model.grid.radial_distances(seed_origin);
  arma::vec rho0 = nucleation::make_initial_density(cfg, func.model.grid, r, rho_l, rho_out);

  std::println(std::cout, "Seed centre: ({:.3f}, {:.3f}, {:.3f})", seed_origin[0], seed_origin[1], seed_origin[2]);

  export_dir = std::filesystem::path("exports") / cfg.export_directory();
  std::filesystem::create_directories(export_dir);
  std::println(std::cout, "Export directory: {}", export_dir.string());

#ifdef DFT_HAS_MATPLOTLIB
  try {
    plot::plot_initial_condition(
        nucleation::extract_profile_slice(rho0, func.model.grid, cfg),
        rho0,
        func.model.grid,
        cfg,
        rho_v,
        rho_l,
        cfg.model.functional.name,
        export_dir.string()
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "Initial condition plot failed: {}", e.what());
  }
#endif

  auto result = nucleation::
      find_critical_cluster(func, cfg, wall_potential, wall_field, seed_origin, rho_v, rho_l, rho_out, mu_out);
  const auto& info = result.cluster;
  const auto& rho_initial = result.rho_initial;
  const auto& rho_critical = info.minimization.densities[0];

  if (info.effective_radius < 0.5 || rho_critical.max() < 2.0 * info.background) {
    std::println(std::cout, "Cluster evaporated. Exiting.");
    return 0;
  }

  auto critical_profile = nucleation::extract_profile_slice(rho_critical, func.model.grid, cfg);
  auto initial_profile = nucleation::extract_profile_slice(rho_initial, func.model.grid, cfg);

#ifdef DFT_HAS_MATPLOTLIB
  try {
    auto critical_views = plot::detail::density_views(rho_critical, func.model.grid, cfg);
    std::println(std::cout, "\nExporting critical-cluster preview...");
    plot::detail::plot_critical_cluster(
        critical_profile,
        initial_profile,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        cfg.model.functional.name,
        cfg,
        export_dir.string()
    );

    plot::detail::plot_density_views(
        critical_views,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        std::format(R"(Critical cluster [{}])", cfg.model.functional.name),
        export_dir.string() + "/critical/sections.pdf",
        cfg
    );

    auto critical_window = plot::detail::packing_window(critical_views, info.rho_vapor_meta, info.rho_liquid_meta);
    plot::detail::plot_density_views(
        critical_views,
        critical_window.vmin,
        critical_window.vmax,
        std::format(R"(Critical cluster packing contrast [{}])", cfg.model.functional.name),
        export_dir.string() + "/critical/sections_packing.pdf",
        cfg,
        "turbo",
        128,
        "both"
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "Critical-cluster preview plot failed: {}", e.what());
  }
#endif

  if (stop_after_critical_cluster) {
    std::println(std::cout, "Stopping after critical-cluster preview as requested.");
    return 0;
  }

  // Phase 2: eigenvalue (confirm saddle point)
  // Radial initial guess avoids near-zero translation modes on fine grids.
  // Mask all box faces AND the wall depletion zone (where rho ~ 0 and
  // the ideal-gas Hessian 1/rho diverges) to keep H well-conditioned.

  arma::uvec eig_bdry = func.model.grid.boundary_mask();
  double depletion_threshold = 0.01 * info.background;
  for (arma::uword i = 0; i < rho_critical.n_elem; ++i) {
    if (rho_critical(i) < depletion_threshold) {
      eig_bdry(i) = 1;
    }
  }

  auto eig_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
    auto result = func.evaluate(rho, info.mu_background, wall_field);
    return {result.grand_potential, fixed_boundary(result.forces[0], eig_bdry)};
  };

  arma::vec eig_init = rho_critical - info.background;
  eig_init /= arma::norm(eig_init);

  auto eig =
      algorithms::saddle_point::EigenvalueSolver{
          .tolerance = cfg.eigen.tolerance,
          .max_iterations = cfg.eigen.max_iterations,
          .hessian_eps = cfg.eigen.hessian_eps,
          .log_interval = cfg.eigen.log_interval,
          .boundary_mask = eig_bdry,
      }
          .solve(eig_force_fn, rho_critical, eig_init);

  std::println(std::cout, "\nEigenvalue: {:.6f}  converged={}", eig.eigenvalue, eig.converged);

  if (!eig.converged) {
    std::println(std::cout, "Eigenvalue solver did not converge after {} iterations. Aborting.", eig.iterations);
    return 1;
  }

  // Phase 3: DDFT dynamics (growth and dissolution)
  // The wall depletion zone has rho ~ 0 and forces ~ 1/rho ~ 1e18.
  // The 3-arg grand_potential_callback zeros forces at frozen points
  // so the spectral RHS stays clean; frozen_boundary pins densities.

  arma::uvec ddft_frozen = info.boundary_mask;
  for (arma::uword i = 0; i < rho_critical.n_elem; ++i) {
    if (rho_critical(i) < depletion_threshold) {
      ddft_frozen(i) = 1;
    }
  }

  auto gc_force_fn = func.grand_potential_callback(info.mu_background, wall_field, ddft_frozen);

  algorithms::dynamics::Simulation ddft{
      .step =
          {.dt = cfg.ddft.dt,
           .diffusion_coefficient = cfg.ddft.diffusion_coefficient,
           .min_density = 1e-18,
           .dt_max = cfg.ddft.dt_max,
           .fp_tolerance = cfg.ddft.fp_tolerance,
           .fp_max_iterations = cfg.ddft.fp_max_iterations},
      .n_steps = cfg.ddft.n_steps,
      .snapshot_interval = cfg.ddft.snapshot_interval,
      .log_interval = cfg.ddft.log_interval,
      .energy_offset = info.omega_background,
      .boundary = algorithms::dynamics::frozen_boundary(ddft_frozen, rho_critical),
      .frozen_mask = ddft_frozen,
  };

  arma::vec ev = algorithms::saddle_point::orient_eigenvector(eig.eigenvector, func.model.grid.cell_volume());

  double dv = func.model.grid.cell_volume();
  double ev_mass = arma::accu(ev) * dv;
  double mass_critical = arma::accu(rho_critical) * dv;
  auto rho_grow_test =
      algorithms::saddle_point::eigenvector_perturbation(rho_critical, ev, cfg.ddft.perturb_scale, +1.0);
  double mass_grow = arma::accu(rho_grow_test) * dv;
  auto [e_grow_test, _] = gc_force_fn({rho_grow_test});
  std::println(
      std::cout,
      "\nEigenvector: delta_mass={:.4f}  mass_critical={:.4f}  mass_grow={:.4f}  E_grow={:.6f}  (barrier={:.6f})",
      ev_mass,
      mass_critical,
      mass_grow,
      e_grow_test - info.omega_background,
      info.barrier
  );

  std::println(std::cout, "\nGrowth (perturb along +eigenvector):");
  auto rho_grow = algorithms::saddle_point::eigenvector_perturbation(rho_critical, ev, cfg.ddft.perturb_scale, +1.0);
  auto sim_grow = ddft.run({rho_grow}, func.model.grid, gc_force_fn);

  // Save growth result immediately so it survives a plotting crash.
  auto data_dir = export_dir / "data";
  std::filesystem::create_directories(data_dir);
  rho_critical.save(data_dir / "rho_critical.arma", arma::arma_binary);
  sim_grow.densities[0].save(data_dir / "rho_growth_final.arma", arma::arma_binary);
  std::println(std::cout, "  Saved critical + growth densities to {}/", data_dir.string());

  std::println(std::cout, "\nDissolution (perturb along -eigenvector):");
  auto rho_shrink = algorithms::saddle_point::eigenvector_perturbation(rho_critical, ev, cfg.ddft.perturb_scale, -1.0);

  auto ddft_dissolve = ddft;
  ddft_dissolve.stop_condition = [&, count = 0](int, double, double energy) mutable -> bool {
    count = (std::abs(energy - info.omega_background) < 0.01) ? count + 1 : 0;
    return count >= 3;
  };
  auto sim_shrink = ddft_dissolve.run({rho_shrink}, func.model.grid, gc_force_fn);

  // Save dissolution result immediately.
  sim_shrink.densities[0].save(data_dir / "rho_dissolution_final.arma", arma::arma_binary);
  std::println(std::cout, "  Saved dissolution density to {}/", data_dir.string());

  // Post-processing and plots

  double delta_rho = rho_l - rho_v;
  auto dyn_grow = nucleation::extract_dynamics(sim_grow, func.model.grid, r, info.background, delta_rho, cfg);
  auto dyn_shrink = nucleation::extract_dynamics(sim_shrink, func.model.grid, r, info.background, delta_rho, cfg);

#ifdef DFT_HAS_MATPLOTLIB
  std::println(std::cout, "\nGenerating plots...");
  try {
    double rho_center_critical = nucleation::center_density(rho_critical, func.model.grid, cfg);

    plot::make_plots(
        critical_profile,
        initial_profile,
        rho_critical,
        dyn_shrink,
        dyn_grow,
        sim_shrink,
        sim_grow,
        func.model.grid,
        cfg,
        {.radius = info.effective_radius, .energy = info.omega_cluster, .rho_center = rho_center_critical},
        info.omega_background,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        cfg.model.functional.name,
        export_dir.string()
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "Plotting failed: {} — simulation data saved in {}/data/", e.what(), export_dir.string());
  }
#endif

  std::println(std::cout, "\nDone.");
}
