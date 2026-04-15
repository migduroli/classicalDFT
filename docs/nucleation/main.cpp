#include "plot.hpp"
#include "utils.hpp"

#include <dftlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace dft;

#ifdef DFT_HAS_MATPLOTLIB
namespace {
  void close_all_figures() {
    try {
      for (int i = 0; i < 200; ++i)
        matplotlibcpp::close();
    } catch (...) {}
  }
} // namespace
#endif

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

  // Step 1: Coexistence conditions

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

  // Steps 2-4: Critical cluster + eigenvalue (with checkpoint)

  auto wall_potential = cfg.wall.build();
  auto wall_field = nucleation::build_external_field(cfg, func.model.grid, wall_potential);

  auto seed_origin = nucleation::seed_center(func.model.grid, cfg);
  auto r = func.model.grid.radial_distances(seed_origin);

  export_dir = std::filesystem::path("exports") / cfg.export_directory();
  auto data_dir = export_dir / "data";
  std::filesystem::create_directories(data_dir);
  std::println(std::cout, "Export directory: {}", export_dir.string());

  auto stored_config = data_dir / "config.toml";
  if (std::filesystem::exists(stored_config)) {
    auto stored_cfg = nucleation::read_config(stored_config);
    if (stored_cfg.model.grid.dx != cfg.model.grid.dx || stored_cfg.model.grid.box_size != cfg.model.grid.box_size) {
      std::println(
          std::cout,
          "Grid mismatch: stored dx={:.4f}, current dx={:.4f}. Discarding old checkpoints.",
          stored_cfg.model.grid.dx,
          cfg.model.grid.dx
      );
      std::filesystem::remove_all(data_dir);
      std::filesystem::create_directories(data_dir);
    }
  }
  std::filesystem::copy_file(config_file, stored_config, std::filesystem::copy_options::overwrite_existing);

  auto checkpoint_critical = data_dir / "rho_critical.arma";
  auto checkpoint_eigvec = data_dir / "eigenvector.arma";
  auto checkpoint_info = data_dir / "critical_info.csv";

  algorithms::saddle_point::ConstrainedResult info;
  arma::vec rho_initial;
  arma::vec rho_critical;
  algorithms::saddle_point::EigenvalueResult eig;

  bool loaded_checkpoint = false;

  if (std::filesystem::exists(checkpoint_critical) && std::filesystem::exists(checkpoint_eigvec)
      && std::filesystem::exists(checkpoint_info)) {
    std::println(std::cout, "\nLoading checkpoint from {}/", data_dir.string());

    rho_critical.load(checkpoint_critical, arma::arma_binary);
    eig.eigenvector.load(checkpoint_eigvec, arma::arma_binary);

    std::ifstream ifs(checkpoint_info);
    std::string header;
    std::getline(ifs, header);
    char comma{};
    ifs >> info.background >> comma >> info.mu_background >> comma >> info.omega_cluster >> comma
        >> info.omega_background >> comma >> info.barrier >> comma >> info.effective_radius >> comma
        >> info.rho_vapor_meta >> comma >> info.rho_liquid_meta >> comma >> eig.eigenvalue;
    eig.converged = true;

    info.boundary_mask = func.model.grid.boundary_mask();
    rho_initial = rho_critical;
    loaded_checkpoint = true;

    std::println(
        std::cout,
        "  barrier={:.6f}  R_eff={:.4f}  eigenvalue={:.6f}",
        info.barrier,
        info.effective_radius,
        eig.eigenvalue
    );
  }

  if (!loaded_checkpoint) {
    // Step 2: Initial seed density

    arma::vec rho0 = nucleation::make_initial_density(cfg, func.model.grid, r, rho_l, rho_out);

    std::println(std::cout, "Seed centre: ({:.3f}, {:.3f}, {:.3f})", seed_origin[0], seed_origin[1], seed_origin[2]);

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

    // Step 3: Constrained minimization (FIRE)

    auto result = nucleation::
        find_critical_cluster(func, cfg, wall_potential, wall_field, seed_origin, rho_v, rho_l, rho_out, mu_out);
    info = result.cluster;
    rho_initial = result.rho_initial;
    rho_critical = info.minimization.densities[0];

    if (info.effective_radius < 0.5 || rho_critical.max() < 2.0 * info.background) {
      std::println(std::cout, "Cluster evaporated. Exiting.");
      return 0;
    }

    // Step 4: Eigenvalue (saddle point verification)

    arma::uvec eig_bdry = nucleation::depletion_mask(rho_critical, info.background, func.model.grid.boundary_mask());

    auto eig_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
      auto result = func.evaluate(rho, info.mu_background, wall_field);
      return {result.grand_potential, fixed_boundary(result.forces[0], eig_bdry)};
    };

    auto eig_init = nucleation::radial_gaussian_guess(r, info.effective_radius, eig_bdry);
    auto deflation = nucleation::translation_modes(rho_critical, func.model.grid);

    eig =
        algorithms::saddle_point::EigenvalueSolver{
            .tolerance = cfg.eigen.tolerance,
            .max_iterations = cfg.eigen.max_iterations,
            .hessian_eps = cfg.eigen.hessian_eps,
            .log_interval = cfg.eigen.log_interval,
            .boundary_mask = eig_bdry,
            .deflation_vectors = deflation,
        }
            .solve(eig_force_fn, rho_critical, eig_init);

    std::println(std::cout, "\nEigenvalue: {:.6f}  converged={}", eig.eigenvalue, eig.converged);

    if (!eig.converged) {
      std::println(std::cout, "Eigenvalue solver did not converge. Aborting.");
      return 1;
    }

    // Save checkpoint.
    rho_critical.save(checkpoint_critical, arma::arma_binary);
    eig.eigenvector.save(checkpoint_eigvec, arma::arma_binary);
    {
      std::ofstream ofs(checkpoint_info);
      ofs << "background,mu_background,omega_cluster,omega_background,barrier,effective_radius,rho_vapor_meta,"
             "rho_liquid_meta,eigenvalue\n";
      ofs << std::format(
          "{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e},{:.12e}\n",
          info.background,
          info.mu_background,
          info.omega_cluster,
          info.omega_background,
          info.barrier,
          info.effective_radius,
          info.rho_vapor_meta,
          info.rho_liquid_meta,
          eig.eigenvalue
      );
    }
    std::println(std::cout, "  Saved checkpoint to {}/", data_dir.string());
  }

#ifdef DFT_HAS_MATPLOTLIB
  try {
    auto critical_profile = nucleation::extract_profile_slice(rho_critical, func.model.grid, cfg);
    auto initial_profile = nucleation::extract_profile_slice(rho_initial, func.model.grid, cfg);
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
  } catch (const std::exception& e) {
    std::println(std::cerr, "Critical-cluster preview plot failed: {}", e.what());
  }
#endif

  if (stop_after_critical_cluster) {
    std::println(std::cout, "Stopping after critical-cluster preview as requested.");
    return 0;
  }

  // Step 5: DDFT from perturbed critical cluster (dissolution + growth)

  arma::uvec ddft_frozen = nucleation::depletion_mask(rho_critical, info.background, info.boundary_mask);
  double dv = func.model.grid.cell_volume();
  double delta_rho = rho_l - rho_v;

  arma::vec rho_vapor = rho_out * arma::ones(arma::size(rho_critical));
  auto boundary_fn = algorithms::dynamics::frozen_boundary(ddft_frozen, rho_vapor);

  auto gc_force_fn = func.grand_potential_callback(info.mu_background, wall_field, ddft_frozen);

  // Perturb along the unstable eigenvector.
  double perturb_amp = cfg.ddft.perturb_scale * std::abs(eig.eigenvalue);
  arma::vec perturb = perturb_amp * eig.eigenvector / arma::norm(eig.eigenvector);

  arma::vec rho_sub = arma::clamp(rho_critical - perturb, 1e-18, arma::datum::inf);
  arma::vec rho_sup = arma::clamp(rho_critical + perturb, 1e-18, arma::datum::inf);

  double mass_sub = arma::accu(rho_sub) * dv;
  double mass_sup = arma::accu(rho_sup) * dv;
  double mass_crit = arma::accu(rho_critical) * dv;

  std::println(
      std::cout,
      "\nStep 5: DDFT dynamics (N_crit={:.2f}, N_sub={:.2f}, N_sup={:.2f})",
      mass_crit,
      mass_sub,
      mass_sup
  );

  auto make_ddft = [&]() {
    return algorithms::dynamics::Simulation{
        .step =
            {
                .dt = cfg.ddft.dt,
                .diffusion_coefficient = cfg.ddft.diffusion_coefficient,
                .min_density = 1e-18,
                .dt_max = cfg.ddft.dt_max,
                .fp_tolerance = cfg.ddft.fp_tolerance,
                .fp_max_iterations = cfg.ddft.fp_max_iterations,
            },
        .n_steps = cfg.ddft.n_steps,
        .snapshot_interval = cfg.ddft.snapshot_interval,
        .log_interval = cfg.ddft.log_interval,
        .boundary = boundary_fn,
        .frozen_mask = ddft_frozen,
    };
  };

  // Dissolution: stop early when energy returns to background.
  std::println(std::cout, "  Running DDFT (dissolution)...");
  auto ddft_dissolve = make_ddft();
  ddft_dissolve.stop_condition = [&, count = 0](int, double, double energy) mutable -> bool {
    count = (std::abs(energy - info.omega_background) < 0.01) ? count + 1 : 0;
    return count >= 3;
  };
  auto sim_dissolve = ddft_dissolve.run({rho_sub}, func.model.grid, gc_force_fn);
  auto dissolution = nucleation::extract_dynamics(sim_dissolve, func.model.grid, r, info.background, delta_rho, cfg);
  std::println(std::cout, "    {} snapshots", dissolution.profiles.size());

  // Growth.
  std::println(std::cout, "  Running DDFT (growth)...");
  auto ddft_growth = make_ddft();
  auto sim_growth = ddft_growth.run({rho_sup}, func.model.grid, gc_force_fn);
  auto growth = nucleation::extract_dynamics(sim_growth, func.model.grid, r, info.background, delta_rho, cfg);
  std::println(std::cout, "    {} snapshots", growth.profiles.size());

  // Critical cluster pathway point.
  double R_crit = dft::effective_radius(rho_critical, info.background, delta_rho, dv);
  double n_crit = dft::cluster_average_density(rho_critical, r, R_crit);
  nucleation::PathwayPoint critical_pt{
      .radius = R_crit,
      .energy = info.omega_cluster,
      .rho_center = nucleation::center_density(rho_critical, func.model.grid, cfg),
      .n_cluster = n_crit,
  };

  // Save pathway CSV.
  {
    std::ofstream csv(data_dir / "dynamics_pathway.csv");
    csv << "branch,step,time,radius,energy,delta_energy,rho_center,n_cluster\n";
    auto write_branch = [&](const std::string& name, const nucleation::DynamicsResult& dyn) {
      for (std::size_t i = 0; i < dyn.pathway.size(); ++i) {
        const auto& p = dyn.pathway[i];
        csv << std::format(
            "{},{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}\n",
            name,
            i,
            dyn.profiles[i].time,
            p.radius,
            p.energy,
            p.energy - info.omega_background,
            p.rho_center,
            p.n_cluster
        );
      }
    };
    write_branch("dissolution", dissolution);
    write_branch("growth", growth);
    std::println(std::cout, "  Saved dynamics pathway to {}", (data_dir / "dynamics_pathway.csv").string());
  }

#ifdef DFT_HAS_MATPLOTLIB
  std::println(std::cout, "\nExporting dynamics plots...");

  std::filesystem::create_directories(export_dir / "dynamics");

  auto critical_profile = nucleation::extract_profile_slice(rho_critical, func.model.grid, cfg);

  try {
    plot::detail::plot_dynamics(
        dissolution.profiles,
        critical_profile,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        "Dissolution (subcritical)",
        0,
        187,
        213,
        0,
        60,
        100,
        export_dir.string() + "/dynamics/dissolution.pdf",
        cfg.model.functional.name
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "Dissolution plot failed: {}", e.what());
  }

  try {
    plot::detail::plot_dynamics(
        growth.profiles,
        critical_profile,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        "Growth (supercritical)",
        226,
        88,
        34,
        120,
        20,
        0,
        export_dir.string() + "/dynamics/growth.pdf",
        cfg.model.functional.name
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "Growth plot failed: {}", e.what());
  }

  try {
    plot::detail::plot_energy_barrier(
        dissolution.pathway,
        growth.pathway,
        critical_pt,
        info.omega_background,
        cfg.model.functional.name,
        export_dir.string()
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "Energy barrier plot failed: {}", e.what());
  }

  try {
    plot::detail::plot_n_vs_R(
        dissolution.pathway,
        growth.pathway,
        critical_pt,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        cfg.model.functional.name,
        export_dir.string()
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "n vs R plot failed: {}", e.what());
  }

  try {
    plot::detail::plot_rho_center_vs_radius(
        dissolution.pathway,
        growth.pathway,
        critical_pt,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        cfg.model.functional.name,
        export_dir.string()
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "rho_center vs R plot failed: {}", e.what());
  }

  try {
    plot::plot_density_frames(
        sim_dissolve,
        func.model.grid,
        cfg,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        "Dissolution",
        export_dir.string() + "/frames/dissolution"
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "Dissolution frames failed: {}", e.what());
  }

  close_all_figures();

  try {
    plot::plot_density_frames(
        sim_growth,
        func.model.grid,
        cfg,
        info.rho_vapor_meta,
        info.rho_liquid_meta,
        "Growth",
        export_dir.string() + "/frames/growth"
    );
  } catch (const std::exception& e) {
    std::println(std::cerr, "Growth frames failed: {}", e.what());
  }

  close_all_figures();
#endif

  std::println(std::cout, "\nDone.");
}
