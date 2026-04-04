#include "dft.hpp"
#include "dft/algorithms/eigenvalue.hpp"
#include "dft/algorithms/hessian.hpp"
#include "legacy/classicaldft.hpp"
#include "legacy/algorithms.hpp"
#include "plot.hpp"
#include "utils.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>

using namespace dft;

static auto make_fm(const functionals::fmt::Measures& m) -> legacy::fmt::FundamentalMeasures {
  legacy::fmt::FundamentalMeasures fm;
  fm.eta = m.eta;
  fm.s0 = m.n0;
  fm.s1 = m.n1;
  fm.s2 = m.n2;
  for (int i = 0; i < 3; ++i) {
    fm.v1[i] = m.v0(i);
    fm.v2[i] = m.v1(i);
    for (int j = 0; j < 3; ++j) {
      fm.T[i][j] = m.T(i, j);
    }
  }
  fm.calculate_derived_quantities();
  return fm;
}

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");
  std::cout << std::fixed << std::setprecision(12);
  std::cout << std::unitbuf;  // flush after every output

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  auto cfg = config::parse_config("config.ini", config::FileType::INI);

  // ================================================================
  // Step 0: Parameters — identical to Jim's Droplet/LJ/M103/T0.8/input.dat
  // ================================================================

  double sigma = config::get<double>(cfg, "model.sigma");     // 1.0
  double epsilon = config::get<double>(cfg, "model.epsilon");  // 1.0
  double rcut = config::get<double>(cfg, "model.cutoff");      // 3.0
  double kT = config::get<double>(cfg, "model.temperature");   // 0.8
  double dx = config::get<double>(cfg, "model.dx");
  double box_length = config::get<double>(cfg, "model.box_length");  // 12.8
  double rho_in = config::get<double>(cfg, "droplet.density_inside");  // 0.75
  double rho_out = config::get<double>(cfg, "droplet.density_outside");  // 0.0095
  double R0 = config::get<double>(cfg, "droplet.radius");  // 3.0

  std::cout << "================================================================\n";
  std::cout << "  SYSTEMATIC COMPARISON: our code vs Jim's code\n";
  std::cout << "================================================================\n\n";

  std::cout << "  Parameters:\n";
  std::cout << "    sigma=" << sigma << " eps=" << epsilon << " rcut=" << rcut << "\n";
  std::cout << "    kT=" << kT << " dx=" << dx << " L=" << box_length << "\n";
  std::cout << "    rho_in=" << rho_in << " rho_out=" << rho_out << " R=" << R0 << "\n\n";

  // ================================================================
  // Step 1: Potential quantities
  // ================================================================

  std::cout << "================================================================\n";
  std::cout << "  Step 1: LJ potential\n";
  std::cout << "================================================================\n\n";

  // Jim's code
  auto jim_lj = legacy::potentials::make_LJ(sigma, epsilon, rcut);
  std::cout << "  Jim:  shift = " << jim_lj.shift << "\n";
  std::cout << "  Jim:  rmin  = " << jim_lj.rmin << "\n";
  std::cout << "  Jim:  Vmin  = " << jim_lj.Vmin << "\n";
  std::cout << "  Jim:  r0    = " << jim_lj.r0 << "\n";

  // Our code
  auto our_lj = physics::potentials::make_lennard_jones(sigma, epsilon, rcut);
  std::cout << "  Ours: shift = " << our_lj.epsilon_shift << "\n";
  std::cout << "  Ours: rmin  = " << our_lj.r_min << "\n";
  std::cout << "  Ours: Vmin  = " << our_lj.v_min << "\n";
  std::cout << "  Ours: r0    = " << our_lj.r_zero << "\n\n";

  // Test V(r) at several points
  std::cout << "  V(r) comparison:\n";
  std::cout << "  " << std::setw(8) << "r" << std::setw(20) << "Jim V(r)"
            << std::setw(20) << "Ours V(r)" << std::setw(15) << "diff\n";
  for (double r : {0.9, 1.0, 1.05, 1.1, 1.12246, 1.5, 2.0, 2.5, 2.9, 3.0}) {
    double vj = legacy::potentials::V(jim_lj, r);
    double vo = physics::potentials::energy(physics::potentials::Potential{our_lj}, r);
    std::cout << "  " << std::setw(8) << r << std::setw(20) << vj
              << std::setw(20) << vo << std::setw(15) << (vj - vo) << "\n";
  }

  // Test Watt(r)
  std::cout << "\n  Watt(r) = attractive tail comparison:\n";
  std::cout << "  " << std::setw(8) << "r" << std::setw(20) << "Jim Watt(r)"
            << std::setw(20) << "Ours att(r)" << std::setw(15) << "diff\n";
  auto our_pot = physics::potentials::Potential{our_lj};
  for (double r : {0.5, 0.9, 1.0, 1.12246, 1.2, 1.5, 2.0, 2.5, 2.9, 3.0, 3.1}) {
    double wj = legacy::potentials::Watt(jim_lj, r);
    double wo = physics::potentials::attractive(our_pot, r, physics::potentials::SplitScheme::WeeksChandlerAndersen);
    std::cout << "  " << std::setw(8) << r << std::setw(20) << wj
              << std::setw(20) << wo << std::setw(15) << (wj - wo) << "\n";
  }

  // ================================================================
  // Step 2: HSD
  // ================================================================

  std::cout << "\n================================================================\n";
  std::cout << "  Step 2: Hard-sphere diameter\n";
  std::cout << "================================================================\n\n";

  double hsd_jim = legacy::potentials::getHSD(jim_lj, kT);
  double hsd_ours = physics::potentials::hard_sphere_diameter(our_pot, kT, physics::potentials::SplitScheme::WeeksChandlerAndersen);
  std::cout << "  Jim:  HSD = " << hsd_jim << "\n";
  std::cout << "  Ours: HSD = " << hsd_ours << "\n";
  std::cout << "  diff = " << (hsd_jim - hsd_ours) << "\n";

  // ================================================================
  // Step 3: VDW parameter (analytical, from integration)
  // ================================================================

  std::cout << "\n================================================================\n";
  std::cout << "  Step 3: VDW parameter (analytical)\n";
  std::cout << "================================================================\n\n";

  double vdw_jim = legacy::potentials::getVDW(jim_lj, kT);
  double vdw_ours = physics::potentials::vdw_integral(our_pot, kT, physics::potentials::SplitScheme::WeeksChandlerAndersen);
  std::cout << "  Jim:  a_vdw/(2pi/kT) = " << vdw_jim << " (a_vdw = " << vdw_jim << ")\n";
  std::cout << "  Ours: vdw_integral   = " << vdw_ours << "\n";
  std::cout << "  diff = " << (vdw_jim - vdw_ours) << "\n";
  // Jim's Interaction stores a_vdw_ = 0.5 * (what getVDW returns)? No:
  // Jim's Mu function: mu += 0.5*a_vdw_*x[s2]
  // and a_vdw_ = sum_R w(R) in the grid.
  // But for the VDW parameter: Jim reports v_->getVDW_Parameter(kT) / kT
  // and the interaction Fhelmholtz = 0.5*a_vdw*rho*rho.
  // Our make_bulk_weights: a_vdw = 2 * vdw_integral(pot, kT, split)
  // And mu_mf = a_vdw * rho. So we multiply by 2 in make_bulk_weights.
  double our_a_vdw_bulk = 2.0 * vdw_ours;
  std::cout << "  Our bulk a_vdw = 2*vdw_integral = " << our_a_vdw_bulk << "\n";

  // ================================================================
  // Step 4: Grid weight a_vdw (discrete sum, QF quadrature)
  // ================================================================

  std::cout << "\n================================================================\n";
  std::cout << "  Step 4: Grid weight a_vdw (QF quadrature, dx=" << dx << ")\n";
  std::cout << "================================================================\n\n";

  long N = static_cast<long>(std::round(box_length / dx));
  auto jim_wt = legacy::interactions::compute_a_vdw_QF_detailed(sigma, epsilon, rcut, kT, dx);
  std::cout << "  Jim QF:  a_vdw (before /kT) = " << jim_wt.a_vdw << "\n";
  std::cout << "  Jim QF:  a_vdw/kT           = " << jim_wt.a_vdw_over_kT << "\n";
  std::cout << "  Jim QF:  nonzero points     = " << jim_wt.n_nonzero << "\n";

  // Our code's grid a_vdw — generate using our make_mean_field_weights
  auto model_grid = make_grid(dx, {box_length, box_length, box_length});
  auto potential_obj = physics::potentials::make_lennard_jones(sigma, epsilon, rcut);

  // Use InterpolationQuadraticF (Jim's QF scheme — 27-point)
  physics::Interaction inter_QF{
      .species_i = 0, .species_j = 0, .potential = potential_obj,
      .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      .weight_scheme = physics::WeightScheme::InterpolationQuadraticF};
  auto mf_QF = functionals::make_mean_field_weights(model_grid, {inter_QF}, kT);
  std::cout << "  Ours QF: a_vdw (grid)       = " << mf_QF.interactions[0].a_vdw << "\n";

  // Use InterpolationLinearF (our old default)
  physics::Interaction inter_LF{
      .species_i = 0, .species_j = 0, .potential = potential_obj,
      .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      .weight_scheme = physics::WeightScheme::InterpolationLinearF};
  auto mf_LF = functionals::make_mean_field_weights(model_grid, {inter_LF}, kT);
  std::cout << "  Ours LF: a_vdw (grid)       = " << mf_LF.interactions[0].a_vdw << "\n";

  // Use InterpolationZero (point evaluation at cell center)
  physics::Interaction inter_Z{
      .species_i = 0, .species_j = 0, .potential = potential_obj,
      .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      .weight_scheme = physics::WeightScheme::InterpolationZero};
  auto mf_Z = functionals::make_mean_field_weights(model_grid, {inter_Z}, kT);
  std::cout << "  Ours Z:  a_vdw (grid)       = " << mf_Z.interactions[0].a_vdw << "\n";

  std::cout << "\n  NOTE: Jim uses QF (27-point), we use LF (8-point). These differ!\n";
  std::cout << "  The QF scheme is critical for correct coarse-grid weights.\n";

  // ================================================================
  // Step 5: FMT setup and chemical potential / coexistence
  // ================================================================

  std::cout << "\n================================================================\n";
  std::cout << "  Step 5: Chemical potential and coexistence\n";
  std::cout << "================================================================\n\n";

  // Build model with current (LF) scheme and with QF a_vdw
  physics::Model model{
      .grid = model_grid,
      .species = {Species{.name = "LJ", .hard_sphere_diameter = hsd_ours}},
      .interactions = {{
          .species_i = 0, .species_j = 0, .potential = potential_obj,
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
          .weight_scheme = physics::WeightScheme::InterpolationQuadraticF,
      }},
      .temperature = kT,
  };

  // Build bulk weights using the GRID a_vdw, not the analytical integral.
  // Jim uses his grid a_vdw for ALL bulk thermodynamics (mu, pressure,
  // coexistence).  The analytical value differs at coarse dx.
  auto bulk_weights = functionals::make_bulk_weights(
      functionals::fmt::RSLT{}, model.interactions, kT);

  double analytical_a_vdw = bulk_weights.mean_field.interactions[0].a_vdw;
  // Override with grid a_vdw from the QF weight generation:
  bulk_weights.mean_field.interactions[0].a_vdw = mf_QF.interactions[0].a_vdw;

  double mu_rho_out = functionals::bulk::chemical_potential(
      arma::vec{rho_out}, model.species, bulk_weights, 0);
  std::cout << "  mu(rho_out=" << rho_out << ") = " << mu_rho_out << "\n";

  std::cout << "  analytical a_vdw = " << analytical_a_vdw << "\n";
  std::cout << "  grid a_vdw (QF)  = " << bulk_weights.mean_field.interactions[0].a_vdw << "\n";
  std::cout << "  Jim grid a_vdw/kT = " << jim_wt.a_vdw_over_kT << "\n\n";

  // Debug: scan dP/drho to see if there's a van der Waals loop
  std::cout << "  Scanning dP/drho for van der Waals loop:\n";
  double prev_dp_test = 0;
  for (double rho_test = 0.005; rho_test < 0.7; rho_test += 0.005) {
    double h = 1e-7;
    double p_plus = functionals::bulk::pressure(arma::vec{rho_test + h}, model.species, bulk_weights);
    double p_minus = functionals::bulk::pressure(arma::vec{rho_test - h}, model.species, bulk_weights);
    double dp = (p_plus - p_minus) / (2.0 * h);
    if (prev_dp_test > 0 && dp <= 0) {
      std::cout << " ** SIGN CHANGE (pos->neg) at rho=" << rho_test << "\n";
    }
    if (prev_dp_test < 0 && dp >= 0) {
      std::cout << " ** SIGN CHANGE (neg->pos) at rho=" << rho_test << "\n";
    }
    prev_dp_test = dp;
  }

  // Coexistence
  auto spinodal = functionals::bulk::find_spinodal(model.species, bulk_weights,
      {.rho_max = 1.0, .rho_scan_step = 0.005,
       .newton = {.max_iterations = 300, .tolerance = 1e-10}});
  if (spinodal) {
    std::cout << "  Spinodal:     rho_low = " << spinodal->rho_low
              << "  rho_high = " << spinodal->rho_high << "\n";
  } else {
    std::cout << "  Spinodal: NOT FOUND\n";
  }

  auto coex = functionals::bulk::find_coexistence(
      model.species, bulk_weights,
      {.rho_max = 1.0, .rho_scan_step = 0.005,
       .newton = {.max_iterations = 300, .tolerance = 1e-10}});

  if (coex) {
    std::cout << "  Coexistence:  rho_v = " << coex->rho_vapor
              << "  rho_l = " << coex->rho_liquid << "\n";
  } else {
    std::cout << "  Coexistence: NOT FOUND\n";
  }

  double rho_v = coex ? coex->rho_vapor : rho_out;
  double rho_l = coex ? coex->rho_liquid : rho_in;
  double S = rho_out / rho_v;
  std::cout << "  Supersaturation S = rho_out/rho_v = " << S << "\n";

  // If the requested rho_out is below vapor coexistence, adjust it
  // to a supersaturation of ~1.1 so that a cluster can survive.
  if (S < 1.0 && coex) {
    double S_target = 1.1;
    rho_out = S_target * rho_v;
    S = S_target;
    std::cout << "  ** rho_out adjusted to " << rho_out << " (S=" << S << ")\n";
    mu_rho_out = functionals::bulk::chemical_potential(
        arma::vec{rho_out}, model.species, bulk_weights, 0);
    std::cout << "  ** mu(rho_out) updated to " << mu_rho_out << "\n";
  }
  std::cout << "\n";

  // ================================================================
  // Step 6: Now run the actual FIRE minimisation (Phase 1)
  // ================================================================

  std::cout << "================================================================\n";
  std::cout << "  Step 6: FIRE minimisation (fixed mass, Phase 1)\n";
  std::cout << "================================================================\n\n";

  auto weights = functionals::make_weights(functionals::fmt::RSLT{}, model);

  std::cout << "  Grid a_vdw used in minimisation: "
            << weights.mean_field.interactions[0].a_vdw << "\n";
  std::cout << "  Jim QF grid a_vdw/kT:            " << jim_wt.a_vdw_over_kT << "\n";
  std::cout << "  Difference = " << (weights.mean_field.interactions[0].a_vdw - jim_wt.a_vdw_over_kT)
            << "\n\n";

  auto r = nucleation::radial_distances(model.grid);
  arma::vec rho0 = nucleation::step_function(r, R0, rho_l, rho_out);
  double target_mass = arma::accu(rho0) * model.grid.cell_volume();

  std::cout << "  Initial: rho_in=" << rho_l << " rho_out=" << rho_out << " R=" << R0 << "\n";
  std::cout << "  Target mass = " << target_mass << "\n\n";

  auto cluster = algorithms::minimize_at_fixed_mass(
      model, weights, rho0, mu_rho_out, target_mass,
      {.fire = {
           .dt = config::get<double>(cfg, "fire.dt"),
           .dt_max = config::get<double>(cfg, "fire.dt_max"),
           .alpha_start = config::get<double>(cfg, "fire.alpha_start"),
           .f_alpha = config::get<double>(cfg, "fire.alpha_fac"),
           .force_tolerance = config::get<double>(cfg, "fire.force_tolerance"),
           .max_steps = config::get<int>(cfg, "fire.max_steps"),
       },
       .param = algorithms::parametrization::Unbounded{.rho_min = 1e-99},
       .homogeneous_boundary = true,
       .log_interval = config::get<int>(cfg, "fire.log_interval")});

  arma::uvec bdry = boundary_mask(model.grid);
  double background = 0.0;
  arma::uword n_bdry = 0;
  for (arma::uword i = 0; i < cluster.densities[0].n_elem; ++i) {
    if (bdry(i)) { background += cluster.densities[0](i); n_bdry++; }
  }
  if (n_bdry > 0) background /= static_cast<double>(n_bdry);

  // Compute the grand potential at the cluster using mu from the background density.
  double mu_bg = functionals::bulk::chemical_potential(
      arma::vec{background}, model.species, bulk_weights, 0);
  auto cluster_state = init::from_profile(model, cluster.densities[0]);
  cluster_state.species[0].chemical_potential = mu_bg;
  auto cluster_result = functionals::total(model, cluster_state, weights);

  auto bg_state = init::from_profile(model, arma::vec(rho0.n_elem, arma::fill::value(background)));
  bg_state.species[0].chemical_potential = mu_bg;
  double omega_bg = functionals::total(model, bg_state, weights).grand_potential;

  std::cout << "\n  Converged = " << cluster.converged << " (" << cluster.iterations << " iters)\n";
  std::cout << "  F (Helmholtz) = " << cluster.free_energy << "\n";
  std::cout << "  mu_bg = " << mu_bg << "\n";
  std::cout << "  Omega* = " << cluster_result.grand_potential << "\n";
  std::cout << "  Omega_bg = " << omega_bg << "\n";
  std::cout << "  Delta_Omega/kT = " << (cluster_result.grand_potential - omega_bg) << "\n";
  std::cout << "  Background = " << background << "\n";
  std::cout << "  rho_max = " << cluster.densities[0].max()
            << "  rho_min = " << cluster.densities[0].min() << "\n";
  std::cout << "  Final mass = " << arma::accu(cluster.densities[0]) * model.grid.cell_volume() << "\n";

  // ================================================================
  // Step 6b: Jim's FIRE (side-by-side comparison)
  // ================================================================

  std::cout << "\n================================================================\n";
  std::cout << "  Step 6b: Jim's FIRE (lutsko_alg) — same initial condition\n";
  std::cout << "================================================================\n\n";

  // Force function with mu=0 (fixed mass uses lambda projection).
  auto jim_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
    auto state = init::from_profile(model, rho);
    state.species[0].chemical_potential = 0.0;
    auto result = functionals::total(model, state, weights);
    return {result.free_energy, result.forces[0]};
  };

  auto jim_cluster = legacy::algorithms::fire_minimize_fixed_mass(
      jim_force_fn, rho0, target_mass, model.grid.cell_volume(),
      model.grid.shape[0], model.grid.shape[1], model.grid.shape[2],
      {.dt = config::get<double>(cfg, "fire.dt"),
       .dt_max = config::get<double>(cfg, "fire.dt_max"),
       .alpha_start = config::get<double>(cfg, "fire.alpha_start"),
       .alpha_fac = config::get<double>(cfg, "fire.alpha_fac"),
       .force_tolerance = config::get<double>(cfg, "fire.force_tolerance"),
       .max_steps = config::get<int>(cfg, "fire.max_steps"),
       .log_interval = config::get<int>(cfg, "fire.log_interval")});

  // Compute Jim's grand potential using the same background density.
  double jim_bg = 0.0;
  arma::uword jim_n_bdry = 0;
  for (arma::uword i = 0; i < jim_cluster.density.n_elem; ++i) {
    if (bdry(i)) { jim_bg += jim_cluster.density(i); jim_n_bdry++; }
  }
  if (jim_n_bdry > 0) jim_bg /= static_cast<double>(jim_n_bdry);

  double jim_mu_bg = functionals::bulk::chemical_potential(
      arma::vec{jim_bg}, model.species, bulk_weights, 0);
  auto jim_state = init::from_profile(model, jim_cluster.density);
  jim_state.species[0].chemical_potential = jim_mu_bg;
  auto jim_result = functionals::total(model, jim_state, weights);

  auto jim_bg_state = init::from_profile(model, arma::vec(rho0.n_elem, arma::fill::value(jim_bg)));
  jim_bg_state.species[0].chemical_potential = jim_mu_bg;
  double jim_omega_bg = functionals::total(model, jim_bg_state, weights).grand_potential;

  std::cout << "\n  Jim FIRE converged = " << jim_cluster.converged
            << " (" << jim_cluster.iterations << " iters)\n";
  std::cout << "  Jim F (Helmholtz) = " << jim_cluster.energy << "\n";
  std::cout << "  Jim lambda = " << jim_cluster.lambda << "\n";
  std::cout << "  Jim mu_bg = " << jim_mu_bg << "\n";
  std::cout << "  Jim Omega* = " << jim_result.grand_potential << "\n";
  std::cout << "  Jim Omega_bg = " << jim_omega_bg << "\n";
  std::cout << "  Jim Delta_Omega/kT = " << (jim_result.grand_potential - jim_omega_bg) << "\n";
  std::cout << "  Jim background = " << jim_bg << "\n";
  std::cout << "  Jim rho_max = " << jim_cluster.density.max()
            << "  rho_min = " << jim_cluster.density.min() << "\n";
  std::cout << "  Jim mass = " << arma::accu(jim_cluster.density) * model.grid.cell_volume() << "\n";

  // Profile comparison
  {
    long nx = model.grid.shape[0];
    long iy = model.grid.shape[1] / 2;
    long iz = model.grid.shape[2] / 2;
    double centre = model.grid.box_size[0] / 2.0;
    arma::vec ours_rho = cluster.densities[0];
    arma::vec jims_rho = jim_cluster.density;
    double max_diff = 0;
    std::cout << "\n  Cluster profile comparison (x-slice):\n";
    std::cout << std::format("  {:>8s}  {:>18s}  {:>18s}  {:>14s}\n", "x", "ours", "jims", "diff");
    for (long ix = 0; ix < nx; ++ix) {
      auto idx = static_cast<arma::uword>(model.grid.flat_index(ix, iy, iz));
      double d = ours_rho(idx) - jims_rho(idx);
      max_diff = std::max(max_diff, std::abs(d));
      std::cout << std::format("  {:>8.2f}  {:>18.12f}  {:>18.12f}  {:>14.6e}\n",
          ix * dx - centre, ours_rho(idx), jims_rho(idx), d);
    }
    std::cout << "  max |diff| = " << max_diff << "\n";

    // EARLY STOP: if profiles differ significantly, stop and diagnose.
    if (max_diff > 1e-3) {
      std::cout << "\n  *** EARLY STOP: FIRE profiles differ by " << max_diff << " ***\n";
      std::cout << "  Fix FIRE algorithm mismatch before proceeding to eigenvalue/DDFT.\n";
      return 1;
    }
    std::cout << "\n  PASS: FIRE profiles match (max |diff| = " << max_diff << ")\n";
  }

  // ================================================================
  // Phase 2 & 3 (eigenvalue + DDFT) only if cluster is non-trivial
  // ================================================================

  double delta_rho = rho_l - rho_v;
  double R_eff = nucleation::effective_radius(
      cluster.densities[0], background, delta_rho, model.grid.cell_volume());
  std::cout << "  R_eff = " << R_eff << "\n";

  if (R_eff < 0.5 || cluster.densities[0].max() < 2.0 * background) {
    std::cout << "\n  WARNING: cluster evaporated or not resolved. Stopping here.\n";
    return 0;
  }

  std::cout << "\n  Cluster survived! Proceeding to eigenvalue and DDFT.\n";

  // ================================================================
  // Step 7: Functional evaluation comparison (component by component)
  // ================================================================

  std::cout << "\n================================================================\n";
  std::cout << "  Step 7: DFT functional evaluation comparison\n";
  std::cout << "================================================================\n\n";

  {
    // Evaluate at our cluster density with RSLT (Jim's model) and esFMT (our model).
    auto cs = init::from_profile(model, cluster.densities[0]);
    cs.species[0].chemical_potential = mu_bg;

    // Shared components (model-independent).
    auto c_id = functionals::ideal_gas(model.grid, cs);
    auto c_mf = functionals::mean_field(
        model.grid, cs, model.species, weights.mean_field);
    auto c_ext = functionals::external_field(model.grid, cs);

    // RSLT hard sphere (Jim's model — using the same weights already built).
    auto c_hs_rslt = functionals::hard_sphere(
        weights.fmt_model, model.grid, cs, model.species, weights.fmt);

    // esFMT hard sphere (our model — build separate weights).
    auto esfmt_model = functionals::fmt::EsFMT{.A = 1.0, .B = 0.0};
    auto esfmt_weights = functionals::make_weights(esfmt_model, model);
    auto c_hs_esfmt = functionals::hard_sphere(
        esfmt_weights.fmt_model, model.grid, cs, model.species, esfmt_weights.fmt);

    // Jim's ideal gas (beta units, no kT prefactor).
    const arma::vec& rho_c = cluster.densities[0];
    double dv = model.grid.cell_volume();
    arma::vec log_rho_c = arma::log(arma::clamp(rho_c, 1e-300, arma::datum::inf));
    double f_id_jim = arma::dot(rho_c, log_rho_c - 1.0) * dv;
    arma::vec force_id_jim = log_rho_c * dv;

    // Jim's chemical potential contribution (beta units).
    double mu_beta = mu_bg / kT;
    arma::vec force_ext_jim = arma::vec(rho_c.n_elem, arma::fill::value(-mu_beta * dv));

    // Total forces: RSLT path (must match Jim exactly).
    arma::vec force_rslt = c_id.forces[0] + c_hs_rslt.forces[0] + c_mf.forces[0] + c_ext.forces[0];
    arma::vec force_jim = force_id_jim + c_hs_rslt.forces[0] + c_mf.forces[0] + force_ext_jim;

    // Total forces: esFMT path (our model).
    arma::vec force_esfmt = c_id.forces[0] + c_hs_esfmt.forces[0] + c_mf.forces[0] + c_ext.forces[0];

    std::cout << std::setprecision(12);

    // 1. Validate our RSLT against Jim (must be zero diff).
    std::cout << "  a) RSLT vs Jim (code validation):\n";
    std::cout << "                        Ours              Jim's            diff\n";
    std::cout << "    F_id:    " << std::setw(18) << c_id.free_energy
              << std::setw(18) << f_id_jim << std::setw(14)
              << (c_id.free_energy - kT * f_id_jim) << " (ours - kT*jim)\n";
    std::cout << "    F_hs:    " << std::setw(18) << c_hs_rslt.free_energy
              << std::setw(18) << c_hs_rslt.free_energy << std::setw(14)
              << 0.0 << " (same code path)\n";
    std::cout << "    F_mf:    " << std::setw(18) << c_mf.free_energy
              << std::setw(18) << c_mf.free_energy << std::setw(14)
              << 0.0 << " (same code path)\n";

    double max_force_diff_rslt = arma::max(arma::abs(force_rslt - force_jim));
    std::cout << "    max|force_diff| = " << max_force_diff_rslt << "\n";
    if (max_force_diff_rslt < 1e-12) {
      std::cout << "    PASS: RSLT forces match Jim to machine precision.\n";
    }

    // 2. Compare RSLT vs esFMT (model difference).
    double dF_hs = c_hs_esfmt.free_energy - c_hs_rslt.free_energy;
    double max_force_diff_model = arma::max(arma::abs(c_hs_esfmt.forces[0] - c_hs_rslt.forces[0]));
    std::cout << "\n  b) esFMT vs RSLT (model comparison at same density):\n";
    std::cout << "    F_hs(esFMT)  = " << c_hs_esfmt.free_energy << "\n";
    std::cout << "    F_hs(RSLT)   = " << c_hs_rslt.free_energy << "\n";
    std::cout << "    diff         = " << dF_hs << "\n";
    std::cout << "    max|dF/drho diff| = " << max_force_diff_model << "\n";

    // 3. Our EsFMT vs Jim's esFMT — point-by-point Phi3 and derivative comparison.
    std::cout << "\n  c) Our EsFMT vs Jim's esFMT (formula-level comparison):\n";

    {
      // Compute weighted densities using esFMT weights (tensor weights needed).
      auto n_pts = static_cast<arma::uword>(model.grid.total_points());
      std::vector<long> shape(model.grid.shape.begin(), model.grid.shape.end());
      double R = 0.5 * model.species[0].hard_sphere_diameter;

      dft::math::FourierTransform rho_ft(shape);
      rho_ft.set_real(rho_c);
      rho_ft.forward();

      auto wd = dft::functionals::detail::convolve_weights(
          esfmt_weights.fmt.per_species[0], rho_ft.fourier(), shape);

      double max_phi3_diff = 0.0;
      double max_dphi3_dn2_diff = 0.0;
      double max_dphi3_dv1_diff = 0.0;
      double max_dphi3_dT_diff = 0.0;
      double max_phi_diff = 0.0;
      double jim_F_hs = 0.0;
      double our_F_hs = 0.0;

      auto esfmt = functionals::fmt::EsFMT{.A = 1.0, .B = 0.0};

      for (arma::uword idx = 0; idx < n_pts; ++idx) {
        auto m = dft::functionals::detail::assemble_measures(wd, idx, R);
        m.products = dft::functionals::fmt::inner_products(m);

        if (m.eta < 1e-30) continue;

        // Phi3
        double our_p3 = esfmt.phi3(m);
        auto fm = make_fm(m);
        double jim_p3 = legacy::fmt::esFMT_model::Phi3(1.0, 0.0, fm);
        max_phi3_diff = std::max(max_phi3_diff, std::abs(our_p3 - jim_p3));

        // dPhi3/dn2
        double our_dn2 = esfmt.d_phi3_d_n2(m);
        double jim_dn2 = legacy::fmt::esFMT_model::dPhi3_dS2(1.0, 0.0, fm);
        max_dphi3_dn2_diff = std::max(max_dphi3_dn2_diff, std::abs(our_dn2 - jim_dn2));

        // dPhi3/dv1
        auto our_dv = esfmt.d_phi3_d_v1(m);
        for (int k = 0; k < 3; ++k) {
          double jim_dv_k = legacy::fmt::esFMT_model::dPhi3_dV2(1.0, k, fm);
          max_dphi3_dv1_diff = std::max(max_dphi3_dv1_diff, std::abs(our_dv(k) - jim_dv_k));
        }

        // dPhi3/dT
        for (int i = 0; i < 3; ++i) {
          for (int j = i; j < 3; ++j) {
            double our_dT = esfmt.d_phi3_d_T(i, j, m);
            double jim_dT = legacy::fmt::esFMT_model::dPhi3_dT(1.0, 0.0, i, j, fm);
            max_dphi3_dT_diff = std::max(max_dphi3_dT_diff, std::abs(our_dT - jim_dT));
          }
        }

        // Full Phi (free energy density)
        double our_phi = dft::functionals::fmt::phi(
            dft::functionals::fmt::FMTModel{esfmt}, m);
        double jim_phi = legacy::fmt::phi(fm, 1.0, 0.0);
        max_phi_diff = std::max(max_phi_diff, std::abs(our_phi - jim_phi));

        our_F_hs += our_phi * dv;
        jim_F_hs += jim_phi * dv;
      }

      std::cout << "    max|Phi3 diff|       = " << max_phi3_diff << "\n";
      std::cout << "    max|dPhi3/dn2 diff|  = " << max_dphi3_dn2_diff << "\n";
      std::cout << "    max|dPhi3/dv1 diff|  = " << max_dphi3_dv1_diff << "\n";
      std::cout << "    max|dPhi3/dT diff|   = " << max_dphi3_dT_diff << "\n";
      std::cout << "    max|Phi diff|         = " << max_phi_diff << "\n";
      std::cout << "    F_hs(ours)  = " << our_F_hs << "\n";
      std::cout << "    F_hs(Jim)   = " << jim_F_hs << "\n";
      std::cout << "    F_hs diff   = " << (our_F_hs - jim_F_hs) << "\n";

      // Bulk EOS comparison at background density.
      double eta_bg = (std::numbers::pi / 6.0)
                    * model.species[0].hard_sphere_diameter
                    * model.species[0].hard_sphere_diameter
                    * model.species[0].hard_sphere_diameter * rho_out;
      double jim_fex = legacy::fmt::esFMT_model::fex(eta_bg, 1.0, 0.0);
      double our_fex_density = dft::functionals::fmt::free_energy_density(
          dft::functionals::fmt::FMTModel{esfmt}, rho_out,
          model.species[0].hard_sphere_diameter);
      double our_fex = our_fex_density / rho_out;
      std::cout << "\n    Bulk fex at eta=" << eta_bg << ":\n";
      std::cout << "      Ours  = " << our_fex << "\n";
      std::cout << "      Jim   = " << jim_fex << "\n";
      std::cout << "      diff  = " << (our_fex - jim_fex) << "\n";

      bool all_match = max_phi3_diff < 1e-14
                    && max_dphi3_dn2_diff < 1e-14
                    && max_dphi3_dv1_diff < 1e-14
                    && max_dphi3_dT_diff < 1e-14
                    && max_phi_diff < 1e-14;
      if (all_match) {
        std::cout << "    PASS: Our EsFMT matches Jim's esFMT to machine precision.\n";
      }
    }
  }


  // Phase 2: eigenvalue
  std::cout << "\n================================================================\n";
  std::cout << "  Phase 2: Smallest eigenvalue\n";
  std::cout << "================================================================\n\n";

  auto eig_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
    auto state = init::from_profile(model, rho);
    state.species[0].chemical_potential = mu_bg;
    auto result = functionals::total(model, state, weights);
    arma::vec grad = result.forces[0];
    grad = fixed_boundary(grad, bdry);
    return {result.grand_potential, grad};
  };

  auto eig_result = algorithms::smallest_eigenvalue(
      eig_force_fn, cluster.densities[0],
      {.tolerance = config::get<double>(cfg, "eigen.tolerance"),
       .max_iterations = config::get<int>(cfg, "eigen.max_iterations"),
       .hessian_eps = config::get<double>(cfg, "eigen.hessian_eps"),
       .log_interval = config::get<int>(cfg, "eigen.log_interval")});

  std::cout << "  Ours: eigenvalue = " << eig_result.eigenvalue
            << "  converged = " << eig_result.converged << "\n";

  // Jim's eigenvalue via FIRE2 on Rayleigh quotient
  std::cout << "\n  Jim's eigenvalue (FIRE2 on Rayleigh quotient):\n";

  arma::uvec jim_bdry = legacy::algorithms::boundary_mask_3d(
      model.grid.shape[0], model.grid.shape[1], model.grid.shape[2]);

  auto jim_eig_result = legacy::algorithms::eigenvalue_fire2(
      eig_force_fn, cluster.densities[0], jim_bdry,
      {.tolerance = config::get<double>(cfg, "eigen.tolerance"),
       .max_iterations = config::get<int>(cfg, "eigen.max_iterations"),
       .hessian_eps = config::get<double>(cfg, "eigen.hessian_eps"),
       .log_interval = config::get<int>(cfg, "eigen.log_interval")});

  std::cout << "  Jim:  eigenvalue = " << jim_eig_result.eigenvalue
            << "  converged = " << jim_eig_result.converged << "\n";
  std::cout << "  diff = " << (eig_result.eigenvalue - jim_eig_result.eigenvalue) << "\n";

  // Compare eigenvectors (account for sign ambiguity).
  double ev_dot = arma::dot(eig_result.eigenvector, jim_eig_result.eigenvector);
  double ev_sign = (ev_dot >= 0) ? 1.0 : -1.0;
  arma::vec ev_diff = eig_result.eigenvector - ev_sign * jim_eig_result.eigenvector;
  double max_ev_diff = arma::max(arma::abs(ev_diff));
  std::cout << "  eigenvector dot = " << ev_dot << " (sign=" << ev_sign << ")\n";
  std::cout << "  max |ev_diff| = " << max_ev_diff << "\n";

  if (std::abs(eig_result.eigenvalue - jim_eig_result.eigenvalue)
      / (1.0 + std::abs(eig_result.eigenvalue)) > 0.1) {
    std::cout << "\n  *** EARLY STOP: eigenvalue mismatch ***\n";
    return 1;
  }
  std::cout << "\n  Eigenvalue comparison done.\n";

  // ================================================================
  // Diagnostics: dump slices for comparison with Jim's code
  // ================================================================

  {
    long nx = model.grid.shape[0];
    long ny = model.grid.shape[1];
    long nz = model.grid.shape[2];
    double centre = model.grid.box_size[0] / 2.0;
    arma::vec rho_c = cluster.densities[0];
    arma::vec ev = eig_result.eigenvector;

    // 1. X-slice through centre: rho(x, Ny/2, Nz/2) and eigenvector
    {
      std::ofstream f("exports/x_slice.csv");
      f << "x,rho,eigvec\n" << std::setprecision(15);
      long iy = ny / 2;
      long iz = nz / 2;
      for (long ix = 0; ix < nx; ++ix) {
        auto idx = static_cast<arma::uword>(model.grid.flat_index(ix, iy, iz));
        f << (ix * dx - centre) << "," << rho_c(idx) << "," << ev(idx) << "\n";
      }
    }

    // 2. Y-slice through centre: rho(Nx/2, y, Nz/2)
    {
      std::ofstream f("exports/y_slice.csv");
      f << "y,rho,eigvec\n" << std::setprecision(15);
      long ix = nx / 2;
      long iz = nz / 2;
      for (long iy = 0; iy < ny; ++iy) {
        auto idx = static_cast<arma::uword>(model.grid.flat_index(ix, iy, iz));
        f << (iy * dx - centre) << "," << rho_c(idx) << "," << ev(idx) << "\n";
      }
    }

    // 3. 2D slice rho(x,y) at z=Nz/2
    {
      std::ofstream f("exports/xy_plane.csv");
      f << "ix,iy,x,y,rho,eigvec\n" << std::setprecision(15);
      long iz = nz / 2;
      for (long ix = 0; ix < nx; ++ix) {
        for (long iy = 0; iy < ny; ++iy) {
          auto idx = static_cast<arma::uword>(model.grid.flat_index(ix, iy, iz));
          f << ix << "," << iy << ","
            << (ix * dx - centre) << "," << (iy * dx - centre) << ","
            << rho_c(idx) << "," << ev(idx) << "\n";
        }
      }
    }

    // 4. Radial profile (azimuthal average)
    {
      std::ofstream f("exports/radial.csv");
      f << "r,rho,eigvec,count\n" << std::setprecision(15);
      long n_bins = static_cast<long>(centre / dx) + 1;
      arma::vec sum_rho(static_cast<arma::uword>(n_bins), arma::fill::zeros);
      arma::vec sum_ev(static_cast<arma::uword>(n_bins), arma::fill::zeros);
      arma::ivec cnt(static_cast<arma::uword>(n_bins), arma::fill::zeros);

      for (long ix = 0; ix < nx; ++ix) {
        double rx = ix * dx - centre;
        for (long iy = 0; iy < ny; ++iy) {
          double ry = iy * dx - centre;
          for (long iz2 = 0; iz2 < nz; ++iz2) {
            double rz = iz2 * dx - centre;
            double rad = std::sqrt(rx * rx + ry * ry + rz * rz);
            long bin = static_cast<long>(rad / dx);
            if (bin < n_bins) {
              auto b = static_cast<arma::uword>(bin);
              auto idx = static_cast<arma::uword>(model.grid.flat_index(ix, iy, iz2));
              sum_rho(b) += rho_c(idx);
              sum_ev(b) += ev(idx);
              cnt(b) += 1;
            }
          }
        }
      }
      for (arma::uword i = 0; i < static_cast<arma::uword>(n_bins); ++i) {
        if (cnt(i) > 0) {
          f << ((i + 0.5) * dx) << "," << (sum_rho(i) / cnt(i)) << ","
            << (sum_ev(i) / cnt(i)) << "," << cnt(i) << "\n";
        }
      }
    }

    // 5. Eigenvector statistics
    double ev_max = ev.max();
    double ev_min = ev.min();
    auto idx_max = ev.index_max();
    auto idx_min = ev.index_min();
    long ix_max = static_cast<long>(idx_max) / (ny * nz);
    long iy_max = (static_cast<long>(idx_max) / nz) % ny;
    long iz_max = static_cast<long>(idx_max) % nz;
    long ix_min = static_cast<long>(idx_min) / (ny * nz);
    long iy_min = (static_cast<long>(idx_min) / nz) % ny;
    long iz_min = static_cast<long>(idx_min) % nz;

    std::cout << "\n  Eigenvector diagnostics:\n";
    std::cout << "    ev range: [" << ev_min << ", " << ev_max << "]\n";
    std::cout << "    ev max at (" << ix_max << "," << iy_max << "," << iz_max << ")\n";
    std::cout << "    ev min at (" << ix_min << "," << iy_min << "," << iz_min << ")\n";
    std::cout << "    ev(centre) = " << ev(static_cast<arma::uword>(
        model.grid.flat_index(nx/2, ny/2, nz/2))) << "\n";
    std::cout << "    ev norm = " << arma::norm(ev) << "\n";
    std::cout << "    ev mass = " << arma::accu(ev) * model.grid.cell_volume() << "\n";

    // 6. Perturbed profile diagnostics
    double ps = config::get<double>(cfg, "ddft.perturb_scale");
    arma::vec rho_plus = rho_c + ps * ev;
    arma::vec rho_minus = rho_c - ps * ev;
    {
      std::ofstream f("exports/perturbed_x_slice.csv");
      f << "x,rho_cluster,rho_plus,rho_minus\n" << std::setprecision(15);
      long iy = ny / 2;
      long iz = nz / 2;
      for (long ix = 0; ix < nx; ++ix) {
        auto idx = static_cast<arma::uword>(model.grid.flat_index(ix, iy, iz));
        f << (ix * dx - centre) << "," << rho_c(idx) << ","
          << rho_plus(idx) << "," << rho_minus(idx) << "\n";
      }
    }
    std::cout << "    rho_cluster(centre) = " << rho_c(static_cast<arma::uword>(
        model.grid.flat_index(nx/2, ny/2, nz/2))) << "\n";

    // Mass of perturbation
    double mass_plus = arma::accu(rho_plus) * model.grid.cell_volume();
    double mass_minus = arma::accu(rho_minus) * model.grid.cell_volume();
    double mass_cluster = arma::accu(rho_c) * model.grid.cell_volume();
    std::cout << "    mass: cluster=" << mass_cluster << " plus=" << mass_plus
              << " minus=" << mass_minus << "\n";

    std::cout << "\n  CSV files written to exports/\n";
  }
  std::cout << "\n================================================================\n";
  std::cout << "  Phase 3: DDFT dynamics (step-by-step comparison)\n";
  std::cout << "================================================================\n\n";

  // Single-species force function for both our and Jim's DDFT.
  auto ddft_force_fn = [&](const std::vector<arma::vec>& densities)
      -> std::pair<double, std::vector<arma::vec>> {
    auto state = init::from_profile(model, densities[0]);
    state.species[0].chemical_potential = mu_bg;
    auto result = functionals::total(model, state, weights);
    return {result.grand_potential, result.forces};
  };

  auto jim_ddft_force = [&](const arma::vec& rho)
      -> std::pair<double, arma::vec> {
    auto state = init::from_profile(model, rho);
    state.species[0].chemical_potential = mu_bg;
    auto result = functionals::total(model, state, weights);
    return {result.grand_potential, result.forces[0]};
  };

  double perturb_scale = config::get<double>(cfg, "ddft.perturb_scale");
  double ddft_dt = config::get<double>(cfg, "ddft.dt");
  double ddft_dt_max = config::get<double>(cfg, "ddft.dt_max");
  double ddft_fp_tol = config::get<double>(cfg, "ddft.fp_tolerance");
  int ddft_fp_max_it = config::get<int>(cfg, "ddft.fp_max_iterations");
  int ddft_n_steps = config::get<int>(cfg, "ddft.n_steps");
  int ddft_log_interval = config::get<int>(cfg, "ddft.log_interval");
  arma::vec rho_cluster = cluster.densities[0];

  // Run dissolution only: perturb along -eigenvector.
  arma::vec rho_init = rho_cluster - perturb_scale * eig_result.eigenvector;
  rho_init = arma::clamp(rho_init, 1e-18, arma::datum::inf);

  // Step-by-step comparison: capped at 50 steps for speed.
  // The full dynamics run uses the simulate() function below.
  int comparison_steps = std::min(ddft_n_steps, 50);

  std::cout << "  Running DDFT comparison: our code vs Jim's code (" << comparison_steps << " steps)\n";
  std::cout << "  perturb_scale=" << perturb_scale
            << "  dt=" << ddft_dt
            << "  n_steps=" << comparison_steps << "\n\n";

  // Our DDFT state.
  auto our_ddft_st = algorithms::ddft::make_ddft_state(model.grid);
  algorithms::ddft::DdftConfig our_ddft_cfg{
      .dt = ddft_dt,
      .diffusion_coefficient = 1.0,
      .min_density = 1e-18,
      .dt_max = ddft_dt_max,
      .fp_tolerance = ddft_fp_tol,
      .fp_max_iterations = ddft_fp_max_it,
  };

  // Jim's DDFT state.
  auto jim_ddft_st = legacy::algorithms::make_ddft_state(
      model.grid.shape[0], model.grid.shape[1], model.grid.shape[2], model.grid.dx);
  double jim_dt = ddft_dt;

  // Track Delta_Omega = Omega - Omega_bg to see whether we are above or below
  // the saddle point.  Delta_Omega > 0 means above (critical cluster).
  // Delta_Omega < 0 means the droplet has crossed the barrier.

  // Start from same initial density.
  std::vector<arma::vec> our_rho = {rho_init};
  arma::vec jim_rho = rho_init;
  double our_time = 0.0;
  double jim_time = 0.0;
  int our_successes = 0;
  int jim_successes = 0;
  double max_ever_diff = 0.0;

  std::cout << std::format("  {:>6s}  {:>12s}  {:>14s}  {:>14s}  {:>12s}  {:>12s}\n",
      "step", "time", "dE_ours", "dE_jims", "max|drho|", "max|d_ij|");
  std::cout << "  " << std::string(84, '-') << "\n";

  for (int step = 1; step <= comparison_steps; ++step) {
    // Our step.
    double our_dt_before = our_ddft_cfg.dt;
    auto our_result = algorithms::ddft::integrating_factor_step(
        our_rho, model.grid, our_ddft_st, ddft_force_fn, our_ddft_cfg);
    our_rho = std::move(our_result.densities);
    our_time += our_result.dt_used;

    // Adaptive dt (our side): detect restarts via pre/post dt comparison
    // (integrating_factor_step mutates config.dt on restart).
    if (our_result.dt_used < our_dt_before) our_successes = 0;
    else ++our_successes;
    if (our_successes >= 5 && our_ddft_cfg.dt < ddft_dt_max) {
      our_ddft_cfg.dt = std::min(2.0 * our_ddft_cfg.dt, ddft_dt_max);
      our_successes = 0;
    }

    // Jim's step.
    double jim_dt_before = jim_dt;
    auto jim_result = legacy::algorithms::ddft_step(
        jim_rho, jim_ddft_st, jim_ddft_force,
        model.grid.cell_volume(), ddft_fp_tol, ddft_fp_max_it, jim_dt, ddft_dt_max);
    jim_rho = std::move(jim_result.density);
    jim_time += jim_result.dt_used;

    // Adaptive dt (Jim's side).
    bool jim_decreased = (jim_result.dt_used < jim_dt_before);
    if (jim_decreased) jim_successes = 0;
    else ++jim_successes;
    if (jim_successes >= 5 && jim_dt < ddft_dt_max) {
      jim_dt = std::min(2.0 * jim_dt, ddft_dt_max);
      jim_successes = 0;
    }

    // Compare our density vs Jim's density (should be ~0).
    double max_ij_diff = arma::max(arma::abs(our_rho[0] - jim_rho));
    max_ever_diff = std::max(max_ever_diff, max_ij_diff);

    // Track density change from the initial state (shows actual dynamics).
    double max_drho = arma::max(arma::abs(our_rho[0] - rho_init));

    // Delta_Omega = Omega - Omega_bg (positive = above saddle).
    double dE_ours = our_result.energy - omega_bg;
    double dE_jims = jim_result.energy - omega_bg;

    if (ddft_log_interval > 0 &&
        (step % 10 == 0 || step == comparison_steps || step <= 5)) {
      std::cout << std::format("  {:>6d}  {:>12.6f}  {:>14.6f}  {:>14.6f}  {:>12.4e}  {:>12.4e}\n",
          step, our_time, dE_ours, dE_jims, max_drho, max_ij_diff);
    }

    if (max_ij_diff > 1e-3) {
      std::cout << "\n  *** EARLY STOP: DDFT diverged at step " << step
                << ", max|diff|=" << max_ij_diff << " ***\n";
      break;
    }
  }

  std::cout << "\n  Max ever |ours-jims| across all steps: " << max_ever_diff << "\n";

  if (max_ever_diff < 1e-6) {
    std::cout << "  PASS: DDFT dynamics match perfectly.\n";
  } else if (max_ever_diff < 1e-3) {
    std::cout << "  PASS: DDFT dynamics match within tolerance.\n";
  } else {
    std::cout << "  FAIL: DDFT dynamics do not match.\n";
    return 1;
  }

  // Now run full dissolution and growth for plotting.
  // Display Delta_Omega = Omega - Omega_bg (positive = above saddle).
  algorithms::ddft::SimulationConfig sim_cfg{
      .ddft = {.dt = ddft_dt,
               .diffusion_coefficient = 1.0,
               .min_density = 1e-18,
               .dt_max = ddft_dt_max,
               .fp_tolerance = ddft_fp_tol,
               .fp_max_iterations = ddft_fp_max_it},
      .n_steps = ddft_n_steps,
      .snapshot_interval = config::get<int>(cfg, "ddft.snapshot_interval"),
      .log_interval = ddft_log_interval,
      .energy_offset = omega_bg,
  };

  std::cout << "\n  Dissolution (perturb along -eigenvector, Delta_Omega should decrease):\n";
  arma::vec rho_shrink = rho_cluster - perturb_scale * eig_result.eigenvector;
  rho_shrink = arma::clamp(rho_shrink, 1e-18, arma::datum::inf);
  auto sim_shrink = algorithms::ddft::simulate({rho_shrink}, model.grid, ddft_force_fn, sim_cfg);

  std::cout << "\n  Growth (perturb along +eigenvector, Delta_Omega should decrease):\n";
  arma::vec rho_grow = rho_cluster + perturb_scale * eig_result.eigenvector;
  rho_grow = arma::clamp(rho_grow, 1e-18, arma::datum::inf);
  auto sim_grow = algorithms::ddft::simulate({rho_grow}, model.grid, ddft_force_fn, sim_cfg);

#ifdef DFT_HAS_MATPLOTLIB
  std::cout << "\n  Generating plots...\n";
  plot::make_plots(
      nucleation::extract_x_slice(rho_cluster, model.grid),
      nucleation::extract_x_slice(rho0, model.grid),
      nucleation::extract_dynamics(sim_shrink, model.grid, r, background, delta_rho),
      nucleation::extract_dynamics(sim_grow, model.grid, r, background, delta_rho),
      {.radius = R_eff, .energy = cluster_result.grand_potential},
      omega_bg, rho_v, rho_l);
#endif

  std::cout << "\nDone.\n";
}
