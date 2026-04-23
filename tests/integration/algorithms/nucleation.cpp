// Cross-validation of FIRE, eigenvalue, and DDFT algorithms against
// legacy code. Uses a nucleation model (32^3 LJ, kT=0.8) as the test
// case. All three algorithms share a lazily-initialized fixture that is
// computed once on first access.

#include "dft.hpp"
#include "legacy/algorithms.hpp"
#include "legacy/classicaldft.hpp"

#include "dft/algorithms/minimization.hpp"
#include "dft/algorithms/saddle_point.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;
using Catch::Approx;

namespace {

  // Nucleation model parameters (from docs/nucleation/config.ini).

  constexpr double SIGMA = 1.0;
  constexpr double EPSILON = 1.0;
  constexpr double RCUT = 3.0;
  constexpr double KT = 0.8;
  constexpr double DX = 0.4;
  constexpr double BOX_LENGTH = 12.8;
  constexpr double RHO_IN = 0.75;
  constexpr double RHO_OUT = 0.0095;
  constexpr double R0 = 3.0;

  // Radial distances from the box centre (for step-function initial
  // condition).

  auto radial_distances(const Grid& grid) -> arma::vec {
    auto n = static_cast<arma::uword>(grid.total_points());
    arma::vec r2(n, arma::fill::zeros);

    for (int d = 0; d < 3; ++d) {
      double centre = grid.box_size[d] / 2.0;
      long nd = grid.shape[d];
      arma::vec coord(static_cast<arma::uword>(nd));
      for (long i = 0; i < nd; ++i) {
        coord(static_cast<arma::uword>(i)) = i * grid.dx - centre;
      }

      arma::uword stride = 1;
      for (int dd = d + 1; dd < 3; ++dd) {
        stride *= static_cast<arma::uword>(grid.shape[dd]);
      }
      arma::uword repeat = n / (static_cast<arma::uword>(nd) * stride);

      for (arma::uword rep = 0; rep < repeat; ++rep) {
        for (arma::uword i = 0; i < static_cast<arma::uword>(nd); ++i) {
          arma::uword offset = rep * static_cast<arma::uword>(nd) * stride + i * stride;
          r2.subvec(offset, offset + stride - 1) += coord(i) * coord(i);
        }
      }
    }
    return arma::sqrt(r2);
  }

  auto step_function(const arma::vec& r, double R, double rho_in, double rho_out) -> arma::vec {
    arma::vec rho(r.n_elem, arma::fill::value(rho_out));
    rho.elem(arma::find(r < R)).fill(rho_in);
    return rho;
  }

  // Complete test fixture: model + FIRE + eigenvalue + DDFT results.

  struct Fixture {
    physics::Model model;
    functionals::Weights weights;
    functionals::Weights bulk_weights;
    double mu_bg;
    double omega_bg;
    arma::uvec boundary;
    arma::vec rho0;
    double target_mass;

    arma::vec our_cluster;
    arma::vec ref_cluster;
    double fire_max_diff;
    bool our_fire_converged;
    bool ref_fire_converged;
    double background;

    double our_eigenvalue;
    double ref_eigenvalue;
    arma::vec our_eigenvector;
    arma::vec ref_eigenvector;
    bool our_eigen_converged;
    bool ref_eigen_converged;

    double ddft_max_ever_diff;
    int ddft_comparison_steps;
  };

  auto fixture() -> const Fixture& {
    static auto f = []() -> Fixture {
      Fixture fx;

      // Seed random number generator for reproducible eigenvalue init.
      arma::arma_rng::set_seed(42);

      // Model setup.

      auto lj = physics::potentials::make_lennard_jones(SIGMA, EPSILON, RCUT);
      double hsd = lj.hard_sphere_diameter(KT, physics::potentials::SplitScheme::WeeksChandlerAndersen);

      fx.model = physics::Model{
          .grid = make_grid(DX, {BOX_LENGTH, BOX_LENGTH, BOX_LENGTH}),
          .species = {Species{.name = "LJ", .hard_sphere_diameter = hsd}},
          .interactions = {{
              .species_i = 0,
              .species_j = 0,
              .potential = lj,
              .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
              .weight_scheme = physics::WeightScheme::InterpolationQuadraticF,
          }},
          .temperature = KT,
      };

      // Bulk weights with grid a_vdw (matching legacy approach).

      fx.weights = functionals::make_weights(functionals::fmt::RSLT{}, fx.model);
      fx.bulk_weights = functionals::make_bulk_weights(functionals::fmt::RSLT{}, fx.model.interactions, KT);
      fx.bulk_weights.mean_field.interactions[0].a_vdw = fx.weights.mean_field.interactions[0].a_vdw;

      auto eos = functionals::bulk::make_bulk_thermodynamics(fx.model.species, fx.bulk_weights);

      // Coexistence.

      auto coex =
          functionals::bulk::PhaseSearch{
              .rho_max = 1.0,
              .rho_scan_step = 0.005,
              .newton = {.max_iterations = 300, .tolerance = 1e-10},
          }
              .find_coexistence(eos);

      double rho_v = coex ? coex->rho_vapor : RHO_OUT;
      double rho_l = coex ? coex->rho_liquid : RHO_IN;
      double rho_out_used = RHO_OUT;

      // Adjust supersaturation if needed.
      double S = rho_out_used / rho_v;
      if (S < 1.0 && coex) {
        rho_out_used = 1.1 * rho_v;
      }

      double mu_rho_out = eos.chemical_potential(arma::vec{rho_out_used}, 0);

      // Initial condition.

      auto r = radial_distances(fx.model.grid);
      fx.rho0 = step_function(r, R0, rho_l, rho_out_used);
      fx.target_mass = arma::accu(fx.rho0) * fx.model.grid.cell_volume();
      fx.boundary = fx.model.grid.boundary_mask();

      // FIRE minimization: ours.

      auto cluster =
          algorithms::minimization::Minimizer{
              .fire =
                  {.dt = 0.1,
                   .dt_max = 1.0,
                   .alpha_start = 0.01,
                   .f_alpha = 0.99,
                   .force_tolerance = 1e-6,
                   .max_steps = 200000},
              .param = algorithms::minimization::Unbounded{.rho_min = 1e-99},
              .use_homogeneous_boundary = true,
              .log_interval = 0,
          }
              .fixed_mass(fx.model, fx.weights, fx.rho0, mu_rho_out, fx.target_mass);

      fx.our_cluster = cluster.densities[0];
      fx.our_fire_converged = cluster.converged;

      // FIRE minimization: legacy.

      auto ref_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
        auto state = init::from_profile(fx.model, rho);
        state.species[0].chemical_potential = 0.0;
        auto result = functionals::total(fx.model, state, fx.weights);
        return {result.free_energy, result.forces[0]};
      };

      long nx = fx.model.grid.shape[0];
      long ny = fx.model.grid.shape[1];
      long nz = fx.model.grid.shape[2];

      auto ref_cluster = legacy::algorithms::fire_minimize_fixed_mass(
          ref_force_fn,
          fx.rho0,
          fx.target_mass,
          fx.model.grid.cell_volume(),
          nx,
          ny,
          nz,
          {.dt = 0.1,
           .dt_max = 1.0,
           .alpha_start = 0.01,
           .alpha_fac = 0.99,
           .force_tolerance = 1e-6,
           .max_steps = 200000,
           .log_interval = 0}
      );

      fx.ref_cluster = ref_cluster.density;
      fx.ref_fire_converged = ref_cluster.converged;

      // Profile comparison.

      fx.fire_max_diff = arma::max(arma::abs(fx.our_cluster - fx.ref_cluster));

      // Background density.

      fx.background = 0.0;
      arma::uword n_bdry = 0;
      for (arma::uword i = 0; i < fx.our_cluster.n_elem; ++i) {
        if (fx.boundary(i)) {
          fx.background += fx.our_cluster(i);
          n_bdry++;
        }
      }
      if (n_bdry > 0)
        fx.background /= static_cast<double>(n_bdry);

      fx.mu_bg = eos.chemical_potential(arma::vec{fx.background}, 0);

      auto bg_state = init::from_profile(fx.model, arma::vec(fx.rho0.n_elem, arma::fill::value(fx.background)));
      bg_state.species[0].chemical_potential = fx.mu_bg;
      fx.omega_bg = functionals::total(fx.model, bg_state, fx.weights).grand_potential;

      // Eigenvalue: ours.

      auto eig_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
        auto state = init::from_profile(fx.model, rho);
        state.species[0].chemical_potential = fx.mu_bg;
        auto result = functionals::total(fx.model, state, fx.weights);
        arma::vec grad = result.forces[0];
        grad = fixed_boundary(grad, fx.boundary);
        return {result.grand_potential, grad};
      };

      auto our_eig =
          algorithms::saddle_point::EigenvalueSolver{
              .tolerance = 1e-4,
              .max_iterations = 300,
              .hessian_eps = 1e-6,
              .log_interval = 0,
              .boundary_mask = fx.boundary,
          }
              .solve(eig_force_fn, fx.our_cluster);

      fx.our_eigenvalue = our_eig.eigenvalue;
      fx.our_eigenvector = our_eig.eigenvector;
      fx.our_eigen_converged = our_eig.converged;

      // Eigenvalue: legacy.

      arma::uvec ref_bdry = legacy::algorithms::boundary_mask_3d(nx, ny, nz);

      auto ref_eig = legacy::algorithms::eigenvalue_fire2(
          eig_force_fn,
          fx.our_cluster,
          ref_bdry,
          {.tolerance = 1e-4, .max_iterations = 300, .hessian_eps = 1e-6, .log_interval = 0}
      );

      fx.ref_eigenvalue = ref_eig.eigenvalue;
      fx.ref_eigenvector = ref_eig.eigenvector;
      fx.ref_eigen_converged = ref_eig.converged;

      // DDFT step-by-step comparison (50 steps).

      auto ddft_force_fn = [&](const std::vector<arma::vec>& densities) -> std::pair<double, std::vector<arma::vec>> {
        auto state = init::from_profile(fx.model, densities[0]);
        state.species[0].chemical_potential = fx.mu_bg;
        auto result = functionals::total(fx.model, state, fx.weights);
        return {result.grand_potential, result.forces};
      };

      auto ref_ddft_force = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
        auto state = init::from_profile(fx.model, rho);
        state.species[0].chemical_potential = fx.mu_bg;
        auto result = functionals::total(fx.model, state, fx.weights);
        return {result.grand_potential, result.forces[0]};
      };

      double ddft_dt = 1e-2;
      double ddft_dt_max = 1.0;
      double ddft_fp_tol = 1e-4;
      int ddft_fp_max_it = 100;
      double perturb_scale = 1.0;
      fx.ddft_comparison_steps = 20;

      arma::vec rho_init = fx.our_cluster - perturb_scale * fx.our_eigenvector;
      rho_init = arma::clamp(rho_init, 1e-18, arma::datum::inf);

      // Our DDFT state.
      auto our_ddft_st = algorithms::dynamics::detail::make_if_state(fx.model.grid);
      algorithms::dynamics::StepConfig our_ddft_cfg{
          .dt = ddft_dt,
          .diffusion_coefficient = 1.0,
          .min_density = 1e-18,
          .dt_max = ddft_dt_max,
          .fp_tolerance = ddft_fp_tol,
          .fp_max_iterations = ddft_fp_max_it
      };

      // Legacy DDFT state.
      auto ref_ddft_st = legacy::algorithms::make_ddft_state(nx, ny, nz, DX);
      double ref_dt = ddft_dt;

      std::vector<arma::vec> our_rho = {rho_init};
      arma::vec ref_rho = rho_init;
      int our_successes = 0;
      int ref_successes = 0;
      fx.ddft_max_ever_diff = 0.0;

      for (int step = 1; step <= fx.ddft_comparison_steps; ++step) {
        double our_dt_before = our_ddft_cfg.dt;
        auto our_result = algorithms::dynamics::integrating_factor_step(
            our_rho,
            fx.model.grid,
            our_ddft_st,
            ddft_force_fn,
            our_ddft_cfg
        );
        our_rho = std::move(our_result.densities);

        if (our_result.dt_used < our_dt_before)
          our_successes = 0;
        else
          ++our_successes;
        if (our_successes >= 5 && our_ddft_cfg.dt < ddft_dt_max) {
          our_ddft_cfg.dt = std::min(2.0 * our_ddft_cfg.dt, ddft_dt_max);
          our_successes = 0;
        }

        double ref_dt_before = ref_dt;
        auto ref_result = legacy::algorithms::ddft_step(
            ref_rho,
            ref_ddft_st,
            ref_ddft_force,
            fx.model.grid.cell_volume(),
            ddft_fp_tol,
            ddft_fp_max_it,
            ref_dt,
            ddft_dt_max
        );
        ref_rho = std::move(ref_result.density);

        bool ref_decreased = (ref_result.dt_used < ref_dt_before);
        if (ref_decreased)
          ref_successes = 0;
        else
          ++ref_successes;
        if (ref_successes >= 5 && ref_dt < ddft_dt_max) {
          ref_dt = std::min(2.0 * ref_dt, ddft_dt_max);
          ref_successes = 0;
        }

        double max_diff = arma::max(arma::abs(our_rho[0] - ref_rho));
        fx.ddft_max_ever_diff = std::max(fx.ddft_max_ever_diff, max_diff);
      }

      return fx;
    }();
    return f;
  }

} // anonymous namespace

TEST_CASE("FIRE minimization matches legacy", "[integration][algorithms]") {
  auto& fx = fixture();
  REQUIRE(fx.our_fire_converged);
  REQUIRE(fx.ref_fire_converged);
  CHECK(fx.fire_max_diff < 1e-12);
}

TEST_CASE("Eigenvalue matches legacy", "[integration][algorithms]") {
  auto& fx = fixture();
  REQUIRE(fx.our_eigen_converged);
  REQUIRE(fx.ref_eigen_converged);

  double rel_diff = std::abs(fx.our_eigenvalue - fx.ref_eigenvalue) / (1.0 + std::abs(fx.our_eigenvalue));
  CHECK(rel_diff < 0.1);

  double dot = arma::dot(fx.our_eigenvector, fx.ref_eigenvector);
  CHECK(std::abs(dot) > 0.9);
}

TEST_CASE("DDFT dynamics match legacy step-by-step", "[integration][algorithms]") {
  auto& fx = fixture();
  INFO("max |ours - ref| across " << fx.ddft_comparison_steps << " steps = " << fx.ddft_max_ever_diff);
  CHECK(fx.ddft_max_ever_diff < 1e-3);
}
