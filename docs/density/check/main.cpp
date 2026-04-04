// check.cpp — Cross-validation of the DFT functional evaluation pipeline.
//
// Verifies that the full inhomogeneous DFT (FFT convolutions for FMT
// weighted densities and mean-field attraction, force back-convolution)
// correctly reproduces bulk thermodynamics when evaluated at uniform
// density. The bulk thermodynamics have already been cross-validated
// against Jim's library (check_thermodynamics, check_fmt).
//
// Steps:
//   1-4: F_total at uniform rho = 0.1, 0.3, 0.5, 0.7 vs f_bulk * V
//   5:   Zero force at equilibrium (uniform rho with mu = mu_bulk)
//   6:   Omega/V = -P at both coexistence densities

#include "dft.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

using namespace dft;

static int g_failures = 0;
static int g_checks = 0;

static void check_rel(
    std::string_view label, double got, double ref, double tol = 1e-8
) {
  ++g_checks;
  double scale = std::max(std::abs(ref), 1e-30);
  double rel = std::abs(got - ref) / scale;
  if (rel > tol) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": got=" << got << " ref=" << ref
              << " rel=" << rel << "\n";
  }
}

static void check(
    std::string_view label, double got, double ref, double tol = 1e-6
) {
  ++g_checks;
  double diff = std::abs(got - ref);
  if (diff > tol) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": got=" << got << " ref=" << ref
              << " diff=" << diff << "\n";
  }
}

static void section(std::string_view title) {
  std::cout << "\n" << title << "\n";
  std::cout << std::string(title.size(), '-') << "\n";
}

int main() {
  std::cout << std::setprecision(15);

  // ------------------------------------------------------------------
  // System: LJ sigma=1, epsilon=1, r_c=2.5, WCA, White Bear II.
  // Grid: dx=0.5, L=6 in all directions (12^3 = 1728 points).
  // Temperature: kT = 0.7.
  // ------------------------------------------------------------------

  constexpr double dx = 0.5;
  constexpr double L = 6.0;
  constexpr double kT = 0.7;

  physics::Model model{
      .grid = make_grid(dx, {L, L, L}),
      .species = {Species{.name = "LJ", .hard_sphere_diameter = 1.0}},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = physics::potentials::make_lennard_jones(1.0, 1.0, 2.5),
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      }},
      .temperature = kT,
  };

  auto fmt_model = functionals::fmt::WhiteBearII{};
  auto n_points = static_cast<arma::uword>(model.grid.total_points());
  double dv = model.grid.cell_volume();
  double V = dv * static_cast<double>(n_points);

  // Inhomogeneous weights: FFT-based, includes FMT weights + mean-field.
  auto weights = functionals::make_weights(fmt_model, model);

  // Bulk weights: use the GRID a_vdw so the comparison isolates the FFT
  // pipeline from cell quadrature accuracy. The analytical a_vdw has
  // already been cross-validated against Jim's code (check_potentials
  // and check_interaction); here we verify the assembly.
  auto bulk_weights = functionals::make_bulk_weights(
      fmt_model, model.interactions, kT
  );
  double a_vdw_analytical = bulk_weights.mean_field.interactions[0].a_vdw;
  double a_vdw_grid = weights.mean_field.interactions[0].a_vdw;
  bulk_weights.mean_field.interactions[0].a_vdw = a_vdw_grid;

  std::cout << "Grid: " << model.grid.shape[0] << "x" << model.grid.shape[1]
            << "x" << model.grid.shape[2] << " = " << n_points
            << " points, V = " << V << "\n";
  std::cout << "Temperature: kT = " << kT << "\n";
  std::cout << "a_vdw (analytical) = " << a_vdw_analytical << "\n";
  std::cout << "a_vdw (grid dx="  << dx << ") = " << a_vdw_grid
            << "  (ratio = " << a_vdw_grid / a_vdw_analytical << ")\n";

  // ------------------------------------------------------------------
  // Steps 1-4: F_total at uniform density.
  //
  // For uniform rho, the inhomogeneous DFT must reproduce bulk
  // thermodynamics: F_total = f_bulk(rho) * V. This tests the full
  // FFT convolution pipeline, FMT weighted density assembly, Phi
  // evaluation, and mean-field energy accumulation.
  // ------------------------------------------------------------------

  int step = 0;
  for (double rho : {0.1, 0.3, 0.5, 0.7}) {
    ++step;
    section("Step " + std::to_string(step) + ": F_total at uniform rho = "
            + std::to_string(rho));

    arma::vec density(n_points, arma::fill::value(rho));
    State state{
        .species = {{
            .density = {.values = density, .external_field = arma::zeros(n_points)},
            .force = arma::zeros(n_points),
            .chemical_potential = 0.0,
        }},
        .temperature = kT,
    };

    auto result = functionals::total(model, state, weights);

    // Break down individual contributions for diagnostics.
    auto c_id = functionals::ideal_gas(model.grid, state);
    auto c_hs = functionals::hard_sphere(
        weights.fmt_model, model.grid, state, model.species, weights.fmt
    );
    auto c_mf = functionals::mean_field(
        model.grid, state, model.species, weights.mean_field
    );

    double f_id_bulk = functionals::bulk::ideal_free_energy_density(arma::vec{rho});
    double f_hs_bulk = functionals::bulk::hard_sphere_free_energy_density(
        weights.fmt_model, arma::vec{rho}, model.species
    );
    double f_mf_bulk = functionals::bulk::mean_field_free_energy_density(
        bulk_weights.mean_field, arma::vec{rho}
    );

    double f_bulk = functionals::bulk::free_energy_density(
        arma::vec{rho}, model.species, bulk_weights
    );
    double F_ref = f_bulk * V;

    double rel = std::abs(result.free_energy - F_ref)
               / std::max(std::abs(F_ref), 1e-30);

    std::cout << "  F_inhomogeneous = " << result.free_energy << "\n";
    std::cout << "    F_id  = " << c_id.free_energy << "  (bulk: " << f_id_bulk * V << ")\n";
    std::cout << "    F_hs  = " << c_hs.free_energy << "  (bulk: " << f_hs_bulk * V << ")\n";
    std::cout << "    F_mf  = " << c_mf.free_energy << "  (bulk: " << f_mf_bulk * V << ")\n";
    std::cout << "  f_bulk * V      = " << F_ref << "\n";
    std::cout << "  relative diff   = " << rel << "\n";

    check_rel("F(rho=" + std::to_string(rho) + ")",
              result.free_energy, F_ref, 1e-6);
  }

  // ------------------------------------------------------------------
  // Step 5: Forces at uniform density with equilibrium mu.
  //
  // At uniform rho with mu = mu_bulk(rho), every grid point should
  // have zero force. This validates the derivative chain: FMT
  // back-convolution, mean-field force, and ideal gas force.
  // ------------------------------------------------------------------

  ++step;
  section("Step " + std::to_string(step) + ": Zero force at equilibrium");

  constexpr double rho_test = 0.4;
  double mu_test = functionals::bulk::chemical_potential(
      arma::vec{rho_test}, model.species, bulk_weights, 0
  );

  {
    arma::vec density(n_points, arma::fill::value(rho_test));
    State state{
        .species = {{
            .density = {.values = density, .external_field = arma::zeros(n_points)},
            .force = arma::zeros(n_points),
            .chemical_potential = mu_test,
        }},
        .temperature = kT,
    };

    auto result = functionals::total(model, state, weights);
    double max_force = arma::max(arma::abs(result.forces[0]));
    double mean_force = arma::mean(arma::abs(result.forces[0]));

    std::cout << "  rho = " << rho_test << ", mu = " << mu_test << "\n";
    std::cout << "  max|force|  = " << max_force << "\n";
    std::cout << "  mean|force| = " << mean_force << "\n";

    // Force = (delta beta Omega / delta rho) * dv.
    // At equilibrium, should vanish to machine precision.
    check("max|force|/dv", max_force / dv, 0.0, 1e-6);
  }

  // ------------------------------------------------------------------
  // Step 6: Grand potential at coexistence.
  //
  // At coexistence (P_v = P_l), both phases have Omega/V = -P.
  // Evaluating the inhomogeneous DFT at uniform rho_v and rho_l with
  // mu = mu_coex should give the same Omega/V = -P_coex.
  // ------------------------------------------------------------------

  ++step;
  section("Step " + std::to_string(step) + ": Omega at coexistence");

  auto coex = functionals::bulk::find_coexistence(
      model.species, bulk_weights,
      {.rho_max = 1.0, .rho_scan_step = 0.005,
       .newton = {.max_iterations = 300, .tolerance = 1e-10}}
  );

  if (!coex) {
    std::cout << "  FAIL: coexistence not found\n";
    g_failures += 2;
    g_checks += 2;
  } else {
    double mu_coex = functionals::bulk::chemical_potential(
        arma::vec{coex->rho_vapor}, model.species, bulk_weights, 0
    );
    double P_coex = functionals::bulk::pressure(
        arma::vec{coex->rho_vapor}, model.species, bulk_weights
    );
    std::cout << "  rho_v=" << coex->rho_vapor << " rho_l=" << coex->rho_liquid
              << " mu=" << mu_coex << " P=" << P_coex << "\n";

    // Vapor
    arma::vec rho_v(n_points, arma::fill::value(coex->rho_vapor));
    State state_v{
        .species = {{
            .density = {.values = rho_v, .external_field = arma::zeros(n_points)},
            .force = arma::zeros(n_points),
            .chemical_potential = mu_coex,
        }},
        .temperature = kT,
    };
    auto res_v = functionals::total(model, state_v, weights);

    // Liquid
    arma::vec rho_l(n_points, arma::fill::value(coex->rho_liquid));
    State state_l{
        .species = {{
            .density = {.values = rho_l, .external_field = arma::zeros(n_points)},
            .force = arma::zeros(n_points),
            .chemical_potential = mu_coex,
        }},
        .temperature = kT,
    };
    auto res_l = functionals::total(model, state_l, weights);

    std::cout << "  Omega_v/V = " << res_v.grand_potential / V
              << "  (-P_coex = " << -P_coex << ")\n";
    std::cout << "  Omega_l/V = " << res_l.grand_potential / V << "\n";

    check_rel("Omega_v/V = -P", res_v.grand_potential / V, -P_coex, 1e-6);
    check_rel("Omega_l/V = -P", res_l.grand_potential / V, -P_coex, 1e-6);
  }

  // ------------------------------------------------------------------
  // Summary
  // ------------------------------------------------------------------

  std::cout << "\n========================================\n";
  std::cout << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures > 0 ? 1 : 0;
}
