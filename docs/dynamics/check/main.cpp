// check.cpp — Cross-validation of DDFT integrating-factor step.
//
// Both our code and Jim's DDFT.cpp implement the same
// integrating-factor scheme with FWD1 divergence stencil (verified
// by code inspection). This check validates:
//   1. Propagator coefficients U0 match exp(Lambda * dt)
//   2. Pure diffusion: ideal gas step gives rho_k^1 = exp(Lam*dt) * rho_k^0
//   3. Mass conservation through one full DFT step
//   4. Grand potential decreases through one full DFT step

#include <cmath>
#include <dftlib>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <string_view>

using namespace dft;

static int g_failures = 0;
static int g_checks = 0;

static void check(std::string_view label, double got, double ref, double tol = 1e-10) {
  ++g_checks;
  double diff = std::abs(got - ref);
  if (diff > tol) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": got=" << got << " ref=" << ref << " diff=" << diff << "\n";
  }
}

static void check_rel(std::string_view label, double got, double ref, double tol = 1e-8) {
  ++g_checks;
  double scale = std::max(std::abs(ref), 1e-30);
  double rel = std::abs(got - ref) / scale;
  if (rel > tol) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": got=" << got << " ref=" << ref << " rel=" << rel << "\n";
  }
}

static void section(std::string_view title) {
  std::cout << "\n" << title << "\n";
  std::cout << std::string(title.size(), '-') << "\n";
}

int main() {
  std::cout << std::setprecision(15);

  // ------------------------------------------------------------------
  // System: small grid for fast testing.
  // ------------------------------------------------------------------

  constexpr double dx = 0.5;
  Grid grid{.dx = dx, .box_size = {4.0, 4.0, 4.0}, .shape = {8, 8, 8}};
  long n = grid.total_points();
  double dv = grid.cell_volume();
  constexpr double D = 1.0;
  constexpr double dt = 1e-4;

  // ------------------------------------------------------------------
  // Step 1: Propagator coefficients
  //
  // The integrating factor exp(Lambda_k * D * dt) where Lambda_k are
  // the discrete Laplacian eigenvalues. Verify at several k-values.
  // ------------------------------------------------------------------

  section("Step 1: Propagator coefficients");

  auto ddft_st = algorithms::dynamics::_internal::make_if_state(grid);
  algorithms::dynamics::_internal::update_timestep(ddft_st, D * dt);

  // Lambda at ix=0: cos(0)-1 = 0 → exp(0) = 1
  check("fx[0]", ddft_st.fx(0), 1.0);
  check("fy[0]", ddft_st.fy(0), 1.0);
  check("fz[0]", ddft_st.fz(0), 1.0);

  // Lambda at ix=1: 2/dx² * (cos(2pi/8)-1) = 2/0.25*(cos(pi/4)-1)
  //               = 8 * (0.70711 - 1) = 8 * (-0.29289) = -2.34315
  double lam1 = 2.0 / (dx * dx) * (std::cos(2.0 * std::numbers::pi / 8.0) - 1.0);
  check("fx[1]", ddft_st.fx(1), std::exp(lam1 * D * dt), 1e-14);

  // Lambda at ix=4 (Nyquist): 2/dx² * (cos(pi)-1) = 8*(-2) = -16
  double lam4 = 2.0 / (dx * dx) * (std::cos(2.0 * std::numbers::pi * 4.0 / 8.0) - 1.0);
  check("fx[4]", ddft_st.fx(4), std::exp(lam4 * D * dt), 1e-14);

  std::cout << "  lam[1] = " << lam1 << " → exp = " << std::exp(lam1 * D * dt) << "\n";
  std::cout << "  lam[4] = " << lam4 << " → exp = " << std::exp(lam4 * D * dt) << "\n";

  // ------------------------------------------------------------------
  // Step 2: Pure diffusion (ideal gas only)
  //
  // For ideal gas forces only, R[rho] = 0 and the step reduces to
  // rho_k^1 = exp(Lambda_k * D * dt) * rho_k^0.
  // Start from rho(x) = rho0 + A*cos(2*pi*x/Lx) on a 1D slice.
  // ------------------------------------------------------------------

  section("Step 2: Pure diffusion step");

  double rho0 = 0.5;
  double A = 0.01;
  arma::vec density(static_cast<arma::uword>(n), arma::fill::value(rho0));

  // Add a cosine perturbation along x.
  for (long ix = 0; ix < grid.shape[0]; ++ix) {
    double x = ix * dx;
    double perturbation = A * std::cos(2.0 * std::numbers::pi * x / grid.box_size[0]);
    for (long iy = 0; iy < grid.shape[1]; ++iy) {
      for (long iz = 0; iz < grid.shape[2]; ++iz) {
        long idx = grid.flat_index(ix, iy, iz);
        density(static_cast<arma::uword>(idx)) += perturbation;
      }
    }
  }

  double mass_before = arma::accu(density) * dv;

  // Force callback: ideal gas only.
  // beta F_id = sum rho*(ln(rho)-1)*dv, force_i = ln(rho_i)*dv.
  auto ideal_force = [&](const std::vector<arma::vec>& densities) -> std::pair<double, std::vector<arma::vec>> {
    const auto& rho = densities[0];
    arma::vec log_rho = arma::log(arma::clamp(rho, 1e-300, arma::datum::inf));
    double energy = arma::dot(rho, log_rho - 1.0) * dv;
    return {energy, {log_rho * dv}};
  };

  algorithms::dynamics::StepConfig ddft_cfg{
      .dt = dt,
      .diffusion_coefficient = D,
  };

  auto result = algorithms::dynamics::integrating_factor_step({density}, grid, ddft_st, ideal_force, ddft_cfg);

  double mass_after = arma::accu(result.densities[0]) * dv;

  std::cout << "  mass before = " << mass_before << "\n";
  std::cout << "  mass after  = " << mass_after << "\n";

  check_rel("mass conservation (ideal)", mass_after, mass_before, 1e-10);

  // The perturbation amplitude should decay by exp(Lambda_1 * D * dt).
  // The ix=1 mode has Lambda = lam1 from above.
  double expected_decay = std::exp(lam1 * D * dt);
  double mean_after = arma::mean(result.densities[0]);

  // Extract the ix=1 Fourier mode amplitude.
  arma::vec slice_before(static_cast<arma::uword>(grid.shape[0]));
  arma::vec slice_after(static_cast<arma::uword>(grid.shape[0]));
  for (long ix = 0; ix < grid.shape[0]; ++ix) {
    slice_before(static_cast<arma::uword>(ix)) = density(static_cast<arma::uword>(grid.flat_index(ix, 0, 0)));
    slice_after(static_cast<arma::uword>(ix)) = result.densities[0](static_cast<arma::uword>(grid.flat_index(ix, 0, 0))
    );
  }
  double A_before = arma::max(slice_before) - arma::min(slice_before);
  double A_after = arma::max(slice_after) - arma::min(slice_after);
  double decay = A_after / A_before;
  std::cout << "  amplitude: " << A_before << " → " << A_after << " (decay = " << decay
            << ", expected = " << expected_decay << ")\n";
  check_rel("diffusion decay rate", decay, expected_decay, 1e-4);

  // ------------------------------------------------------------------
  // Step 3: Full DFT step — mass conservation and Omega decrease
  //
  // Take one DDFT step with full DFT forces (FMT + mean-field) on a
  // tanh slab profile at kT=0.7. Mass must be conserved and Omega
  // must not increase.
  // ------------------------------------------------------------------

  section("Step 3: Full DFT step properties");

  constexpr double kT = 0.7;
  physics::Model model{
      .grid = make_grid(0.25, {5.0, 2.5, 2.5}),
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
  auto weights = functionals::make_weights(fmt_model, model);
  auto bulk_wt = functionals::make_bulk_weights(fmt_model, model.interactions, kT);

  auto eos = functionals::bulk::make_bulk_thermodynamics(model.species, bulk_wt);

  auto coex =
      functionals::bulk::PhaseSearch{
          .rho_max = 1.0,
          .rho_scan_step = 0.005,
          .newton = {.max_iterations = 300, .tolerance = 1e-10},
      }
          .find_coexistence(eos);

  if (!coex) {
    std::cout << "  SKIP: coexistence not found\n";
  } else {
    double mu_coex = eos.chemical_potential(arma::vec{coex->rho_vapor}, 0);

    // Tanh slab profile.
    long nx = model.grid.shape[0];
    long ny = model.grid.shape[1];
    long nz = model.grid.shape[2];
    long n_full = model.grid.total_points();
    arma::vec rho_slab(static_cast<arma::uword>(n_full));
    double Lx = model.grid.box_size[0];
    double center = Lx / 2;
    double width = Lx / 4;
    double alpha = 1.0;
    for (long ix = 0; ix < nx; ++ix) {
      double x = ix * model.grid.dx;
      double profile = coex->rho_vapor
          + 0.5 * (coex->rho_liquid - coex->rho_vapor)
              * (std::tanh((x - center + width) / alpha) - std::tanh((x - center - width) / alpha));
      for (long iy = 0; iy < ny; ++iy) {
        for (long iz = 0; iz < nz; ++iz) {
          long idx = model.grid.flat_index(ix, iy, iz);
          rho_slab(static_cast<arma::uword>(idx)) = profile;
        }
      }
    }

    double dv_full = model.grid.cell_volume();
    double mass_0 = arma::accu(rho_slab) * dv_full;

    // Force callback wrapping functionals::total.
    auto force_fn = [&](const std::vector<arma::vec>& densities) -> std::pair<double, std::vector<arma::vec>> {
      State state{
          .species = {{
              .density = {.values = densities[0], .external_field = arma::zeros(static_cast<arma::uword>(n_full))},
              .force = arma::zeros(static_cast<arma::uword>(n_full)),
              .chemical_potential = mu_coex,
          }},
          .temperature = kT,
      };
      auto res = functionals::total(model, state, weights);
      return {res.grand_potential, std::move(res.forces)};
    };

    auto [omega_0, forces_0] = force_fn({rho_slab});

    auto st = algorithms::dynamics::_internal::make_if_state(model.grid);
    algorithms::dynamics::StepConfig cfg{
        .dt = 1e-4,
        .diffusion_coefficient = 1.0,
    };
    auto step_result = algorithms::dynamics::integrating_factor_step({rho_slab}, model.grid, st, force_fn, cfg);

    double mass_1 = arma::accu(step_result.densities[0]) * dv_full;
    auto [omega_1, forces_1] = force_fn(step_result.densities);

    std::cout << "  mass: " << mass_0 << " → " << mass_1 << " (rel diff = " << std::abs(mass_1 - mass_0) / mass_0
              << ")\n";
    std::cout << "  Omega: " << omega_0 << " → " << omega_1 << " (diff = " << omega_1 - omega_0 << ")\n";

    check_rel("mass conservation (DFT)", mass_1, mass_0, 1e-6);

    // Omega should decrease (or stay the same) for a dissipative DDFT step.
    ++g_checks;
    if (omega_1 > omega_0 + 1e-8 * std::abs(omega_0)) {
      ++g_failures;
      std::cout << "  FAIL Omega increased: " << omega_0 << " → " << omega_1 << "\n";
    }
  }

  // ------------------------------------------------------------------
  // Summary
  // ------------------------------------------------------------------

  std::cout << "\n========================================\n";
  std::cout << g_checks << " checks, " << g_failures << " failures\n";
  return g_failures > 0 ? 1 : 0;
}
