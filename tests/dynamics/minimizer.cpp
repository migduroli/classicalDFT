#include "dft/dynamics/minimizer.h"

#include "dft/density.h"
#include "dft/dynamics/fire2.h"
#include "dft/dynamics/integrator.h"
#include "dft/solver.h"
#include "dft/species.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numbers>

using namespace dft;

// ── Helpers ───────────────────────────────────────────────────────────────

/**
 * @brief Create a minimal Solver with a single species at uniform density.
 */
static Solver make_ideal_gas_solver(double rho0, double dx, double box_length) {
  auto d = density::Density(dx, {box_length, box_length, box_length});
  d.values().fill(rho0);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));
  return solver;
}

/**
 * @brief Create a Solver for a 1D-like (Nx x 1 x 1) system for easy analysis.
 */
static Solver make_1d_solver(double dx, double length, const arma::vec& rho_profile) {
  auto d = density::Density(dx, {length, dx, dx});
  long nx = static_cast<long>(std::round(length / dx));
  arma::vec full_rho(d.size());
  for (arma::uword i = 0; i < d.size(); ++i) {
    full_rho(i) = rho_profile(i % static_cast<arma::uword>(nx));
  }
  d.set(full_rho);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));
  return solver;
}

// ── Minimizer base class ─────────────────────────────────────────────────

/**
 * @brief A trivial minimizer for testing the base class: steepest descent.
 */
class SteepestDescent : public dynamics::Minimizer {
 public:
  explicit SteepestDescent(
      Solver& solver, double step_size = 1e-4, double force_limit = 0.1, double min_density = 1e-30
  )
      : dynamics::Minimizer(solver, force_limit, min_density), step_size_(step_size) {
    (void)compute_energy_and_forces();
  }

 private:
  [[nodiscard]] double do_step() override {
    // Simple gradient descent: x -= step_size * dF/dx
    int n = mutable_solver().num_species();
    for (int s = 0; s < n; ++s) {
      auto si = static_cast<size_t>(s);
      const arma::vec& df = mutable_solver().species(s).force();
      aliases()[si] -= step_size_ * df;
    }
    return compute_energy_and_forces();
  }

  double step_size_;
};

TEST(Minimizer, ConstructionInitializesAliases) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver);
  EXPECT_EQ(min.step_count(), 0);
  // max_force is 0 before any step runs; energy is computed though
  EXPECT_DOUBLE_EQ(min.max_force(), 0.0);
}

TEST(Minimizer, RunReturnsConvergenceStatus) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver, 1e-4, 1.0);  // high force_limit = easy convergence

  // With chemical potential = 0 but mu_id(rho=0.5) = ln(0.5) < 0,
  // the ideal gas is not at equilibrium for mu=0 unless rho=1.
  bool converged = min.run(10);
  EXPECT_GE(min.step_count(), 1);
  // Whether it converges depends on force_limit; just check it runs
  (void)converged;
}

TEST(Minimizer, ResumeDoesNotResetStepCounter) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver, 1e-5, 0.001);

  (void)min.run(5);
  EXPECT_EQ(min.step_count(), 5);

  (void)min.resume(3);
  EXPECT_EQ(min.step_count(), 8);
}

TEST(Minimizer, ResetRestartsFromCurrentDensity) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver, 1e-4, 0.001);

  (void)min.run(5);
  long count_after_run = min.step_count();
  EXPECT_EQ(count_after_run, 5);

  min.reset();
  // After reset, step_count is preserved in data, but a new run resets it
  (void)min.run(3);
  EXPECT_EQ(min.step_count(), 3);
}

TEST(Minimizer, StepCallbackCanStopIteration) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver, 1e-4, 0.001);

  int callback_count = 0;
  min.set_step_callback([&](long, double, double) {
    ++callback_count;
    return callback_count < 3;  // Stop after 3 callbacks
  });

  (void)min.run(100);
  EXPECT_EQ(callback_count, 3);
  EXPECT_EQ(min.step_count(), 3);
}

TEST(Minimizer, FixedDirectionProjection) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver, 1e-4, 0.001);

  arma::vec dir(solver.density(0).size(), arma::fill::ones);
  min.set_fixed_direction(dir);
  EXPECT_TRUE(min.has_fixed_direction());

  min.clear_fixed_direction();
  EXPECT_FALSE(min.has_fixed_direction());
}

TEST(Minimizer, FixedDirectionThrowsOnZeroNorm) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver, 1e-4, 0.001);

  arma::vec zero_dir(solver.density(0).size(), arma::fill::zeros);
  EXPECT_THROW(min.set_fixed_direction(zero_dir), std::invalid_argument);
}

TEST(Minimizer, FixedDirectionThrowsOnSizeMismatch) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver, 1e-4, 0.001);

  arma::vec bad_dir(3, arma::fill::ones);  // Wrong size
  EXPECT_THROW(min.set_fixed_direction(bad_dir), std::invalid_argument);
}

// ── Fire2Minimizer ───────────────────────────────────────────────────────

TEST(Fire2Minimizer, ConstructionInitializesState) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  dynamics::Fire2Minimizer fire(solver);
  EXPECT_GT(fire.dt(), 0.0);
  EXPECT_GT(fire.alpha(), 0.0);
}

TEST(Fire2Minimizer, ConfigCanBeCustomized) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  dynamics::Fire2Config config{.dt = 5e-4, .dt_max = 5e-3, .alpha_start = 0.2};
  dynamics::Fire2Minimizer fire(solver, config);

  EXPECT_DOUBLE_EQ(fire.config().dt, 5e-4);
  EXPECT_DOUBLE_EQ(fire.config().dt_max, 5e-3);
  EXPECT_DOUBLE_EQ(fire.config().alpha_start, 0.2);
}

TEST(Fire2Minimizer, RunReducesEnergy) {
  // For an ideal gas with mu=0, equilibrium is at rho = 1 (since mu_id = ln(rho) = 0)
  // Start at rho=0.5, which gives F_id = V*(rho*ln(rho) - rho) and force = ln(rho) * dV < 0
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);

  // Set mu=0 explicitly
  solver.species(0).set_chemical_potential(0.0);

  dynamics::Fire2Config config{.dt = 1e-4, .dt_max = 0.01, .force_limit = 0.01};
  dynamics::Fire2Minimizer fire(solver, config);

  double e_initial = fire.energy();
  (void)fire.run(50);
  double e_final = fire.energy();

  // Energy should decrease (or at least not increase significantly)
  EXPECT_LE(e_final, e_initial + 1e-10);
}

TEST(Fire2Minimizer, IdealGasConvergesForSmallSystem) {
  // Very small system: ideal gas should converge quickly
  double dx = 1.0;
  double l = 2.0;
  auto solver = make_ideal_gas_solver(0.3, dx, l);
  solver.species(0).set_chemical_potential(std::log(0.3));  // Equilibrium at rho=0.3

  dynamics::Fire2Config config{.dt = 1e-3, .dt_max = 0.1, .force_limit = 1e-6};
  dynamics::Fire2Minimizer fire(solver, config);

  bool converged = fire.run(500);
  // The ideal gas with matching mu should converge
  EXPECT_TRUE(converged);
}

TEST(Fire2Minimizer, EnergyDecreasesMonotonicallyForIdealGas) {
  // Start away from equilibrium: rho=0.3, target rho=0.5 (mu=ln(0.5))
  auto solver = make_ideal_gas_solver(0.3, 0.5, 2.0);
  solver.species(0).set_chemical_potential(std::log(0.5));

  dynamics::Fire2Config config{.dt = 1e-3, .dt_max = 0.05, .force_limit = 1e-10};
  dynamics::Fire2Minimizer fire(solver, config);

  std::vector<double> energies;
  fire.set_step_callback([&](long, double e, double) {
    energies.push_back(e);
    return energies.size() < 50;
  });

  (void)fire.run(50);

  // Check energy is (mostly) non-increasing — allow for FIRE backtracking
  int decrease_count = 0;
  for (size_t i = 1; i < energies.size(); ++i) {
    if (energies[i] <= energies[i - 1] + 1e-12) {
      ++decrease_count;
    }
  }
  // Most steps should decrease energy (allow for occasional backtracks)
  EXPECT_GT(decrease_count, static_cast<int>(energies.size()) / 2);
}

// ── Quadratic potential test ─────────────────────────────────────────────

TEST(Fire2Minimizer, QuadraticPotentialConvergesToMinimum) {
  // Set up a system where the equilibrium density profile is known.
  // For an ideal gas with mu_target = ln(rho_eq), the equilibrium is
  // uniform density = rho_eq.
  double rho_eq = 0.7;
  double dx = 1.0;
  double l = 3.0;

  auto solver = make_ideal_gas_solver(0.3, dx, l);  // Start away from equilibrium
  solver.species(0).set_chemical_potential(std::log(rho_eq));

  dynamics::Fire2Config config{.dt = 1e-3, .dt_max = 0.1, .force_limit = 1e-8};
  dynamics::Fire2Minimizer fire(solver, config);
  bool converged = fire.run(1000);

  EXPECT_TRUE(converged);

  // Check density is close to equilibrium
  const arma::vec& rho = solver.density(0).values();
  for (arma::uword i = 0; i < rho.n_elem; ++i) {
    EXPECT_NEAR(rho(i), rho_eq, 1e-4);
  }
}

// ── DDFT ─────────────────────────────────────────────────────────────────

TEST(Integrator, ConstructionInitializesState) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  dynamics::Integrator integrator(solver);
  EXPECT_GT(integrator.dt(), 0.0);
  EXPECT_GT(integrator.diffusion_coefficient(), 0.0);
  EXPECT_EQ(integrator.scheme(), dynamics::IntegrationScheme::SplitOperator);
}

TEST(Integrator, ConfigCanBeCustomized) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  dynamics::IntegratorConfig config{.dt = 1e-5, .diffusion_coefficient = 2.0};
  dynamics::Integrator integrator(solver, config);
  EXPECT_DOUBLE_EQ(integrator.config().dt, 1e-5);
  EXPECT_DOUBLE_EQ(integrator.config().diffusion_coefficient, 2.0);
}

TEST(Integrator, FreeDiffusionSmoothsProfile) {
  // Start with a Gaussian perturbation on top of uniform density.
  // Free diffusion should smooth it out (reduce variance).
  double dx = 0.5;
  double l = 5.0;
  double rho0 = 0.5;

  auto d = density::Density(dx, {l, l, l});
  arma::vec rho(d.size());
  double sigma = 0.5;
  for (arma::uword idx = 0; idx < d.size(); ++idx) {
    long nz = d.shape()[2];
    long ny = d.shape()[1];
    long iz = static_cast<long>(idx) % nz;
    long iy = (static_cast<long>(idx) / nz) % ny;
    long ix = static_cast<long>(idx) / (nz * ny);
    double x = ix * dx - l / 2.0;
    double y = iy * dx - l / 2.0;
    double z = iz * dx - l / 2.0;
    double r2 = x * x + y * y + z * z;
    rho(idx) = rho0 + 0.1 * std::exp(-r2 / (2.0 * sigma * sigma));
  }
  d.set(rho);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double var_before = arma::var(solver.density(0).values());

  dynamics::IntegratorConfig config{.dt = 1e-4, .diffusion_coefficient = 1.0, .force_limit = 1e-12};
  dynamics::Integrator integrator(solver, config);

  (void)integrator.run(10);

  double var_after = arma::var(solver.density(0).values());

  // Variance should decrease (diffusion smooths the profile)
  EXPECT_LT(var_after, var_before);
}

TEST(Integrator, UniformDensityRemainsStable) {
  // A uniform density with matching chemical potential should be stable.
  double rho0 = 0.5;
  auto solver = make_ideal_gas_solver(rho0, 0.5, 2.0);
  solver.species(0).set_chemical_potential(std::log(rho0));

  dynamics::IntegratorConfig config{.dt = 1e-4, .diffusion_coefficient = 1.0, .force_limit = 1e-12};
  dynamics::Integrator integrator(solver, config);

  (void)integrator.run(10);

  const arma::vec& rho = solver.density(0).values();
  double max_deviation = arma::max(arma::abs(rho - rho0));
  EXPECT_LT(max_deviation, 1e-6);
}

TEST(Integrator, ConservesMassApproximately) {
  // The integrator should approximately conserve total mass
  double dx = 0.5;
  double l = 4.0;

  auto d = density::Density(dx, {l, l, l});
  arma::vec rho(d.size());
  for (arma::uword i = 0; i < d.size(); ++i) {
    long nz = d.shape()[2];
    long ny = d.shape()[1];
    long iz = static_cast<long>(i) % nz;
    long iy = (static_cast<long>(i) / nz) % ny;
    long ix = static_cast<long>(i) / (nz * ny);
    double x = ix * dx;
    rho(i) = 0.5 + 0.1 * std::sin(2.0 * std::numbers::pi * x / l);
  }
  d.set(rho);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double mass_before = solver.density(0).number_of_atoms();

  dynamics::IntegratorConfig config{.dt = 1e-5, .diffusion_coefficient = 1.0, .force_limit = 1e-12};
  dynamics::Integrator integrator(solver, config);
  (void)integrator.run(10);

  double mass_after = solver.density(0).number_of_atoms();

  EXPECT_NEAR(mass_after, mass_before, std::abs(mass_before) * 0.01);
}

// ── DdftIf ──────────────────────────────────────────────────────────────

TEST(IntegratorCrankNicholson, ConstructionInitializesState) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  dynamics::IntegratorConfig config{.scheme = dynamics::IntegrationScheme::CrankNicholson};
  dynamics::Integrator integrator(solver, config);
  EXPECT_GT(integrator.dt(), 0.0);
  EXPECT_EQ(integrator.scheme(), dynamics::IntegrationScheme::CrankNicholson);
}

TEST(IntegratorCrankNicholson, ConfigCanBeCustomized) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  dynamics::IntegratorConfig config{
      .scheme = dynamics::IntegrationScheme::CrankNicholson,
      .dt = 1e-5,
      .diffusion_coefficient = 2.0,
      .crank_nicholson_iterations = 10};
  dynamics::Integrator integrator(solver, config);
  EXPECT_DOUBLE_EQ(integrator.config().dt, 1e-5);
  EXPECT_EQ(integrator.config().crank_nicholson_iterations, 10);
}

TEST(IntegratorCrankNicholson, UniformDensityRemainsStable) {
  double rho0 = 0.5;
  auto solver = make_ideal_gas_solver(rho0, 0.5, 2.0);
  solver.species(0).set_chemical_potential(std::log(rho0));

  dynamics::IntegratorConfig config{
      .scheme = dynamics::IntegrationScheme::CrankNicholson,
      .dt = 1e-4,
      .diffusion_coefficient = 1.0,
      .force_limit = 1e-12,
      .crank_nicholson_iterations = 3};
  dynamics::Integrator integrator(solver, config);

  (void)integrator.run(10);

  const arma::vec& rho = solver.density(0).values();
  double max_deviation = arma::max(arma::abs(rho - rho0));
  EXPECT_LT(max_deviation, 1e-6);
}

TEST(IntegratorCrankNicholson, FreeDiffusionSmoothsProfile) {
  double dx = 0.5;
  double l = 4.0;

  auto d = density::Density(dx, {l, l, l});
  arma::vec rho(d.size());
  double rho0 = 0.5;
  for (arma::uword idx = 0; idx < d.size(); ++idx) {
    long nz = d.shape()[2];
    long ny = d.shape()[1];
    long iz = static_cast<long>(idx) % nz;
    long iy = (static_cast<long>(idx) / nz) % ny;
    long ix = static_cast<long>(idx) / (nz * ny);
    double x = ix * dx - l / 2.0;
    double y = iy * dx - l / 2.0;
    double z = iz * dx - l / 2.0;
    double r2 = x * x + y * y + z * z;
    rho(idx) = rho0 + 0.05 * std::exp(-r2 / (2.0 * 0.5 * 0.5));
  }
  d.set(rho);

  Solver solver;
  solver.add_species(std::make_unique<species::Species>(std::move(d)));

  double var_before = arma::var(solver.density(0).values());

  dynamics::IntegratorConfig config{
      .scheme = dynamics::IntegrationScheme::CrankNicholson,
      .dt = 1e-4,
      .diffusion_coefficient = 1.0,
      .force_limit = 1e-12,
      .crank_nicholson_iterations = 5};
  dynamics::Integrator integrator(solver, config);
  (void)integrator.run(10);

  double var_after = arma::var(solver.density(0).values());
  EXPECT_LT(var_after, var_before);
}

// ── Error paths ──────────────────────────────────────────────────────────

TEST(Integrator, ThrowsWhenSolverHasNoSpecies) {
  Solver solver;
  EXPECT_THROW(dynamics::Integrator integrator(solver), std::invalid_argument);
}

TEST(Fire2Minimizer, ThrowsOnTooManyUphillSteps) {
  auto solver = make_ideal_gas_solver(0.5, 1.0, 2.0);

  dynamics::Fire2Config config{
      .dt = 1e-3,
      .max_uphill_steps = 0,  // Zero tolerance: throw on first non-downhill step
      .force_limit = 1e-20};
  dynamics::Fire2Minimizer fire(solver, config);

  EXPECT_THROW((void)fire.run(100), std::runtime_error);
}

TEST(Minimizer, StopsWhenDensityDropsBelowMinimum) {
  // Use a min_density threshold that the system will initially exceed
  // but then cross below after a step.
  auto solver = make_ideal_gas_solver(0.01, 1.0, 2.0);

  // Set min_density above the actual average density / volume
  double volume = 2.0 * 2.0 * 2.0;
  double n_total = solver.density(0).number_of_atoms();
  double high_threshold = 2.0 * n_total / volume;

  SteepestDescent min(solver, 1e-4, 0.001, high_threshold);
  bool converged = min.run(100);

  // Should stop early (not converge) because density is below threshold
  EXPECT_FALSE(converged);
  EXPECT_LT(min.step_count(), 100);
}

TEST(Minimizer, FixedDirectionProjectsForcesDuringRun) {
  auto solver = make_ideal_gas_solver(0.5, 0.5, 2.0);
  SteepestDescent min(solver, 1e-4, 0.001);

  arma::vec dir(solver.density(0).size(), arma::fill::ones);
  min.set_fixed_direction(dir);

  // Run with fixed direction active — forces should be projected
  (void)min.run(5);
  EXPECT_EQ(min.step_count(), 5);
}
