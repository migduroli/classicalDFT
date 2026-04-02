#include "dft/algorithms/ddft.hpp"

#include "dft/grid.hpp"
#include "dft/math/fourier.hpp"

#include <armadillo>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft::algorithms::ddft;

static const dft::Grid GRID{.dx = 0.5, .box_size = {4.0, 4.0, 4.0}, .shape = {8, 8, 8}};

// Trivial force callback: ideal gas only (F = sum rho * (ln(rho) - 1)).
// Force = dF/drho = ln(rho), with cell volume factor.

static auto ideal_gas_forces(const std::vector<arma::vec>& densities) -> std::pair<double, std::vector<arma::vec>> {
  double energy = 0.0;
  double dv = GRID.cell_volume();
  std::vector<arma::vec> forces(densities.size());
  for (std::size_t s = 0; s < densities.size(); ++s) {
    arma::vec rho = arma::clamp(densities[s], 1e-18, arma::datum::inf);
    energy += arma::accu(rho % (arma::log(rho) - 1.0)) * dv;
    forces[s] = arma::log(rho) * dv;
  }
  return {energy, forces};
}

// k^2

TEST_CASE("compute_k_squared has correct size", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  long expected = GRID.shape[0] * GRID.shape[1] * (GRID.shape[2] / 2 + 1);
  CHECK(k2.n_elem == static_cast<arma::uword>(expected));
}

TEST_CASE("compute_k_squared has zero at k=0", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  CHECK(k2(0) == Catch::Approx(0.0).margin(1e-14));
}

TEST_CASE("compute_k_squared values are non-negative", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  CHECK(arma::all(k2 >= 0.0));
}

// Diffusion propagator

TEST_CASE("diffusion propagator is 1 at k=0", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  auto prop = diffusion_propagator(k2, 1.0, 0.01);
  CHECK(prop(0) == Catch::Approx(1.0));
}

TEST_CASE("diffusion propagator decays for k>0", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  auto prop = diffusion_propagator(k2, 1.0, 0.01);
  // All values should be in (0, 1]
  CHECK(arma::all(prop > 0.0));
  CHECK(arma::all(prop <= 1.0 + 1e-14));
}

// Split-operator step

TEST_CASE("split-operator step conserves total mass approximately", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  DdftConfig config{.dt = 1e-4, .diffusion_coefficient = 1.0};
  auto prop = diffusion_propagator(k2, config.diffusion_coefficient, config.dt);

  long n = GRID.total_points();
  arma::vec rho0 = 0.5 * arma::ones(static_cast<arma::uword>(n));
  // Add a Gaussian bump
  for (long ix = 0; ix < GRID.shape[0]; ++ix) {
    for (long iy = 0; iy < GRID.shape[1]; ++iy) {
      for (long iz = 0; iz < GRID.shape[2]; ++iz) {
        double x = (ix - GRID.shape[0] / 2.0) * GRID.dx;
        double y = (iy - GRID.shape[1] / 2.0) * GRID.dx;
        double z = (iz - GRID.shape[2] / 2.0) * GRID.dx;
        long idx = GRID.flat_index(ix, iy, iz);
        rho0(static_cast<arma::uword>(idx)) += 0.1 * std::exp(-(x * x + y * y + z * z));
      }
    }
  }

  double mass_before = arma::accu(rho0) * GRID.cell_volume();

  auto result = split_operator_step({rho0}, GRID, k2, prop, ideal_gas_forces, config);

  double mass_after = arma::accu(result.densities[0]) * GRID.cell_volume();

  // Mass conservation (within numerical tolerance for explicit method)
  CHECK(mass_after == Catch::Approx(mass_before).epsilon(1e-3));
}

TEST_CASE("split-operator step keeps density positive", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  DdftConfig config{.dt = 1e-5, .diffusion_coefficient = 1.0, .min_density = 1e-18};
  auto prop = diffusion_propagator(k2, config.diffusion_coefficient, config.dt);

  long n = GRID.total_points();
  arma::vec rho0 = 0.1 * arma::ones(static_cast<arma::uword>(n));

  auto result = split_operator_step({rho0}, GRID, k2, prop, ideal_gas_forces, config);

  CHECK(arma::all(result.densities[0] > 0.0));
}

TEST_CASE("split-operator diffuses a Gaussian towards uniform", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  DdftConfig config{.dt = 1e-3, .diffusion_coefficient = 1.0};
  auto prop = diffusion_propagator(k2, config.diffusion_coefficient, config.dt);

  long n = GRID.total_points();
  arma::vec rho0 = 0.5 * arma::ones(static_cast<arma::uword>(n));
  for (long ix = 0; ix < GRID.shape[0]; ++ix) {
    for (long iy = 0; iy < GRID.shape[1]; ++iy) {
      for (long iz = 0; iz < GRID.shape[2]; ++iz) {
        double x = (ix - GRID.shape[0] / 2.0) * GRID.dx;
        double y = (iy - GRID.shape[1] / 2.0) * GRID.dx;
        double z = (iz - GRID.shape[2] / 2.0) * GRID.dx;
        long idx = GRID.flat_index(ix, iy, iz);
        rho0(static_cast<arma::uword>(idx)) += 0.3 * std::exp(-(x * x + y * y + z * z));
      }
    }
  }

  double var_before = arma::var(rho0);

  // Take several steps
  std::vector<arma::vec> rho = {rho0};
  for (int i = 0; i < 10; ++i) {
    auto result = split_operator_step(rho, GRID, k2, prop, ideal_gas_forces, config);
    rho = result.densities;
  }

  double var_after = arma::var(rho[0]);

  // Variance should decrease (density becomes more uniform)
  CHECK(var_after < var_before);
}

// Crank-Nicholson step

TEST_CASE("crank-nicholson step keeps density positive", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  DdftConfig config{.dt = 1e-4, .diffusion_coefficient = 1.0, .min_density = 1e-18};

  long n = GRID.total_points();
  arma::vec rho0 = 0.1 * arma::ones(static_cast<arma::uword>(n));

  auto result = crank_nicholson_step({rho0}, GRID, k2, ideal_gas_forces, config);

  CHECK(arma::all(result.densities[0] > 0.0));
}

TEST_CASE("crank-nicholson conserves mass approximately", "[ddft]") {
  auto k2 = compute_k_squared(GRID);
  DdftConfig config{.dt = 1e-4, .diffusion_coefficient = 1.0};

  long n = GRID.total_points();
  arma::vec rho0 = 0.5 * arma::ones(static_cast<arma::uword>(n));
  for (long ix = 0; ix < GRID.shape[0]; ++ix) {
    for (long iy = 0; iy < GRID.shape[1]; ++iy) {
      for (long iz = 0; iz < GRID.shape[2]; ++iz) {
        double x = (ix - GRID.shape[0] / 2.0) * GRID.dx;
        double y = (iy - GRID.shape[1] / 2.0) * GRID.dx;
        double z = (iz - GRID.shape[2] / 2.0) * GRID.dx;
        long idx = GRID.flat_index(ix, iy, iz);
        rho0(static_cast<arma::uword>(idx)) += 0.1 * std::exp(-(x * x + y * y + z * z));
      }
    }
  }

  double mass_before = arma::accu(rho0) * GRID.cell_volume();

  auto result = crank_nicholson_step({rho0}, GRID, k2, ideal_gas_forces, config);

  double mass_after = arma::accu(result.densities[0]) * GRID.cell_volume();

  CHECK(mass_after == Catch::Approx(mass_before).epsilon(1e-3));
}
