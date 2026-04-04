// Cross-validation of interaction weights against Jim's code.

#include "dft.hpp"
#include "legacy/classicaldft.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace dft;
namespace pot = physics::potentials;
using Catch::Approx;

static constexpr double SIGMA = 1.0;
static constexpr double EPS = 1.0;
static constexpr double RCUT = 2.5;
static constexpr double KT = 1.0;

static auto lj_potential() -> pot::Potential {
  return pot::make_lennard_jones(SIGMA, EPS, RCUT);
}

TEST_CASE("Cell QF weights match legacy at dx=0.4", "[integration][interaction]") {
  auto potential = lj_potential();
  double dx = 0.4;
  int lim = 1 + static_cast<int>(RCUT / dx);
  double r_check = RCUT + std::sqrt(3.0) * 0.5 * dx;

  for (int ix = 0; ix <= lim + 1; ++ix)
    for (int iy = 0; iy <= ix; ++iy)
      for (int iz = 0; iz <= iy; ++iz) {
        double r2 = (ix * ix + iy * iy + iz * iz) * dx * dx;
        if (r2 > r_check * r_check)
          continue;
        double ours = functionals::detail::cell_weight_quadratic_f(
            potential, pot::SplitScheme::WeeksChandlerAndersen, KT, dx, ix, iy, iz
        );
        double jims = legacy::interactions::generate_weight_QF(SIGMA, EPS, RCUT, ix, iy, iz, dx) / KT;
        CHECK(ours == Approx(jims).margin(1e-10));
      }
}

TEST_CASE("Cell QF weights match legacy at dx=0.2", "[integration][interaction]") {
  auto potential = lj_potential();
  double dx = 0.2;
  int lim = 1 + static_cast<int>(RCUT / dx);
  double r_check = RCUT + std::sqrt(3.0) * 0.5 * dx;

  for (int ix = 0; ix <= lim + 1; ++ix)
    for (int iy = 0; iy <= ix; ++iy)
      for (int iz = 0; iz <= iy; ++iz) {
        double r2 = (ix * ix + iy * iy + iz * iz) * dx * dx;
        if (r2 > r_check * r_check)
          continue;
        double ours = functionals::detail::cell_weight_quadratic_f(
            potential, pot::SplitScheme::WeeksChandlerAndersen, KT, dx, ix, iy, iz
        );
        double jims = legacy::interactions::generate_weight_QF(SIGMA, EPS, RCUT, ix, iy, iz, dx) / KT;
        CHECK(ours == Approx(jims).margin(1e-10));
      }
}

TEST_CASE("Grid a_vdw matches legacy at multiple dx", "[integration][interaction]") {
  auto potential = lj_potential();
  for (double dx : {0.5, 0.4, 0.3, 0.2}) {
    double jim_a = legacy::interactions::compute_a_vdw_QF(SIGMA, EPS, RCUT, KT, dx);

    int lim = 1 + static_cast<int>(RCUT / dx);
    double r_check = RCUT + std::sqrt(3.0) * 0.5 * dx;
    double our_sum = 0.0;
    for (int ix = -(lim + 1); ix <= lim + 1; ++ix)
      for (int iy = -(lim + 1); iy <= lim + 1; ++iy)
        for (int iz = -(lim + 1); iz <= lim + 1; ++iz) {
          double r2 = (ix * ix + iy * iy + iz * iz) * dx * dx;
          if (r2 > r_check * r_check)
            continue;
          int nx = std::abs(ix), ny = std::abs(iy), nz = std::abs(iz);
          if (ny > nx)
            std::swap(nx, ny);
          if (nz > nx)
            std::swap(nx, nz);
          if (nz > ny)
            std::swap(ny, nz);
          our_sum += functionals::detail::cell_weight_quadratic_f(
                         potential, pot::SplitScheme::WeeksChandlerAndersen, KT, dx, nx, ny, nz
                     ) *
              dx * dx * dx;
        }

    INFO("dx=" << dx);
    CHECK(our_sum == Approx(jim_a).margin(1e-10));
  }
}
