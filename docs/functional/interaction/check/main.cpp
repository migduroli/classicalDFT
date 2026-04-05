// check.cpp — Cross-validation of our mean-field interaction weights against
// Jim's classicalDFT library (Interaction.cpp).
//
// Compares:
//   1. Cell-by-cell QF weights at matched grid displacements
//   2. Grid-level a_vdw at multiple dx values
//   3. a_vdw grid convergence toward analytical value

#include "dft.hpp"
#include "legacy/classicaldft.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace dft;

static int g_failures = 0;
static int g_checks = 0;

static void check(std::string_view label, double ours, double jim, double tol = 1e-10) {
  ++g_checks;
  double diff = std::abs(ours - jim);
  bool ok = diff <= tol;
  if (!ok) {
    ++g_failures;
    std::cout << "  FAIL " << label << ": ours=" << ours << " jim=" << jim << " diff=" << diff << "\n";
  }
}

static void section(std::string_view title) {
  std::cout << "\n" << title << "\n";
  std::cout << std::string(title.size(), '-') << "\n";
}

int main() {
  std::cout << std::setprecision(15);

  namespace pot = physics::potentials;

  constexpr double sigma = 1.0;
  constexpr double eps = 1.0;
  constexpr double rcut = 2.5;
  constexpr double kT = 1.0;

  auto our_lj = pot::make_lennard_jones(sigma, eps, rcut);
  pot::Potential plj = our_lj;

  // Step 1: Cell-by-cell QF weight comparison at dx=0.4

  section("Step 1: Cell-by-cell QF weight at dx=0.4");

  double dx = 0.4;
  int Nx_lim = 1 + static_cast<int>(rcut / dx);
  double max_diff_w = 0.0;
  int n_cells = 0;

  for (int ix = -Nx_lim - 1; ix <= Nx_lim + 1; ++ix) {
    for (int iy = -Nx_lim - 1; iy <= Nx_lim + 1; ++iy) {
      for (int iz = -Nx_lim - 1; iz <= Nx_lim + 1; ++iz) {
        double r2_approx = (ix * ix + iy * iy + iz * iz) * dx * dx;
        double r_check = rcut + std::sqrt(3.0) * 0.5 * dx;
        if (r2_approx > r_check * r_check)
          continue;

        double our_w = functionals::detail::cell_weight_quadratic_f(
            plj,
            pot::SplitScheme::WeeksChandlerAndersen,
            kT,
            dx,
            ix,
            iy,
            iz
        );
        double jim_w = legacy::interactions::generate_weight_QF(sigma, eps, rcut, ix, iy, iz, dx) / kT;

        double diff = std::abs(our_w - jim_w);
        max_diff_w = std::max(max_diff_w, diff);
        ++n_cells;
      }
    }
  }

  check("QF cell max|w diff| (dx=0.4)", max_diff_w, 0.0);
  std::cout << "  " << n_cells << " cells compared, max|w diff| = " << max_diff_w << "\n";

  // Step 2: Cell-by-cell at dx=0.2

  section("Step 2: Cell-by-cell QF weight at dx=0.2");

  dx = 0.2;
  Nx_lim = 1 + static_cast<int>(rcut / dx);
  max_diff_w = 0.0;
  n_cells = 0;

  for (int ix = -Nx_lim - 1; ix <= Nx_lim + 1; ++ix) {
    for (int iy = -Nx_lim - 1; iy <= Nx_lim + 1; ++iy) {
      for (int iz = -Nx_lim - 1; iz <= Nx_lim + 1; ++iz) {
        double r2_approx = (ix * ix + iy * iy + iz * iz) * dx * dx;
        double r_check = rcut + std::sqrt(3.0) * 0.5 * dx;
        if (r2_approx > r_check * r_check)
          continue;

        double our_w = functionals::detail::cell_weight_quadratic_f(
            plj,
            pot::SplitScheme::WeeksChandlerAndersen,
            kT,
            dx,
            ix,
            iy,
            iz
        );
        double jim_w = legacy::interactions::generate_weight_QF(sigma, eps, rcut, ix, iy, iz, dx) / kT;

        double diff = std::abs(our_w - jim_w);
        max_diff_w = std::max(max_diff_w, diff);
        ++n_cells;
      }
    }
  }

  check("QF cell max|w diff| (dx=0.2)", max_diff_w, 0.0);
  std::cout << "  " << n_cells << " cells compared, max|w diff| = " << max_diff_w << "\n";

  // Step 3: Grid a_vdw comparison at multiple dx

  section("Step 3: Grid a_vdw at multiple dx values");

  std::vector<double> grid_sizes = { 0.5, 0.4, 0.3, 0.2, 0.1 };
  int N_grid = 32;

  for (double dx_val : grid_sizes) {
    // Our a_vdw: sum all cell weights * dV over all displacements
    double dv = dx_val * dx_val * dx_val;
    int lim = 1 + static_cast<int>(rcut / dx_val);
    double r_check = rcut + std::sqrt(3.0) * 0.5 * dx_val;
    double our_avdw = 0.0;
    double jim_avdw = 0.0;

    for (int ix = -lim - 1; ix <= lim + 1; ++ix) {
      for (int iy = -lim - 1; iy <= lim + 1; ++iy) {
        for (int iz = -lim - 1; iz <= lim + 1; ++iz) {
          double r2_approx = (ix * ix + iy * iy + iz * iz) * dx_val * dx_val;
          if (r2_approx > r_check * r_check)
            continue;

          double our_w = functionals::detail::cell_weight_quadratic_f(
              plj,
              pot::SplitScheme::WeeksChandlerAndersen,
              kT,
              dx_val,
              ix,
              iy,
              iz
          );
          our_avdw += our_w * dv;

          double jim_w = legacy::interactions::generate_weight_QF(sigma, eps, rcut, ix, iy, iz, dx_val);
          jim_avdw += jim_w * dv / kT;
        }
      }
    }

    check("a_vdw(dx=" + std::to_string(dx_val) + ")", our_avdw, jim_avdw, 1e-10);
    std::cout << "  dx=" << dx_val << ": ours=" << our_avdw << " jim=" << jim_avdw
              << " diff=" << std::abs(our_avdw - jim_avdw) << "\n";
  }

  // Step 4: Analytical a_vdw reference

  section("Step 4: Analytical a_vdw reference");

  double a_vdw_analytical = pot::vdw_integral(plj, kT, pot::SplitScheme::WeeksChandlerAndersen);
  std::cout << "  Analytical a_vdw = " << a_vdw_analytical << "\n";

  // Summary

  section("Summary");
  std::cout << "  Total checks: " << g_checks << "\n";
  std::cout << "  Failures:     " << g_failures << "\n";

  if (g_failures > 0) {
    std::cout << "\n  *** " << g_failures << " FAILURES ***\n";
    return 1;
  }
  std::cout << "\n  All checks passed.\n";
  return 0;
}
