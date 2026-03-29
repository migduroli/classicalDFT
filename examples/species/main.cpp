#include "dft.h"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft::density;
using namespace dft::species;

int main() {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  double dx = 0.1;
  arma::rowvec3 box = {10.0, 10.0, 10.0};
  double rho0 = 0.5;

  // ── Alias coordinates ─────────────────────────────────────────────────

  std::cout << "=== Alias coordinates ===" << std::endl;
  std::cout << "rho_min = " << Species::RHO_MIN << std::endl;
  std::cout << "Mapping: rho = rho_min + x^2" << std::endl;
  std::cout << "Inverse: x = sqrt(rho - rho_min)" << std::endl;

  Species s(Density(dx, box), /*mu=*/-2.5);
  s.density().values().fill(rho0);

  arma::vec alias = s.density_alias();
  s.set_density_from_alias(alias);
  double roundtrip_err = arma::max(arma::abs(s.density().values() - rho0));
  std::cout << "Round-trip max error: " << roundtrip_err << std::endl;

  // ── External field: gravitational slab ────────────────────────────────

  std::cout << "\n=== External field ===" << std::endl;

  long nx = s.density().shape()[0];
  long ny = s.density().shape()[1];
  long nz = s.density().shape()[2];
  double Lz = box(2);

  // Linear gravitational potential along z
  double g_field = 1.0;
  for (long ix = 0; ix < nx; ++ix) {
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        double z = dx * iz;
        s.density().external_field()(s.density().flat_index(ix, iy, iz)) = g_field * z;
      }
    }
  }

  // Barometric density: rho(z) ~ exp(-g * z / kT), normalised to N_target atoms
  double n_target = 200.0;
  double norm = 0.0;
  arma::vec barometric(s.density().size());
  for (long ix = 0; ix < nx; ++ix) {
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        double z = dx * iz;
        auto idx = s.density().flat_index(ix, iy, iz);
        barometric(idx) = std::exp(-g_field * z);
        norm += barometric(idx);
      }
    }
  }
  barometric *= n_target / (norm * s.density().cell_volume());
  s.density().set(barometric);

  std::cout << "Gravitational field g = " << g_field << std::endl;
  std::cout << "Target atoms: " << n_target << std::endl;
  std::cout << "Actual atoms: " << s.density().number_of_atoms() << std::endl;
  std::cout << "External field energy: " << s.external_field_energy(false) << std::endl;

  // ── Fixed-mass constraint ─────────────────────────────────────────────

  std::cout << "\n=== Fixed-mass constraint ===" << std::endl;

  s.set_fixed_mass(n_target);
  std::cout << "Constraint: N = " << *s.fixed_mass() << std::endl;

  // Perturb density and show rescaling
  s.density().scale(1.05);
  std::cout << "After 5% perturbation: N = " << s.density().number_of_atoms() << std::endl;

  s.begin_force_calculation();
  std::cout << "After begin_force_calculation: N = " << s.density().number_of_atoms() << std::endl;

  // Save barometric slice for plotting (before switching to uniform for forces)
  std::vector<double> barometric_z(nz), vext_z_saved(nz);
  for (long iz = 0; iz < nz; ++iz) {
    auto uid = s.density().flat_index(0, 0, iz);
    barometric_z[iz] = s.density().values()(uid);
    vext_z_saved[iz] = s.density().external_field()(uid);
  }

  // ── Force protocol ────────────────────────────────────────────────────
  // Use a uniform density (far from equilibrium) so forces are non-trivial.
  // The barometric profile is the equilibrium solution, which yields zero net force.

  std::cout << "\n=== Force protocol ===" << std::endl;

  s.density().values().fill(rho0);
  s.begin_force_calculation();

  s.zero_force();

  // Ideal gas contribution: dF/drho = log(rho)
  arma::vec ideal_force = arma::log(arma::clamp(s.density().values(), Species::RHO_MIN, arma::datum::inf));
  ideal_force *= s.density().cell_volume();
  s.add_to_force(ideal_force);

  // External field contribution (accumulates V_ext * dV to force)
  double e_ext = s.external_field_energy(/*accumulate_force=*/true);
  std::cout << "External field energy: " << e_ext << std::endl;

  s.end_force_calculation();
  std::cout << "Lagrange multiplier mu: " << s.chemical_potential() << std::endl;
  std::cout << "Convergence (max |dF/dV|): " << s.convergence_monitor() << std::endl;

  // ── Alias chain rule ──────────────────────────────────────────────────

  std::cout << "\n=== Alias chain rule ===" << std::endl;

  arma::vec x = s.density_alias();
  arma::vec alias_f = s.alias_force(x);
  std::cout << "Force in rho-space max |f|: " << arma::max(arma::abs(s.force())) << std::endl;
  std::cout << "Force in x-space   max |f|: " << arma::max(arma::abs(alias_f)) << std::endl;
  std::cout << "Chain rule: dF/dx = 2x * dF/drho" << std::endl;

  // ── Plots ──────────────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;
  plt::backend("Agg");

  {
    // Plot 1: Barometric density and external field along z
    std::vector<double> z_vals(nz);
    for (long iz = 0; iz < nz; ++iz) {
      z_vals[iz] = dx * iz;
    }

    plt::figure_size(800, 550);
    plt::named_plot(R"($\rho(z) \propto \exp(-gz)$)", z_vals, barometric_z, "b-");
    plt::named_plot(R"($V_\mathrm{ext} = gz$)", z_vals, vext_z_saved, "r--");
    plt::xlim(0.0, Lz);
    plt::xlabel(R"($z$)");
    plt::ylabel(R"($\rho(z)$ / $V_\mathrm{ext}(z)$)");
    plt::title("Barometric density profile");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/barometric_density.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/barometric_density.png") << std::endl;
  }

  {
    // Plot 2: Alias coordinate mapping and its derivative
    int npts = 300;
    double rho_max = 1.5;
    std::vector<double> rho_vec(npts), x_vec(npts), dxdrho_vec(npts);
    for (int i = 0; i < npts; ++i) {
      double r = Species::RHO_MIN + (rho_max - Species::RHO_MIN) * i / (npts - 1);
      rho_vec[i] = r;
      x_vec[i] = std::sqrt(r - Species::RHO_MIN);
      double xi = x_vec[i];
      dxdrho_vec[i] = (xi > 1e-6) ? 1.0 / (2.0 * xi) : 0.0;
    }

    plt::figure_size(800, 550);
    plt::named_plot(R"($x = (\rho - \rho_\mathrm{min})^{1/2}$)", rho_vec, x_vec, "g-");
    plt::named_plot(R"($dx/d\rho = 1/(2x)$)", rho_vec, dxdrho_vec, "m--");
    plt::xlim(0.0, rho_max);
    plt::ylim(0.0, 3.0);
    plt::xlabel(R"($\rho$)");
    plt::ylabel(R"($x(\rho)$ / $dx/d\rho$)");
    plt::title(R"(Alias mapping $x(\rho)$ and sensitivity)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/alias_mapping.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/alias_mapping.png") << std::endl;
  }

  {
    // Plot 3: Force density in rho-space vs alias-space along z
    double dV = s.density().cell_volume();
    std::vector<double> z_vals(nz), frho_z(nz), fx_z(nz);
    for (long iz = 0; iz < nz; ++iz) {
      z_vals[iz] = dx * iz;
      auto uid = s.density().flat_index(0, 0, iz);
      frho_z[iz] = s.force()(uid) / dV;
      fx_z[iz] = alias_f(uid) / dV;
    }

    plt::figure_size(800, 550);
    plt::named_plot(R"($\delta F/\delta\rho\; / \;dV$)", z_vals, frho_z, "b-");
    plt::named_plot(R"($\delta F/\delta x\; / \;dV$)", z_vals, fx_z, "m--");
    plt::xlim(0.0, Lz);
    plt::xlabel(R"($z$)");
    plt::ylabel(R"(force / $dV$)");
    plt::title("Force density along z (ix=iy=0)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/force_profiles.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/force_profiles.png") << std::endl;
  }
#endif

  return 0;
}
