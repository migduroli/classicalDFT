#include <classicaldft>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>

using namespace dft_core::physics::density;
using namespace dft_core::physics::species;

int main() {
  std::filesystem::create_directories("exports");

  double dx = 0.1;
  arma::rowvec3 box = {10.0, 10.0, 10.0};
  double rho0 = 0.5;

  // ── Alias coordinates ─────────────────────────────────────────────────

  std::cout << "=== Alias coordinates ===" << std::endl;
  std::cout << "rho_min = " << Species::rho_min << std::endl;
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
  arma::vec ideal_force = arma::log(arma::clamp(s.density().values(), Species::rho_min, arma::datum::inf));
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

  // ── Grace plots ────────────────────────────────────────────────────────

#ifdef DFT_HAS_GRACE
  using namespace dft_core::grace_plot;

  {
    // Plot 1: Barometric density and external field along z
    std::vector<double> z_vals(nz);
    for (long iz = 0; iz < nz; ++iz) {
      z_vals[iz] = dx * iz;
    }

    auto gp = Grace();
    gp.set_title("Barometric density profile");
    gp.set_label("z", Axis::X);
    gp.set_label("\\xr\\f{}(z) / V\\sext\\N(z)", Axis::Y);

    auto ds_rho = gp.add_dataset(z_vals, barometric_z);
    gp.set_color(Color::BLUE, ds_rho);
    gp.set_legend("\\xr\\f{}(z) \\x\\c\\f{} exp(-gz)", ds_rho);

    auto ds_vext = gp.add_dataset(z_vals, vext_z_saved);
    gp.set_color(Color::RED, ds_vext);
    gp.set_line_type(LineStyle::DASHEDLINE_EN, ds_vext);
    gp.set_legend("V\\sext\\N = gz", ds_vext);

    gp.set_x_limits(0.0, Lz);
    gp.set_ticks(2.0, 1.0);
    gp.print_to_file("exports/barometric_density.png", ExportFormat::PNG);
    gp.redraw_and_wait(false, false);
  }

  {
    // Plot 2: Alias coordinate mapping and its derivative
    int npts = 300;
    double rho_max = 1.5;
    std::vector<double> rho_vec(npts), x_vec(npts), dxdrho_vec(npts);
    for (int i = 0; i < npts; ++i) {
      double r = Species::rho_min + (rho_max - Species::rho_min) * i / (npts - 1);
      rho_vec[i] = r;
      x_vec[i] = std::sqrt(r - Species::rho_min);
      // dx/drho = 1 / (2x): show scaled version 2 * dx/drho for visibility
      double xi = x_vec[i];
      dxdrho_vec[i] = (xi > 1e-6) ? 1.0 / (2.0 * xi) : 0.0;
    }

    auto gp = Grace();
    gp.set_title("Alias mapping x(\\xr\\f{}) and sensitivity");
    gp.set_label("\\xr", Axis::X);
    gp.set_label("x(\\xr\\f{})  /  dx/d\\xr", Axis::Y);

    auto ds_x = gp.add_dataset(rho_vec, x_vec);
    gp.set_color(Color::DARKGREEN, ds_x);
    gp.set_legend("x = (\\xr\\f{} - \\xr\\f{}\\smin\\N)\\S1/2", ds_x);

    auto ds_dx = gp.add_dataset(rho_vec, dxdrho_vec);
    gp.set_color(Color::ORANGE, ds_dx);
    gp.set_line_type(LineStyle::DASHEDLINE_EN, ds_dx);
    gp.set_legend("dx/d\\xr\\f{} = 1/(2x)", ds_dx);

    gp.set_x_limits(0.0, rho_max);
    gp.set_y_limits(0.0, 3.0);
    gp.set_ticks(0.25, 0.5);
    gp.print_to_file("exports/alias_mapping.png", ExportFormat::PNG);
    gp.redraw_and_wait(false, false);
  }

  {
    // Plot 3: Force density (force / dV) in rho-space vs alias-space along z
    double dV = s.density().cell_volume();
    std::vector<double> z_vals(nz), frho_z(nz), fx_z(nz);
    for (long iz = 0; iz < nz; ++iz) {
      z_vals[iz] = dx * iz;
      auto uid = s.density().flat_index(0, 0, iz);
      frho_z[iz] = s.force()(uid) / dV;
      fx_z[iz] = alias_f(uid) / dV;
    }

    auto gp = Grace();
    gp.set_title("Force density along z (ix=iy=0)");
    gp.set_label("z", Axis::X);
    gp.set_label("force / dV", Axis::Y);

    auto ds_fr = gp.add_dataset(z_vals, frho_z);
    gp.set_color(Color::BLUE, ds_fr);
    gp.set_legend("\\xd\\f{}F/\\xd\\xr / dV", ds_fr);

    auto ds_fx = gp.add_dataset(z_vals, fx_z);
    gp.set_color(Color::MAGENTA, ds_fx);
    gp.set_line_type(LineStyle::DASHEDLINE_EN, ds_fx);
    gp.set_legend("\\xd\\f{}F/\\xd\\f{}x / dV", ds_fx);

    gp.set_x_limits(0.0, Lz);
    gp.print_to_file("exports/force_profiles.png", ExportFormat::PNG);
    gp.redraw_and_wait(true, true);
  }
#endif

  return 0;
}
