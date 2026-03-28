#include <classicaldft>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numbers>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft_core::physics::density;
using namespace dft_core::physics::species;

int main() {
  std::filesystem::create_directories("exports");

  // ── Density on a periodic grid ──────────────────────────────────────────

  double dx = 0.1;
  arma::rowvec3 box = {5.0, 5.0, 5.0};
  Density rho(dx, box);

  long nx = rho.shape()[0];
  long ny = rho.shape()[1];
  long nz = rho.shape()[2];

  std::cout << "=== Density ===" << std::endl;
  std::cout << "Grid shape: " << nx << " x " << ny << " x " << nz << std::endl;
  std::cout << "Total points: " << rho.size() << std::endl;
  std::cout << "Spacing: " << rho.dx() << std::endl;
  std::cout << "Cell volume: " << rho.cell_volume() << std::endl;

  // Sinusoidal density profile: uniform in x,y; oscillates in z
  double rho0 = 0.8;
  double amplitude = 0.3;
  double Lz = box(2);

  for (long ix = 0; ix < nx; ++ix) {
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        double z = dx * iz;
        double val = rho0 + amplitude * std::sin(2.0 * std::numbers::pi * z / Lz);
        rho.set(rho.flat_index(ix, iy, iz), val);
      }
    }
  }

  std::cout << "\nSinusoidal density: rho0 = " << rho0 << ", A = " << amplitude << std::endl;
  std::cout << "Number of atoms: " << rho.number_of_atoms() << std::endl;
  std::cout << "Expected (rho0 * V): " << rho0 * box(0) * box(1) * box(2) << std::endl;

  auto com = rho.center_of_mass();
  std::cout << "Center of mass: (" << com(0) << ", " << com(1) << ", " << com(2) << ")" << std::endl;

  // ── External field ──────────────────────────────────────────────────────

  // Simple hard-wall potential at boundaries
  for (long ix = 0; ix < nx; ++ix) {
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        double z = dx * iz;
        double vext = 0.0;
        if (z < 0.5 || z > Lz - 0.5) vext = 10.0;
        rho.external_field()(rho.flat_index(ix, iy, iz)) = vext;
      }
    }
  }

  std::cout << "\nExternal field energy (hard walls): " << rho.external_field_energy() << std::endl;

  // ── FFT ────────────────────────────────────────────────────────────────

  rho.forward_fft();
  auto fourier = rho.fft().fourier();
  long ntot = rho.size();

  std::cout << "\nFFT DC / N = " << std::abs(fourier[0]) / ntot << " (expected " << rho0 << ")" << std::endl;
  std::cout << "|F(kz=1)| / N = " << std::abs(fourier[1]) / ntot << " (expected " << amplitude / 2.0 << ")"
            << std::endl;

  // ── Species ────────────────────────────────────────────────────────────

  Species s(Density(dx, box), /*mu=*/1.0);
  s.density().values().fill(rho0);

  std::cout << "\n=== Species ===" << std::endl;
  std::cout << "Chemical potential: " << s.chemical_potential() << std::endl;

  arma::vec alias = s.density_alias();
  s.set_density_from_alias(alias);
  double diff = arma::max(arma::abs(s.density().values() - rho0));
  std::cout << "Alias round-trip max error: " << diff << std::endl;

  double target = 50.0;
  s.set_fixed_mass(target);
  s.begin_force_calculation();
  std::cout << "\nFixed-mass rescaling (target = " << target << "):" << std::endl;
  std::cout << "Number of atoms: " << s.density().number_of_atoms() << std::endl;

  s.zero_force();
  arma::vec dF(s.density().size(), arma::fill::value(0.1));
  s.add_to_force(dF);
  s.end_force_calculation();
  std::cout << "Lagrange multiplier mu: " << s.chemical_potential() << std::endl;
  std::cout << "Convergence monitor: " << s.convergence_monitor() << std::endl;

  // ── Save / load ────────────────────────────────────────────────────────

  std::string fname = "exports/density_example.bin";
  rho.save(fname);
  Density rho2(dx, box);
  rho2.load(fname);
  std::cout << "\nSave/load round-trip OK: "
            << (arma::approx_equal(rho.values(), rho2.values(), "absdiff", 1e-15) ? "yes" : "no") << std::endl;

  // ── Plots ──────────────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;
  plt::backend("Agg");

  {
    // Plot 1: Density profile and external field along z-axis
    std::vector<double> z_vals(nz), rho_z(nz), vext_z(nz);
    for (long iz = 0; iz < nz; ++iz) {
      z_vals[iz] = dx * iz;
      rho_z[iz] = rho.values()(rho.flat_index(0, 0, iz));
      vext_z[iz] = rho.external_field()(rho.flat_index(0, 0, iz));
    }

    plt::figure_size(800, 550);
    plt::named_plot(R"($\rho(z)$)", z_vals, rho_z, "b-");
    plt::named_plot(R"($V_\mathrm{ext}(z)$)", z_vals, vext_z, "r--");
    plt::xlim(0.0, Lz);
    plt::ylim(-0.5, 11.0);
    plt::xlabel(R"($z$)");
    plt::ylabel(R"($\rho(z)$ / $V_\mathrm{ext}(z)$)");
    plt::title("Density and external field along z");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/density_profile.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/density_profile.png") << std::endl;
  }

  {
    // Plot 2: FFT power spectrum along kz (kx=ky=0)
    long nk = nz / 2 + 1;
    std::vector<double> k_idx(nk), fft_amp(nk);
    for (long iz = 0; iz < nk; ++iz) {
      k_idx[iz] = static_cast<double>(iz);
      fft_amp[iz] = std::abs(fourier[iz]) / static_cast<double>(ntot);
    }

    plt::figure_size(800, 550);
    plt::named_plot("Fourier amplitudes", k_idx, fft_amp, "ro");
    plt::xlim(-0.5, static_cast<double>(nk));
    plt::ylim(-0.02, rho0 + 0.1);
    plt::xlabel(R"($k_z$ mode index)");
    plt::ylabel(R"($|F(k)| / N$)");
    plt::title(R"(FFT power spectrum ($k_x = k_y = 0$))");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/fft_spectrum.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/fft_spectrum.png") << std::endl;
  }

  {
    // Plot 3: Species alias coordinate mapping
    int npts = 200;
    double rho_max = 2.0;
    std::vector<double> rho_vals(npts), alias_vals(npts);
    for (int i = 0; i < npts; ++i) {
      rho_vals[i] = Species::RHO_MIN + (rho_max - Species::RHO_MIN) * i / (npts - 1);
      alias_vals[i] = std::sqrt(rho_vals[i] - Species::RHO_MIN);
    }

    plt::figure_size(800, 550);
    plt::named_plot(R"($x(\rho)$)", rho_vals, alias_vals, "g-");
    plt::xlim(0.0, rho_max);
    plt::ylim(0.0, std::sqrt(rho_max) + 0.1);
    plt::xlabel(R"($\rho$)");
    plt::ylabel(R"($x = (\rho - \rho_\mathrm{min})^{1/2}$)");
    plt::title(R"(Species alias coordinate $x(\rho)$)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    plt::save("exports/alias_mapping.png");
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute("exports/alias_mapping.png") << std::endl;
  }
#endif

  return 0;
}
