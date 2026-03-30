#include "dft.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
#endif

using namespace dft;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Build a single-component LJ solver at temperature kT.
/// The LJ potential must outlive the returned Solver.
static Solver make_lj_solver(
    double dx, const arma::rowvec3& box, double diameter,
    const potentials::LennardJones& lj, double kT,
    functional::fmt::FMTModel model) {
  Solver s;
  auto dens = density::Density(dx, box);
  dens.values().fill(0.1);
  auto sp = std::make_unique<functional::fmt::Species>(std::move(dens), diameter);
  auto& sp_ref = *sp;
  s.add_species(std::move(sp));
  s.add_interaction(
      std::make_unique<functional::interaction::Interaction>(sp_ref, sp_ref, lj, kT));
  s.set_fmt(std::make_unique<functional::fmt::FMT>(std::move(model)));
  return s;
}

/// Extract the radial density profile from a 3D cubic solver.
/// Averages over a spherical shell at each radius from the box centre.
static void extract_radial_profile(
    const Solver& solver,
    std::vector<double>& r_out, std::vector<double>& rho_out) {
  const auto& rho = solver.density(0).values();
  double dx = solver.density(0).dx();
  const auto& shape = solver.density(0).shape();
  const auto& box = solver.density(0).box_size();

  long nx = shape[0];
  long ny = shape[1];
  long nz = shape[2];
  double cx = box(0) / 2.0;
  double cy = box(1) / 2.0;
  double cz = box(2) / 2.0;

  // Bin by radial distance
  double r_max = std::min({cx, cy, cz});
  int n_bins = static_cast<int>(r_max / dx);
  std::vector<double> sum(static_cast<size_t>(n_bins), 0.0);
  std::vector<int> count(static_cast<size_t>(n_bins), 0);

  for (long ix = 0; ix < nx; ++ix) {
    double x = ix * dx - cx;
    for (long iy = 0; iy < ny; ++iy) {
      double y = iy * dx - cy;
      for (long iz = 0; iz < nz; ++iz) {
        double z = iz * dx - cz;
        double r = std::sqrt(x * x + y * y + z * z);
        int bin = static_cast<int>(r / dx);
        if (bin < n_bins) {
          auto idx = static_cast<arma::uword>(ix * ny * nz + iy * nz + iz);
          sum[static_cast<size_t>(bin)] += rho(idx);
          count[static_cast<size_t>(bin)] += 1;
        }
      }
    }
  }

  r_out.clear();
  rho_out.clear();
  for (int i = 0; i < n_bins; ++i) {
    if (count[static_cast<size_t>(i)] > 0) {
      r_out.push_back((i + 0.5) * dx);
      rho_out.push_back(sum[static_cast<size_t>(i)] / count[static_cast<size_t>(i)]);
    }
  }
}

/// Initialise density with a smooth droplet (tanh interface) centred in the box.
static void init_droplet(Solver& solver, double rho_v, double rho_l,
                         double r_droplet, double interface_width) {
  auto& dens = solver.species(0).density();
  arma::vec rho(dens.size());
  double dx = dens.dx();
  const auto& shape = dens.shape();
  const auto& box = dens.box_size();

  long nx = shape[0];
  long ny = shape[1];
  long nz = shape[2];
  double cx = box(0) / 2.0;
  double cy = box(1) / 2.0;
  double cz = box(2) / 2.0;

  for (long ix = 0; ix < nx; ++ix) {
    double x = ix * dx - cx;
    for (long iy = 0; iy < ny; ++iy) {
      double y = iy * dx - cy;
      for (long iz = 0; iz < nz; ++iz) {
        double z = iz * dx - cz;
        double r = std::sqrt(x * x + y * y + z * z);
        auto idx = static_cast<arma::uword>(ix * ny * nz + iy * nz + iz);

        // tanh profile: liquid inside, vapor outside
        rho(idx) = 0.5 * (rho_l + rho_v)
                   - 0.5 * (rho_l - rho_v) * std::tanh((r - r_droplet) / interface_width);
      }
    }
  }
  dens.set(rho);
}

// ── Demo 1: Equilibrium planar interface ─────────────────────────────────

static void demo_planar_interface(const potentials::LennardJones& lj) {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  Demo 1: Planar liquid-vapor interface (1D slab)\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double kT = 0.80;
  double dx = 0.5;
  arma::rowvec3 box = {6.0, 6.0, 12.0};

  auto solver = make_lj_solver(dx, box, 1.0, lj, kT, functional::fmt::WhiteBearII{});

  // Find coexistence densities
  double rho_v = 0.0;
  double rho_l = 0.0;
  solver.find_coexistence(1.1, 0.005, rho_v, rho_l, 1e-8);
  double mu_coex = solver.chemical_potential(rho_v);

  std::cout << "T* = " << kT << "\n"
            << "rho_vapor  = " << rho_v << "\n"
            << "rho_liquid = " << rho_l << "\n"
            << "mu_coex    = " << mu_coex << "\n\n";

  // Set chemical potential to coexistence
  solver.species(0).set_chemical_potential(mu_coex);

  // Initialise as slab: liquid in the centre, vapor on sides.
  // The box extends 12σ in z; we place the liquid slab in the middle 6σ.
  auto& dens = solver.species(0).density();
  arma::vec rho(dens.size());
  const auto& shape = dens.shape();
  long nx = shape[0];
  long ny = shape[1];
  long nz = shape[2];
  double lz = box(2);
  double z_centre = lz / 2.0;
  double slab_half = lz / 4.0;
  double w = 1.0;   // tanh interface width in units of sigma

  for (long ix = 0; ix < nx; ++ix) {
    for (long iy = 0; iy < ny; ++iy) {
      for (long iz = 0; iz < nz; ++iz) {
        auto idx = static_cast<arma::uword>(ix * ny * nz + iy * nz + iz);
        double z = iz * dx;
        double left_face = z_centre - slab_half;
        double right_face = z_centre + slab_half;
        rho(idx) = rho_v + (rho_l - rho_v)
                   * 0.5 * (std::tanh((z - left_face) / w) - std::tanh((z - right_face) / w));
      }
    }
  }
  dens.set(rho);

  // Minimise with FIRE2
  dynamics::Fire2Config config{
      .dt = 1e-4, .dt_max = 0.005, .force_limit = 5e-3, .min_density = 1e-30};
  dynamics::Fire2Minimizer fire(solver, config);

  fire.set_step_callback([](long step, double energy, double max_force) {
    if (step % 500 == 0) {
      std::cout << "  Step " << std::setw(5) << step
                << "  E = " << std::scientific << std::setprecision(6) << energy
                << "  |F|_max = " << max_force << "\n";
    }
    return true;
  });

  bool converged = fire.run(2000);
  std::cout << "\nConverged: " << std::boolalpha << converged
            << " after " << fire.step_count() << " steps\n";

  // Extract profile along z (average over x, y)
  std::vector<double> z_vals, rho_vals;
  const arma::vec& rho_final = solver.density(0).values();
  long nxy = nx * ny;
  for (long iz = 0; iz < nz; ++iz) {
    double avg = 0.0;
    for (long ix = 0; ix < nx; ++ix) {
      for (long iy = 0; iy < ny; ++iy) {
        auto idx = static_cast<arma::uword>(ix * ny * nz + iy * nz + iz);
        avg += rho_final(idx);
      }
    }
    z_vals.push_back(iz * dx);
    rho_vals.push_back(avg / static_cast<double>(nxy));
  }

  // Print near the interface
  std::cout << "\nDensity profile near left interface:\n";
  std::cout << std::setw(8) << "z" << std::setw(14) << "rho(z)\n";
  for (size_t i = 0; i < z_vals.size(); i += 5) {
    std::cout << std::fixed << std::setprecision(3) << std::setw(8) << z_vals[i]
              << std::setw(14) << std::setprecision(6) << rho_vals[i] << "\n";
  }

#ifdef DFT_HAS_MATPLOTLIB
  plt::figure_size(800, 500);
  plt::named_plot(R"($\rho(z)$)", z_vals, rho_vals, "b-");
  plt::plot({0.0, lz}, {rho_v, rho_v}, {{"color", "gray"}, {"linestyle", ":"}, {"label", R"($\rho_v$)"}});
  plt::plot({0.0, lz}, {rho_l, rho_l}, {{"color", "gray"}, {"linestyle", "--"}, {"label", R"($\rho_l$)"}});
  plt::xlabel(R"($z / \sigma$)");
  plt::ylabel(R"($\rho \sigma^3$)");
  plt::title(R"(Planar liquid-vapor interface ($T^* = 0.80$))");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/planar_interface.png");
  std::cout << "\nPlot: " << std::filesystem::absolute("exports/planar_interface.png") << "\n";
  plt::close();
#endif
}

// ── Demo 2: Liquid droplet nucleation ────────────────────────────────────

static void demo_droplet(const potentials::LennardJones& lj) {
  std::cout << "\n══════════════════════════════════════════════════════════════\n"
            << "  Demo 2: Liquid droplet in metastable vapor (3D)\n"
            << "══════════════════════════════════════════════════════════════\n\n";

  double kT = 0.80;
  double dx = 0.5;
  arma::rowvec3 box = {8.0, 8.0, 8.0};

  auto solver = make_lj_solver(dx, box, 1.0, lj, kT, functional::fmt::WhiteBearII{});

  // Find coexistence
  double rho_v = 0.0;
  double rho_l = 0.0;
  solver.find_coexistence(1.1, 0.005, rho_v, rho_l, 1e-8);
  double mu_coex = solver.chemical_potential(rho_v);

  std::cout << "T* = " << kT << "\n"
            << "rho_vapor  = " << std::setprecision(6) << rho_v << "\n"
            << "rho_liquid = " << rho_l << "\n"
            << "mu_coex    = " << mu_coex << "\n\n";

  // Set chemical potential slightly above coexistence (supersaturated vapor).
  // This stabilises the droplet: the vapor wants to condense.
  double delta_mu = 0.01;
  solver.species(0).set_chemical_potential(mu_coex + delta_mu);

  // Initialise with a liquid droplet of radius ~1.5 sigma
  double r_droplet = 1.5;
  double interface_width = 0.8;
  init_droplet(solver, rho_v, rho_l, r_droplet, interface_width);

  // Extract initial profile
  std::vector<double> r_init, rho_init;
  extract_radial_profile(solver, r_init, rho_init);

  std::cout << "Initial droplet: R = " << r_droplet
            << ", interface width = " << interface_width << "\n"
            << "Supersaturation: delta_mu = " << delta_mu << "\n\n";

  // Minimise with FIRE2
  dynamics::Fire2Config config{
      .dt = 1e-3, .dt_max = 0.01, .force_limit = 5e-3, .min_density = 1e-30};
  dynamics::Fire2Minimizer fire(solver, config);

  std::vector<double> step_log, energy_log, force_log;
  fire.set_step_callback([&](long step, double energy, double max_force) {
    if (step % 50 == 0 || step <= 5) {
      step_log.push_back(static_cast<double>(step));
      energy_log.push_back(energy);
      force_log.push_back(max_force);
    }
    if (step % 100 == 0) {
      std::cout << "  Step " << std::setw(5) << step
                << "  E = " << std::scientific << std::setprecision(6) << energy
                << "  |F|_max = " << max_force << "\n";
    }
    return true;
  });

  bool converged = fire.run(500);
  std::cout << "\nConverged: " << std::boolalpha << converged
            << " after " << fire.step_count() << " steps\n";

  // Extract final profile
  std::vector<double> r_final, rho_final;
  extract_radial_profile(solver, r_final, rho_final);

  // Print profile
  std::cout << "\nRadial density profile:\n";
  std::cout << std::setw(8) << "r" << std::setw(14) << "rho_init" << std::setw(14) << "rho_final\n";
  std::cout << std::string(36, '-') << "\n";
  for (size_t i = 0; i < r_final.size(); i += 2) {
    double ri = (i < rho_init.size()) ? rho_init[i] : 0.0;
    std::cout << std::fixed << std::setprecision(2) << std::setw(8) << r_final[i]
              << std::setprecision(6) << std::setw(14) << ri
              << std::setw(14) << rho_final[i] << "\n";
  }

  // Compute excess grand potential (nucleation barrier)
  double omega_uniform = solver.grand_potential_density(rho_v);
  double v_box = box(0) * box(1) * box(2);
  double omega_droplet = fire.energy();
  double omega_excess = omega_droplet - omega_uniform * v_box;
  std::cout << "\nExcess grand potential (nucleation barrier):\n"
            << "  Omega_drop  = " << std::scientific << omega_droplet << "\n"
            << "  Omega_unif  = " << omega_uniform * v_box << "\n"
            << "  Delta_Omega = " << omega_excess << "\n";

#ifdef DFT_HAS_MATPLOTLIB
  // Plot 1: radial density profiles
  plt::figure_size(800, 500);
  plt::named_plot("Initial", r_init, rho_init, "b--");
  plt::named_plot("Minimised", r_final, rho_final, "r-");
  plt::plot({0.0, r_final.back()}, {rho_v, rho_v},
            {{"color", "gray"}, {"linestyle", ":"}, {"label", R"($\rho_v$)"}});
  plt::plot({0.0, r_final.back()}, {rho_l, rho_l},
            {{"color", "gray"}, {"linestyle", "--"}, {"label", R"($\rho_l$)"}});
  plt::xlabel(R"($r / \sigma$)");
  plt::ylabel(R"($\rho \sigma^3$)");
  plt::title(R"(Liquid droplet: radial density profile ($T^* = 0.80$))");
  plt::legend();
  plt::grid(true);
  plt::tight_layout();
  plt::save("exports/droplet_profile.png");
  std::cout << "\nPlot: " << std::filesystem::absolute("exports/droplet_profile.png") << "\n";
  plt::close();

  // Plot 2: convergence
  plt::figure_size(700, 500);
  plt::named_plot("Energy", step_log, energy_log, "b-");
  plt::xlabel("Step");
  plt::ylabel("Free energy");
  plt::title("Droplet minimisation: energy convergence");
  plt::legend();
  plt::tight_layout();
  plt::save("exports/droplet_convergence.png");
  std::cout << "Plot: " << std::filesystem::absolute("exports/droplet_convergence.png") << "\n";
  plt::close();
#endif
}

// ── Main ─────────────────────────────────────────────────────────────────

int main() {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

  constexpr double sigma = 1.0;
  constexpr double epsilon = 1.0;
  constexpr double r_cutoff = 2.5;
  potentials::LennardJones lj(sigma, epsilon, r_cutoff);

  std::cout << "=== Droplet nucleation example ===\n"
            << "  LJ potential: sigma = " << sigma << ", epsilon = " << epsilon
            << ", r_c = " << r_cutoff << "\n";

  demo_planar_interface(lj);
  demo_droplet(lj);

  std::cout << "\nAll droplet demos completed.\n";
  return 0;
}
