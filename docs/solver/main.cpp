#include "dft.hpp"
#include "plot.hpp"
#include "utils.hpp"

#include <filesystem>
#include <iostream>
#include <print>
#include <vector>

using namespace dft;
using utils::CoexData;
using utils::SpinodalData;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  // Define the Lennard-Jones system declaratively.

  physics::Model model{
    .grid = make_grid(0.1, { 6.0, 6.0, 6.0 }),
    .species = { Species{ .name = "LJ", .hard_sphere_diameter = 1.0 } },
    .interactions = { {
        .species_i = 0,
        .species_j = 0,
        .potential = physics::potentials::make_lennard_jones(1.0, 1.0, 2.5),
        .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
    } },
    .temperature = 1.0,
  };

  functionals::bulk::PhaseSearch config{
    .rho_max = 1.0,
    .rho_scan_step = 0.005,
    .newton = { .max_iterations = 300, .tolerance = 1e-10 },
  };

  std::println(std::cout, "=== LJ fluid phase diagram (mean-field DFT) ===");
  std::println(std::cout, "  sigma = 1.0, epsilon = 1.0, r_c = 2.5\n");

  // Pressure isotherms using arma::linspace for the density grid.

  std::println(std::cout, "=== Pressure isotherms P*(rho) [White Bear II] ===\n");

  std::vector<double> isotherm_temps = { 0.6, 0.7, 0.8, 0.9, 1.0, 1.2 };
  arma::vec rho_grid = arma::linspace(0.01, 1.0, 200);

  std::vector<std::vector<double>> iso_rho(isotherm_temps.size());
  std::vector<std::vector<double>> iso_p(isotherm_temps.size());

  auto wb2_eos_factory =
      functionals::bulk::make_eos_factory(functionals::fmt::WhiteBearII{}, model.species, model.interactions);

  for (std::size_t t = 0; t < isotherm_temps.size(); ++t) {
    double kT = isotherm_temps[t];
    auto eos = wb2_eos_factory(kT);
    auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_grid);
    std::vector<double> p_vec(rho_grid.n_elem);
    for (arma::uword i = 0; i < rho_grid.n_elem; ++i) {
      p_vec[i] = eos.pressure(arma::vec{ rho_grid(i) });
    }
    iso_rho[t] = std::move(rho_vec);
    iso_p[t] = std::move(p_vec);
    std::println(
        std::cout,
        "  T* = {}:  P*({}) = {},  P*({}) = {}",
        kT,
        iso_rho[t].front(),
        iso_p[t].front(),
        iso_rho[t].back(),
        iso_p[t].back()
    );
  }

  // Trace coexistence curve for each FMT model using the library's binodal()
  // function, which uses pseudo-arclength continuation internally.
  // The continuation parameterises the curve by arclength,
  // naturally handling the pitchfork bifurcation at the critical point.

  std::vector<std::pair<std::string, functionals::fmt::FMTModel>> fmt_models = {
    { "Rosenfeld", functionals::fmt::Rosenfeld{} },
    { "RSLT", functionals::fmt::RSLT{} },
    { "White Bear I", functionals::fmt::WhiteBearI{} },
    { "White Bear II", functionals::fmt::WhiteBearII{} },
  };

  functionals::bulk::PhaseDiagramBuilder pd_config{
    .start_temperature = 0.6,
    .search = config,
  };

  std::println(std::cout, "\n=== Coexistence via pseudo-arclength continuation ===\n");

  std::vector<CoexData> all_coex;
  std::vector<SpinodalData> all_spin;
  for (const auto& [name, fmt_model] : fmt_models) {
    CoexData cd{ .name = name };
    SpinodalData sd{ .name = name };

    auto eos_at = functionals::bulk::make_eos_factory(fmt_model, model.species, model.interactions);

    auto b = pd_config.binodal(eos_at);
    if (b) {
      cd.T = b->temperature;
      cd.rho_v = b->rho_vapor;
      cd.rho_l = b->rho_liquid;
      cd.Tc = b->critical_temperature;
      cd.rho_c = b->critical_density;
      std::print(std::cout, "{}: binodal {} pts, T_c ~ {}, rho_c ~ {}", name, cd.T.n_elem, cd.Tc, cd.rho_c);
    } else {
      std::print(std::cout, "{}: binodal failed", name);
    }

    auto s = pd_config.spinodal_curve(eos_at);
    if (s) {
      sd.T = s->temperature;
      sd.rho_lo = s->rho_low;
      sd.rho_hi = s->rho_high;
      std::println(std::cout, "  |  spinodal {} pts", sd.T.n_elem);
    } else {
      std::println(std::cout, "  |  spinodal failed");
    }

    all_coex.push_back(std::move(cd));
    all_spin.push_back(std::move(sd));
  }

  // Detailed White Bear II table.

  const auto& wb2_data = all_coex.back();
  std::println(std::cout, "\n--- White Bear II coexistence data ---");
  std::println(std::cout, "{:>8s}{:>14s}{:>14s}", "T*", "rho_vapor", "rho_liquid");
  for (arma::uword i = 0; i < wb2_data.T.n_elem; ++i) {
    std::println(std::cout, "{:>8.6f}{:>14.6f}{:>14.6f}", wb2_data.T(i), wb2_data.rho_v(i), wb2_data.rho_l(i));
  }

  // Spinodal curve for White Bear II.

  const auto& wb2_spin = all_spin.back();
  std::println(std::cout, "\n=== Spinodal curve (White Bear II) ===\n");
  if (!wb2_spin.T.is_empty()) {
    std::println(std::cout, "Spinodal: {} points", wb2_spin.T.n_elem);

    std::println(std::cout, "\n--- White Bear II spinodal data ---");
    std::println(std::cout, "{:>8s}{:>14s}{:>14s}", "T*", "rho_low", "rho_high");
    for (arma::uword i = 0; i < wb2_spin.T.n_elem; ++i) {
      std::println(std::cout, "{:>8.6f}{:>14.6f}{:>14.6f}", wb2_spin.T(i), wb2_spin.rho_lo(i), wb2_spin.rho_hi(i));
    }
  }

  // Demonstrate spline interpolation on the phase diagram.

  std::println(std::cout, "\n=== Interpolated phase boundaries (White Bear II) ===\n");

  functionals::bulk::PhaseDiagram wb2_pd{
      .binodal = {
          .temperature = wb2_data.T,
          .rho_vapor = wb2_data.rho_v,
          .rho_liquid = wb2_data.rho_l,
          .critical_temperature = wb2_data.Tc,
          .critical_density = wb2_data.rho_c,
      },
      .spinodal = {
          .temperature = wb2_spin.T,
          .rho_low = wb2_spin.rho_lo,
          .rho_high = wb2_spin.rho_hi,
      },
      .critical_temperature = wb2_data.Tc,
      .critical_density = wb2_data.rho_c,
  };

  std::println(
      std::cout,
      "{:>8s}{:>14s}{:>14s}{:>14s}{:>14s}",
      "T*",
      "rho_v(bin)",
      "rho_l(bin)",
      "rho_lo(sp)",
      "rho_hi(sp)"
  );

  for (double T = 0.65; T <= 1.15; T += 0.1) {
    auto pb = wb2_pd.interpolate(T);
    std::println(
        std::cout,
        "{:>8.2f}{:>14.6f}{:>14.6f}{:>14.6f}{:>14.6f}",
        T,
        pb.binodal_vapor,
        pb.binodal_liquid,
        pb.spinodal_low,
        pb.spinodal_high
    );
  }

  // Cross-validate: compute Jim's single-temperature coexistence and
  // spinodal at many temperatures up to T_c, and overlay on the
  // continuation curves.

  double Tc_wb2 = wb2_data.Tc > 0 ? wb2_data.Tc : 1.3;
  std::println(std::cout, "\n=== Jim's single-T coexistence & spinodal (scan + bisect) vs continuation ===\n");

  utils::JimCoexPoints jim_pts;
  utils::JimSpinodalPoints jim_sp;
  for (double kT = 0.60; kT <= Tc_wb2 + 0.02; kT += 0.01) {
    auto eos = wb2_eos_factory(kT);

    auto sp = config.find_spinodal(eos);
    if (sp) {
      jim_sp.T.push_back(kT);
      jim_sp.rho_lo.push_back(sp->rho_low);
      jim_sp.rho_hi.push_back(sp->rho_high);
    }

    auto coex = config.find_coexistence(eos);
    if (coex) {
      jim_pts.T.push_back(kT);
      jim_pts.rho_v.push_back(coex->rho_vapor);
      jim_pts.rho_l.push_back(coex->rho_liquid);
      std::println(
          std::cout,
          "  T* = {:6.4f}:  rho_v = {:10.6f},  rho_l = {:10.6f}",
          kT,
          coex->rho_vapor,
          coex->rho_liquid
      );
    } else {
      std::println(std::cout, "  T* = {:6.4f}:  coexistence not found (near/above T_c)", kT);
    }
  }

  std::println(std::cout, "\n  Jim coexistence points: {}, spinodal points: {}", jim_pts.T.size(), jim_sp.T.size());

  // Collect spinodal data for plots.

  dft::functionals::bulk::SpinodalCurve wb2_sp{ .temperature = wb2_spin.T,
                                                .rho_low = wb2_spin.rho_lo,
                                                .rho_high = wb2_spin.rho_hi };

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(iso_rho, iso_p, isotherm_temps, all_coex, all_spin, wb2_data, wb2_sp, jim_pts, jim_sp);
#endif
}
