#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace dft;

int main() {
#ifdef DOC_SOURCE_DIR
  std::filesystem::current_path(DOC_SOURCE_DIR);
#endif
  std::filesystem::create_directories("exports");

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

  std::cout << std::fixed << std::setprecision(6);

  // Define the Lennard-Jones system declaratively.

  physics::Model model{
      .grid = make_grid(0.1, {6.0, 6.0, 6.0}),
      .species = {Species{.name = "LJ", .hard_sphere_diameter = 1.0}},
      .interactions = {{
          .species_i = 0,
          .species_j = 0,
          .potential = physics::potentials::make_lennard_jones(1.0, 1.0, 2.5),
          .split = physics::potentials::SplitScheme::WeeksChandlerAndersen,
      }},
      .temperature = 1.0,
  };

  functionals::bulk::PhaseSearchConfig config{
      .rho_max = 1.0,
      .rho_scan_step = 0.005,
      .newton = {.max_iterations = 300, .tolerance = 1e-10},
  };

  std::cout << "=== LJ fluid phase diagram (mean-field DFT) ===\n"
            << "  sigma = 1.0, epsilon = 1.0, r_c = 2.5\n\n";

  // Temperature-dependent bulk weight factory.

  auto weight_factory = [&](const functionals::fmt::FMTModel& fmt_model, double kT) {
    return functionals::make_bulk_weights(fmt_model, model.interactions, kT);
  };

  // Pressure isotherms using arma::linspace for the density grid.

  std::cout << "=== Pressure isotherms P*(rho) [White Bear II] ===\n\n";

  auto wb2 = functionals::fmt::WhiteBearII{};
  std::vector<double> isotherm_temps = {0.6, 0.7, 0.8, 0.9, 1.0, 1.2};
  arma::vec rho_grid = arma::linspace(0.01, 1.0, 200);

  std::vector<std::vector<double>> iso_rho(isotherm_temps.size());
  std::vector<std::vector<double>> iso_p(isotherm_temps.size());

  for (std::size_t t = 0; t < isotherm_temps.size(); ++t) {
    double kT = isotherm_temps[t];
    auto w = weight_factory(wb2, kT);
    auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_grid);
    std::vector<double> p_vec(rho_grid.n_elem);
    for (arma::uword i = 0; i < rho_grid.n_elem; ++i) {
      p_vec[i] = functionals::bulk::pressure(arma::vec{rho_grid(i)}, model.species, w);
    }
    iso_rho[t] = std::move(rho_vec);
    iso_p[t] = std::move(p_vec);
    std::cout << "  T* = " << kT
              << ":  P*(" << iso_rho[t].front() << ") = " << iso_p[t].front()
              << ",  P*(" << iso_rho[t].back() << ") = " << iso_p[t].back() << "\n";
  }

  // Trace coexistence curve for each FMT model using the library's binodal()
  // function, which uses pseudo-arclength continuation internally.
  // The continuation parameterises the curve by arclength,
  // naturally handling the pitchfork bifurcation at the critical point.

  std::vector<std::pair<std::string, functionals::fmt::FMTModel>> fmt_models = {
      {"Rosenfeld", functionals::fmt::Rosenfeld{}},
      {"RSLT", functionals::fmt::RSLT{}},
      {"White Bear I", functionals::fmt::WhiteBearI{}},
      {"White Bear II", functionals::fmt::WhiteBearII{}},
  };

  functionals::bulk::PhaseDiagramConfig pd_config{
      .start_temperature = 0.6,
      .search = config,
  };

  std::cout << "\n=== Coexistence via pseudo-arclength continuation ===\n\n";

  std::vector<CoexData> all_coex;
  std::vector<SpinodalData> all_spin;
  for (const auto& [name, fmt_model] : fmt_models) {
    CoexData cd{.name = name};
    SpinodalData sd{.name = name};

    functionals::bulk::WeightFactory wf = [&](double kT) {
      return weight_factory(fmt_model, kT);
    };

    auto b = functionals::bulk::binodal(model.species, wf, pd_config);
    if (b) {
      cd.T = b->temperature;
      cd.rho_v = b->rho_vapor;
      cd.rho_l = b->rho_liquid;
      cd.Tc = b->critical_temperature;
      cd.rho_c = b->critical_density;
      std::cout << name << ": binodal " << cd.T.n_elem << " pts"
                << ", T_c ~ " << cd.Tc << ", rho_c ~ " << cd.rho_c;
    } else {
      std::cout << name << ": binodal failed";
    }

    auto s = functionals::bulk::spinodal(model.species, wf, pd_config);
    if (s) {
      sd.T = s->temperature;
      sd.rho_lo = s->rho_low;
      sd.rho_hi = s->rho_high;
      std::cout << "  |  spinodal " << sd.T.n_elem << " pts\n";
    } else {
      std::cout << "  |  spinodal failed\n";
    }

    all_coex.push_back(std::move(cd));
    all_spin.push_back(std::move(sd));
  }

  // Detailed White Bear II table.

  const auto& wb2_data = all_coex.back();
  std::cout << "\n--- White Bear II coexistence data ---\n";
  std::cout << std::setw(8) << "T*"
            << std::setw(14) << "rho_vapor"
            << std::setw(14) << "rho_liquid" << "\n";
  for (arma::uword i = 0; i < wb2_data.T.n_elem; ++i) {
    std::cout << std::setw(8) << wb2_data.T(i)
              << std::setw(14) << wb2_data.rho_v(i)
              << std::setw(14) << wb2_data.rho_l(i) << "\n";
  }

  // Spinodal curve for White Bear II.

  const auto& wb2_spin = all_spin.back();
  std::cout << "\n=== Spinodal curve (White Bear II) ===\n\n";
  if (!wb2_spin.T.is_empty()) {
    std::cout << "Spinodal: " << wb2_spin.T.n_elem << " points\n";

    std::cout << "\n--- White Bear II spinodal data ---\n";
    std::cout << std::setw(8) << "T*"
              << std::setw(14) << "rho_low"
              << std::setw(14) << "rho_high" << "\n";
    for (arma::uword i = 0; i < wb2_spin.T.n_elem; ++i) {
      std::cout << std::setw(8) << wb2_spin.T(i)
                << std::setw(14) << wb2_spin.rho_lo(i)
                << std::setw(14) << wb2_spin.rho_hi(i) << "\n";
    }
  }

  // Demonstrate spline interpolation on the phase diagram.

  std::cout << "\n=== Interpolated phase boundaries (White Bear II) ===\n\n";

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

  std::cout << std::setw(8) << "T*"
            << std::setw(14) << "rho_v(bin)"
            << std::setw(14) << "rho_l(bin)"
            << std::setw(14) << "rho_lo(sp)"
            << std::setw(14) << "rho_hi(sp)" << "\n";

  for (double T = 0.65; T <= 1.15; T += 0.1) {
    auto pb = functionals::bulk::interpolate(wb2_pd, T);
    std::cout << std::setw(8) << T
              << std::setw(14) << pb.binodal_vapor
              << std::setw(14) << pb.binodal_liquid
              << std::setw(14) << pb.spinodal_low
              << std::setw(14) << pb.spinodal_high << "\n";
  }

#ifdef DFT_HAS_MATPLOTLIB
  plot::isotherms(iso_rho, iso_p, isotherm_temps);
  plot::coexistence(all_coex, all_spin);
  plot::binodal(wb2_data);
  if (!wb2_spin.T.is_empty()) {
    dft::functionals::bulk::SpinodalCurve wb2_sp{
        .temperature = wb2_spin.T, .rho_low = wb2_spin.rho_lo, .rho_high = wb2_spin.rho_hi};
    plot::phase_diagram_plot(wb2_data, wb2_sp);
  }
#endif
}
