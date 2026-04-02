#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace dft;

int main() {
#ifdef EXAMPLE_SOURCE_DIR
  std::filesystem::current_path(EXAMPLE_SOURCE_DIR);
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
  for (const auto& [name, fmt_model] : fmt_models) {
    CoexData cd{.name = name};

    functionals::bulk::WeightFactory wf = [&](double kT) {
      return weight_factory(fmt_model, kT);
    };

    auto curve = functionals::bulk::binodal(model.species, wf, pd_config);

    if (curve) {
      cd.T = curve->temperature;
      cd.rho_v = curve->rho_vapor;
      cd.rho_l = curve->rho_liquid;
      cd.Tc = curve->critical_temperature;
      cd.rho_c = curve->critical_density;
      std::cout << name << ": " << cd.T.n_elem << " continuation points"
                << ", T_c ~ " << cd.Tc << ", rho_c ~ " << cd.rho_c << "\n";
    } else {
      std::cout << name << ": failed to find coexistence\n";
    }

    all_coex.push_back(std::move(cd));
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

  std::cout << "\n=== Spinodal curve (White Bear II) ===\n\n";

  functionals::bulk::WeightFactory wb2_wf = [&](double kT) {
    return weight_factory(wb2, kT);
  };

  auto sp = functionals::bulk::spinodal(model.species, wb2_wf, pd_config);
  if (sp) {
    std::cout << "Spinodal: " << sp->temperature.n_elem << " points"
              << ", T_c ~ " << sp->critical_temperature
              << ", rho_c ~ " << sp->critical_density << "\n";

    std::cout << "\n--- White Bear II spinodal data ---\n";
    std::cout << std::setw(8) << "T*"
              << std::setw(14) << "rho_low"
              << std::setw(14) << "rho_high" << "\n";
    for (arma::uword i = 0; i < sp->temperature.n_elem; ++i) {
      std::cout << std::setw(8) << sp->temperature(i)
                << std::setw(14) << sp->rho_low(i)
                << std::setw(14) << sp->rho_high(i) << "\n";
    }
  } else {
    std::cout << "Spinodal: failed\n";
  }

#ifdef DFT_HAS_MATPLOTLIB
  plot::isotherms(iso_rho, iso_p, isotherm_temps);
  plot::coexistence(all_coex);
  plot::binodal(wb2_data);
  if (sp) {
    plot::phase_diagram_plot(wb2_data, *sp);
  }
#endif
}
