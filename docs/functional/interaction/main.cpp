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
  std::cout << std::fixed << std::setprecision(6);

#ifdef DFT_HAS_MATPLOTLIB
  matplotlibcpp::backend("Agg");
#endif

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

  const auto& pot = model.interactions[0].potential;
  const auto split = model.interactions[0].split;

  console::info("Lennard-Jones interaction parameters");
  std::cout << "  sigma = 1.0, epsilon = 1.0, r_cutoff = 2.5\n";

  // Attractive potential profile: WCA vs BH splitting.

  std::cout << "\n=== Attractive tail: WCA vs BH splitting ===\n\n";

  constexpr int N = 200;
  arma::vec r_arma = arma::linspace(0.85, 2.6, N);
  arma::vec v_full_arma(N), watt_wca_arma(N), watt_bh_arma(N);
  for (arma::uword i = 0; i < r_arma.n_elem; ++i) {
    v_full_arma(i) = physics::potentials::energy(pot, r_arma(i));
    watt_wca_arma(i) = physics::potentials::attractive(pot, r_arma(i), physics::potentials::SplitScheme::WeeksChandlerAndersen);
    watt_bh_arma(i) = physics::potentials::attractive(pot, r_arma(i), physics::potentials::SplitScheme::BarkerHenderson);
  }
  auto r_vec = arma::conv_to<std::vector<double>>::from(r_arma);
  auto v_full = arma::conv_to<std::vector<double>>::from(v_full_arma);
  auto watt_wca = arma::conv_to<std::vector<double>>::from(watt_wca_arma);
  auto watt_bh = arma::conv_to<std::vector<double>>::from(watt_bh_arma);

  std::cout << std::setw(8) << "r" << std::setw(16) << "v(r)"
            << std::setw(16) << "w_att(WCA)" << std::setw(16) << "w_att(BH)" << "\n";
  for (int i = 0; i < N; i += 40) {
    std::cout << std::setw(8) << r_vec[i]
              << std::setw(16) << v_full[i]
              << std::setw(16) << watt_wca[i]
              << std::setw(16) << watt_bh[i]
              << "\n";
  }

  // Van der Waals parameter (analytical).

  double a_vdw = 2.0 * physics::potentials::vdw_integral(pot, model.temperature, split);
  console::info("Van der Waals parameter a_vdw (analytical, kT=1)");
  std::cout << "  a_vdw = " << a_vdw << "\n";

  // Bulk thermodynamics from the Model.

  std::cout << "\n=== Bulk thermodynamics f(rho), mu(rho) at kT=1 ===\n\n";

  auto weights = functionals::make_bulk_weights(
      functionals::fmt::WhiteBearII{}, model.interactions, model.temperature
  );

  constexpr int M = 100;
  arma::vec rho_arma = arma::linspace(0.01, 1.0, M);
  arma::vec f_arma(M), mu_arma(M);

  std::cout << std::setw(10) << "rho" << std::setw(16) << "f_mf"
            << std::setw(16) << "mu_mf" << "\n";

  for (arma::uword i = 0; i < rho_arma.n_elem; ++i) {
    f_arma(i) = functionals::bulk::mean_field_free_energy_density(weights.mean_field, arma::vec{rho_arma(i)});
    mu_arma(i) = functionals::bulk::mean_field_chemical_potential(weights.mean_field, arma::vec{rho_arma(i)}, 0);

    if (i % 20 == 0) {
      std::cout << std::setw(10) << rho_arma(i)
                << std::setw(16) << f_arma(i)
                << std::setw(16) << mu_arma(i)
                << "\n";
    }
  }
  auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_arma);
  auto f_vec = arma::conv_to<std::vector<double>>::from(f_arma);
  auto mu_vec = arma::conv_to<std::vector<double>>::from(mu_arma);

  // Grid convergence of a_vdw vs dx.

  std::cout << "\n=== Grid convergence: a_vdw vs dx (kT=1) ===\n\n";
  double a_analytic = 2.0 * physics::potentials::vdw_integral(pot, model.temperature, split);
  std::cout << "Analytic a_vdw = " << a_analytic << "\n\n";

  std::vector<double> dx_vals = {0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1};
  std::cout << std::setw(8) << "dx" << std::setw(16) << "a_vdw"
            << std::setw(16) << "rel. error" << "\n";

  std::vector<double> a_conv(dx_vals.size()), err_conv(dx_vals.size());
  for (std::size_t i = 0; i < dx_vals.size(); ++i) {
    double dxi = dx_vals[i];
    double L = std::ceil(std::max(5.0, 2.0 * 2.5 + 2.0 * dxi) / dxi) * dxi;
    auto grid = make_grid(dxi, {L, L, L});

    auto mf_weights = functionals::make_mean_field_weights(grid, model.interactions, model.temperature);
    a_conv[i] = mf_weights.interactions[0].a_vdw;
    err_conv[i] = std::abs((a_conv[i] - a_analytic) / a_analytic);

    std::cout << std::setw(8) << dxi
              << std::setw(16) << a_conv[i]
              << std::setw(16) << err_conv[i]
              << "\n";
  }

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  plot::wca_bh_splitting(r_vec, v_full, watt_wca, watt_bh);
  plot::mean_field_free_energy(rho_vec, f_vec);
  plot::grid_convergence(dx_vals, a_conv, a_analytic);
#endif
}
