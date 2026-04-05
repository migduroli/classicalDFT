#include "dft.hpp"
#include "plot.hpp"

#include <filesystem>
#include <iostream>
#include <print>
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
  std::println(std::cout, "  sigma = 1.0, epsilon = 1.0, r_cutoff = 2.5");

  // Attractive potential profile: WCA vs BH splitting.

  std::println(std::cout, "\n=== Attractive tail: WCA vs BH splitting ===\n");

  constexpr int N = 200;
  arma::vec r_arma = arma::linspace(0.85, 2.6, N);
  arma::vec v_full_arma(N), watt_wca_arma(N), watt_bh_arma(N);
  for (arma::uword i = 0; i < r_arma.n_elem; ++i) {
    v_full_arma(i) = pot.energy(r_arma(i));
    watt_wca_arma(i) = pot.attractive(r_arma(i), physics::potentials::SplitScheme::WeeksChandlerAndersen);
    watt_bh_arma(i) = pot.attractive(r_arma(i), physics::potentials::SplitScheme::BarkerHenderson);
  }
  auto r_vec = arma::conv_to<std::vector<double>>::from(r_arma);
  auto v_full = arma::conv_to<std::vector<double>>::from(v_full_arma);
  auto watt_wca = arma::conv_to<std::vector<double>>::from(watt_wca_arma);
  auto watt_bh = arma::conv_to<std::vector<double>>::from(watt_bh_arma);

  std::println(std::cout, "{:>8s}{:>16s}{:>16s}{:>16s}", "r", "v(r)", "w_att(WCA)", "w_att(BH)");
  for (int i = 0; i < N; i += 40) {
    std::println(std::cout, "{:>8.6f}{:>16.6f}{:>16.6f}{:>16.6f}",
                 r_vec[i], v_full[i], watt_wca[i], watt_bh[i]);
  }

  // Van der Waals parameter (analytical).

  double a_vdw = 2.0 * pot.vdw_integral(model.temperature, split);
  console::info("Van der Waals parameter a_vdw (analytical, kT=1)");
  std::println(std::cout, "  a_vdw = {:.6f}", a_vdw);

  // Bulk thermodynamics from the Model.

  std::println(std::cout, "\n=== Bulk thermodynamics f(rho), mu(rho) at kT=1 ===\n");

  auto weights = functionals::make_bulk_weights(
      functionals::fmt::WhiteBearII{}, model.interactions, model.temperature
  );

  constexpr int M = 100;
  arma::vec rho_arma = arma::linspace(0.01, 1.0, M);
  arma::vec f_arma(M), mu_arma(M);

  std::println(std::cout, "{:>10s}{:>16s}{:>16s}", "rho", "f_mf", "mu_mf");

  for (arma::uword i = 0; i < rho_arma.n_elem; ++i) {
    f_arma(i) = functionals::bulk::mean_field::free_energy_density(weights.mean_field, arma::vec{rho_arma(i)});
    mu_arma(i) = functionals::bulk::mean_field::chemical_potential(weights.mean_field, arma::vec{rho_arma(i)}, 0);

    if (i % 20 == 0) {
      std::println(std::cout, "{:>10.6f}{:>16.6f}{:>16.6f}", rho_arma(i), f_arma(i), mu_arma(i));
    }
  }
  auto rho_vec = arma::conv_to<std::vector<double>>::from(rho_arma);
  auto f_vec = arma::conv_to<std::vector<double>>::from(f_arma);
  auto mu_vec = arma::conv_to<std::vector<double>>::from(mu_arma);

  // Grid convergence of a_vdw vs dx.

  std::println(std::cout, "\n=== Grid convergence: a_vdw vs dx (kT=1) ===\n");
  double a_analytic = 2.0 * pot.vdw_integral(model.temperature, split);
  std::println(std::cout, "Analytic a_vdw = {:.6f}\n", a_analytic);

  std::vector<double> dx_vals = {0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1};
  std::println(std::cout, "{:>8s}{:>16s}{:>16s}", "dx", "a_vdw", "rel. error");

  std::vector<double> a_conv(dx_vals.size()), err_conv(dx_vals.size());
  for (std::size_t i = 0; i < dx_vals.size(); ++i) {
    double dxi = dx_vals[i];
    double L = std::ceil(std::max(5.0, 2.0 * 2.5 + 2.0 * dxi) / dxi) * dxi;
    auto grid = make_grid(dxi, {L, L, L});

    auto mf_weights = functionals::make_mean_field_weights(grid, model.interactions, model.temperature);
    a_conv[i] = mf_weights.interactions[0].a_vdw;
    err_conv[i] = std::abs((a_conv[i] - a_analytic) / a_analytic);

    std::println(std::cout, "{:>8.3f}{:>16.6f}{:>16.6e}", dxi, a_conv[i], err_conv[i]);
  }

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(r_vec, v_full, watt_wca, watt_bh, rho_vec, f_vec, dx_vals, a_conv, a_analytic);
#endif
}
