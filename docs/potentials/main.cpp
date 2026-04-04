#include "dft.hpp"
#include "plot.hpp"

#include <armadillo>
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

  auto to_vec = [](const arma::vec& v) {
    return arma::conv_to<std::vector<double>>::from(v);
  };

  constexpr int N = 500;
  auto r_arma = arma::linspace(0.85, 2.6, N);
  auto r = to_vec(r_arma);
  double kT = 1.0;

  // Create potentials via factories.

  namespace pot = physics::potentials;

  auto lj = pot::make_lennard_jones(1.0, 1.0, 2.5);
  auto twf = pot::make_ten_wolde_frenkel(1.0, 1.0, -1.0);
  auto wrdf = pot::make_wang_ramirez_dobnikar_frenkel(1.0, 1.0, 3.0);

  pot::Potential plj = lj;
  pot::Potential ptwf = twf;
  pot::Potential pwrdf = wrdf;

  // Evaluate potentials using arma::vec grid.

  arma::vec v_lj_a(N), v_twf_a(N), v_wrdf_a(N);
  arma::vec att_lj_a(N), rep_lj_a(N);
  for (arma::uword i = 0; i < r_arma.n_elem; ++i) {
    v_lj_a(i) = pot::energy(plj, r_arma(i));
    v_twf_a(i) = pot::energy(ptwf, r_arma(i));
    v_wrdf_a(i) = pot::energy(pwrdf, r_arma(i));
    att_lj_a(i) = pot::attractive(plj, r_arma(i), pot::SplitScheme::WeeksChandlerAndersen);
    rep_lj_a(i) = pot::repulsive(plj, r_arma(i), pot::SplitScheme::WeeksChandlerAndersen);
  }
  auto v_lj = arma::conv_to<std::vector<double>>::from(v_lj_a);
  auto v_twf = arma::conv_to<std::vector<double>>::from(v_twf_a);
  auto v_wrdf = arma::conv_to<std::vector<double>>::from(v_wrdf_a);
  auto att_lj = arma::conv_to<std::vector<double>>::from(att_lj_a);
  auto rep_lj = arma::conv_to<std::vector<double>>::from(rep_lj_a);

  // Hard-sphere diameters (Barker-Henderson integral).

  double d_lj = pot::hard_sphere_diameter(plj, kT, pot::SplitScheme::WeeksChandlerAndersen);
  double d_twf = pot::hard_sphere_diameter(ptwf, kT, pot::SplitScheme::WeeksChandlerAndersen);
  double d_wrdf = pot::hard_sphere_diameter(pwrdf, kT, pot::SplitScheme::WeeksChandlerAndersen);

  std::cout << std::setprecision(6);
  console::info("Hard-sphere diameters (kT = 1.0)");
  std::cout << "  LJ:   d_HS = " << d_lj << "\n";
  std::cout << "  tWF:  d_HS = " << d_twf << "\n";
  std::cout << "  WRDF: d_HS = " << d_wrdf << "\n";

  // Van der Waals integral.

  double a_lj = pot::vdw_integral(plj, kT, pot::SplitScheme::WeeksChandlerAndersen);
  console::info("Van der Waals integral (LJ, kT=1)");
  std::cout << "  a_vdw = " << a_lj << "\n";

  // Plots.

#ifdef DFT_HAS_MATPLOTLIB
  plot::potentials_comparison(r, v_lj, v_twf, v_wrdf);
  plot::perturbation_decomposition(r, v_lj, att_lj, rep_lj);
  plot::potential_with_dhs(r, v_lj, lj.r_min, lj.v_min, d_lj);
#endif
}
