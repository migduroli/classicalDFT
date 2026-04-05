#include "dft.hpp"
#include "plot.hpp"
#include "utils.hpp"

#include <armadillo>
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

  constexpr int N = 500;
  auto r_arma = arma::linspace(0.85, 2.6, N);
  auto r = utils::to_vec(r_arma);
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

  double d_lj = plj.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen);
  double d_twf = ptwf.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen);
  double d_wrdf = pwrdf.hard_sphere_diameter(kT, pot::SplitScheme::WeeksChandlerAndersen);

  console::info("Hard-sphere diameters (kT = 1.0)");
  std::println(std::cout, "  LJ:   d_HS = {:.6f}", d_lj);
  std::println(std::cout, "  tWF:  d_HS = {:.6f}", d_twf);
  std::println(std::cout, "  WRDF: d_HS = {:.6f}", d_wrdf);

  // Van der Waals integral.

  double a_lj = plj.vdw_integral(kT, pot::SplitScheme::WeeksChandlerAndersen);
  console::info("Van der Waals integral (LJ, kT=1)");
  std::println(std::cout, "  a_vdw = {:.6f}", a_lj);

#ifdef DFT_HAS_MATPLOTLIB
  plot::make_plots(r, v_lj, v_twf, v_wrdf, att_lj, rep_lj, lj.r_min, lj.v_min, d_lj);
#endif
}
