#include <classicaldft>
#include <armadillo>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

using namespace dft_core::io;

int main()
{
  std::filesystem::create_directories("exports");

  auto to_vec = [](const arma::vec& v) {
    return arma::conv_to<std::vector<double>>::from(v);
  };

  using namespace dft_core::physics::potentials::intermolecular;

  const int N = 200;
  auto x_arma = arma::linspace(0.75, 1.8, N);
  auto x = to_vec(x_arma);
  double kT = 1.0;

  auto lj = LennardJones();
  auto twf = tenWoldeFrenkel();
  auto wrdf = WangRamirezDobnikarFrenkel();

  // ── Compute potentials ──────────────────────────────────────────────

  auto v_lj = to_vec(lj.v_potential(x_arma));
  auto v_twf = to_vec(twf(x_arma));
  auto v_wrdf = to_vec(wrdf(x_arma));

  double d_lj = lj.find_hard_sphere_diameter(kT);
  double d_twf = twf.find_hard_sphere_diameter(kT);
  double d_wrdf = wrdf.find_hard_sphere_diameter(kT);

  auto att_lj = to_vec(lj.w_attractive(x_arma));
  auto rep_lj = to_vec(lj.w_repulsive(x_arma));
  auto att_twf = to_vec(twf.w_attractive(x_arma));
  auto rep_twf = to_vec(twf.w_repulsive(x_arma));
  auto att_wrdf = to_vec(wrdf.w_attractive(x_arma));
  auto rep_wrdf = to_vec(wrdf.w_repulsive(x_arma));

  console::info("LJ   hard-sphere diameter (kT = 1.0) = " + std::to_string(d_lj));
  console::info("tWF  hard-sphere diameter (kT = 1.0) = " + std::to_string(d_twf));
  console::info("WRDF hard-sphere diameter (kT = 1.0) = " + std::to_string(d_wrdf));

  // ── Plots ──────────────────────────────────────────────────────────────

#ifdef DFT_HAS_MATPLOTLIB
  namespace plt = matplotlibcpp;
  plt::backend("Agg");

  auto save_plot = [](const std::string& path) {
    plt::save(path);
    plt::close();
    std::cout << "Plot saved: " << std::filesystem::absolute(path) << std::endl;
  };

  // ── Individual potential: Lennard-Jones ──
  {
    plt::figure_size(800, 550);
    plt::named_plot(R"($v_\mathrm{LJ}(r)$)", x, v_lj, "k-");
    plt::named_plot("Minimum", std::vector<double>{lj.r_min()}, std::vector<double>{lj.v_min()}, "rs");
    plt::plot(std::vector<double>{d_lj, d_lj}, std::vector<double>{-2.0, 10.0}, {{"color", "grey"}, {"linestyle", "--"}, {"label", R"($d_\mathrm{HS}$)"}});
    plt::xlim(0.75, 1.8);
    plt::ylim(-2.0, 10.0);
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($v(r) / \epsilon$)");
    plt::title(R"(Lennard-Jones potential ($d_\mathrm{HS}$ = )" + std::to_string(d_lj).substr(0, 5) + ")");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    save_plot("exports/potential_lj.png");
  }

  // ── Individual potential: ten Wolde-Frenkel ──
  {
    plt::figure_size(800, 550);
    plt::named_plot(R"($v_\mathrm{tWF}(r)$)", x, v_twf, "b-");
    plt::named_plot("Minimum", std::vector<double>{twf.r_min()}, std::vector<double>{twf.v_min()}, "bD");
    plt::plot(std::vector<double>{d_twf, d_twf}, std::vector<double>{-2.0, 10.0}, {{"color", "grey"}, {"linestyle", "--"}, {"label", R"($d_\mathrm{HS}$)"}});
    plt::xlim(0.75, 1.8);
    plt::ylim(-2.0, 10.0);
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($v(r) / \epsilon$)");
    plt::title(R"(ten Wolde-Frenkel potential ($d_\mathrm{HS}$ = )" + std::to_string(d_twf).substr(0, 5) + ")");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    save_plot("exports/potential_twf.png");
  }

  // ── Individual potential: WRDF ──
  {
    plt::figure_size(800, 550);
    plt::named_plot(R"($v_\mathrm{WRDF}(r)$)", x, v_wrdf, "r-");
    plt::named_plot("Minimum", std::vector<double>{wrdf.r_min()}, std::vector<double>{wrdf.v_min()}, "rs");
    plt::plot(std::vector<double>{d_wrdf, d_wrdf}, std::vector<double>{-2.0, 10.0}, {{"color", "grey"}, {"linestyle", "--"}, {"label", R"($d_\mathrm{HS}$)"}});
    plt::xlim(0.75, 1.8);
    plt::ylim(-2.0, 10.0);
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($v(r) / \epsilon$)");
    plt::title(R"(WRDF potential ($d_\mathrm{HS}$ = )" + std::to_string(d_wrdf).substr(0, 5) + ")");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    save_plot("exports/potential_wrdf.png");
  }

  // ── Full potentials comparison ──
  {
    plt::figure_size(800, 550);
    plt::named_plot("Lennard-Jones", x, v_lj, "k-");
    plt::named_plot("ten Wolde-Frenkel", x, v_twf, "b-");
    plt::named_plot("WRDF", x, v_wrdf, "r-");
    plt::named_plot("LJ min", std::vector<double>{lj.r_min()}, std::vector<double>{lj.v_min()}, "ks");
    plt::named_plot("tWF min", std::vector<double>{twf.r_min()}, std::vector<double>{twf.v_min()}, "bD");
    plt::named_plot("WRDF min", std::vector<double>{wrdf.r_min()}, std::vector<double>{wrdf.v_min()}, "rs");
    plt::plot(std::vector<double>{d_lj, d_lj}, std::vector<double>{-2.0, 10.0}, "k--");
    plt::plot(std::vector<double>{d_twf, d_twf}, std::vector<double>{-2.0, 10.0}, "b--");
    plt::plot(std::vector<double>{d_wrdf, d_wrdf}, std::vector<double>{-2.0, 10.0}, "r--");
    plt::xlim(0.75, 1.8);
    plt::ylim(-2.0, 10.0);
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($v(r) / \epsilon$)");
    plt::title("Intermolecular potentials comparison");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    save_plot("exports/potentials_comparison.png");
  }

  // ── Individual perturbation: LJ ──
  {
    plt::figure_size(800, 550);
    plt::named_plot(R"($v_\mathrm{LJ}(r)$)", x, v_lj, "k-");
    plt::named_plot(R"($w_\mathrm{att}(r)$)", x, att_lj, "b-");
    plt::named_plot(R"($w_\mathrm{rep}(r)$)", x, rep_lj, "r--");
    plt::xlim(0.75, 1.8);
    plt::ylim(-2.0, 10.0);
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($w(r) / \epsilon$)");
    plt::title("LJ: WCA perturbation decomposition");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    save_plot("exports/perturbation_lj.png");
  }

  // ── Individual perturbation: tWF ──
  {
    plt::figure_size(800, 550);
    plt::named_plot(R"($v_\mathrm{tWF}(r)$)", x, v_twf, "b-");
    plt::named_plot(R"($w_\mathrm{att}(r)$)", x, att_twf, "c-");
    plt::named_plot(R"($w_\mathrm{rep}(r)$)", x, rep_twf, "m--");
    plt::xlim(0.75, 1.8);
    plt::ylim(-2.0, 10.0);
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($w(r) / \epsilon$)");
    plt::title("tWF: WCA perturbation decomposition");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    save_plot("exports/perturbation_twf.png");
  }

  // ── Individual perturbation: WRDF ──
  {
    plt::figure_size(800, 550);
    plt::named_plot(R"($v_\mathrm{WRDF}(r)$)", x, v_wrdf, "r-");
    plt::named_plot(R"($w_\mathrm{att}(r)$)", x, att_wrdf, "c-");
    plt::named_plot(R"($w_\mathrm{rep}(r)$)", x, rep_wrdf, "m--");
    plt::xlim(0.75, 1.8);
    plt::ylim(-2.0, 10.0);
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($w(r) / \epsilon$)");
    plt::title("WRDF: WCA perturbation decomposition");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    save_plot("exports/perturbation_wrdf.png");
  }

  // ── All perturbation decompositions ──
  {
    plt::figure_size(800, 550);
    plt::named_plot(R"(LJ $w_\mathrm{att}$)", x, att_lj, "k-");
    plt::named_plot(R"(LJ $w_\mathrm{rep}$)", x, rep_lj, "k--");
    plt::named_plot(R"(tWF $w_\mathrm{att}$)", x, att_twf, "b-");
    plt::named_plot(R"(tWF $w_\mathrm{rep}$)", x, rep_twf, "b--");
    plt::named_plot(R"(WRDF $w_\mathrm{att}$)", x, att_wrdf, "r-");
    plt::named_plot(R"(WRDF $w_\mathrm{rep}$)", x, rep_wrdf, "r--");
    plt::xlim(0.75, 1.8);
    plt::ylim(-2.0, 10.0);
    plt::xlabel(R"($r / \sigma$)");
    plt::ylabel(R"($w(r) / \epsilon$)");
    plt::title("Perturbation theory decomposition (all potentials)");
    plt::legend();
    plt::grid(true);
    plt::tight_layout();
    save_plot("exports/perturbation_decomposition.png");
  }
#endif

  return 0;
}