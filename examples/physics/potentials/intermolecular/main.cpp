#include "dft_lib/dft_lib.h"
#include <armadillo>

/// A convenient wrapper to convert arma::vec -> std::vector
auto conv_arma_to_vec(const arma::vec& x)
{
  auto y = arma::conv_to<std::vector<double>>::from(x);
  return y;
}

int main(int argc, char **argv)
{
  console::info("Initialising Grace...");

  //region Grace set up:
  auto g = dft_core::grace_plot::Grace();
  const int N_POINTS = 80;

  //region Grid set up:
  auto x_vector = arma::linspace(0.75, 1.5, N_POINTS);
  auto y_lims = std::vector<double>{-2, 10};
  g.set_x_limits(x_vector.min(), x_vector.max());
  g.set_y_limits(y_lims[0], y_lims[1]);
  //endregion

  //endregion

  //region Instantiation of the potentials:
  using namespace dft_core::physics::potentials::intermolecular;
  auto lj = LennardJones();
  auto twf = tenWoldeFrenkel();
  auto wrdf = WangRamirezDobnikarFrenkel();
  //endregion

  //region Lennard-Jones:
  //region Potential:
  auto lj_vector = lj.v_potential(x_vector); // equivalent to lj(x_vector);
  auto lj_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(lj_vector));
  //endregion

  //region Minimum:
  auto lj_min = g.add_dataset(std::vector<double>{lj.r_min()}, std::vector<double>{lj.v_min()});
  g.set_line_type(dft_core::grace_plot::LineStyle::NO_LINE, lj_min);
  g.set_symbol(dft_core::grace_plot::Symbol::SQUARE, lj_min);
  g.set_symbol_fill(dft_core::grace_plot::Color::RED, lj_min);
  //endregion

  //region Hard-sphere diameter:
  auto y_vec = arma::linspace(y_lims[0], y_lims[1], 10);
  auto hs_diameter = lj.find_hard_sphere_diameter(1.0);
  auto hs_x = arma::vec(10, arma::fill::ones); hs_x *= hs_diameter;
  auto lj_hs = g.add_dataset(conv_arma_to_vec(hs_x), conv_arma_to_vec(y_vec));
  g.set_color(dft_core::grace_plot::Color::RED, lj_hs);
  g.set_line_type(dft_core::grace_plot::LineStyle::DASHEDLINE_EN, lj_hs);
  console::info("LJ hard-sphere diameter (kT = 1.0) = " + std::to_string(hs_diameter));
  //endregion

  //region Pertubation theory:
  auto lj_att = lj.w_attractive(x_vector);
  auto lj_att_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(lj_att));
  g.set_color(dft_core::grace_plot::Color::RED, lj_att_ds);
  g.set_line_type(dft_core::grace_plot::LineStyle::D_DOTTEDDASHEDLINE_EM, lj_att_ds);
  g.set_symbol(dft_core::grace_plot::Symbol::CIRCLE, lj_att_ds);
  g.set_symbol_size(0.25, lj_att_ds);

  auto lj_rep = lj.w_repulsive(x_vector);
  auto lj_rep_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(lj_rep));
  g.set_color(dft_core::grace_plot::Color::RED, lj_rep_ds);
  g.set_line_type(dft_core::grace_plot::LineStyle::D_DOTTEDDASHEDLINE_EN, lj_rep_ds);
  g.set_symbol(dft_core::grace_plot::Symbol::STAR, lj_rep_ds);
  g.set_symbol_size(0.25, lj_rep_ds);
  //endregion
  //endregion

  //region ten Wolde-Frenkel:
  //region Potential:
  auto twf_vector = twf(x_vector);
  auto twf_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(twf_vector));
  g.set_color(dft_core::grace_plot::Color::BLUE,twf_ds);
  //endregion

  //region Minimum:
  auto twf_min = g.add_dataset(std::vector<double>{twf.r_min()}, std::vector<double>{twf.v_min()});
  g.set_line_type(dft_core::grace_plot::LineStyle::NO_LINE, twf_min);
  g.set_symbol(dft_core::grace_plot::Symbol::DIAMOND, twf_min);
  g.set_symbol_fill(dft_core::grace_plot::Color::BLUE, twf_min);
  //endregion

  //region Hard-sphere diameter:
  hs_x /= hs_diameter;
  hs_diameter = twf.find_hard_sphere_diameter(1.0); hs_x *= hs_diameter;
  auto twf_hs = g.add_dataset(conv_arma_to_vec(hs_x), conv_arma_to_vec(y_vec));
  g.set_color(dft_core::grace_plot::Color::BLUE, twf_hs);
  g.set_line_type(dft_core::grace_plot::LineStyle::DASHEDLINE_EN, twf_hs);
  console::info("tWF hard-sphere diameter (kT = 1.0) = " + std::to_string(hs_diameter));
  //endregion

  //region Pertubation theory:
  auto twf_att = twf.w_attractive(x_vector);
  auto twf_att_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(twf_att));
  g.set_color(dft_core::grace_plot::Color::BLUE, twf_att_ds);
  g.set_line_type(dft_core::grace_plot::LineStyle::DOTTEDLINE, twf_att_ds);
  g.set_symbol(dft_core::grace_plot::Symbol::STAR, twf_att_ds);
  g.set_symbol_size(0.25, twf_att_ds);

  auto twf_rep = twf.w_repulsive(x_vector);
  auto twf_rep_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(twf_rep));
  g.set_color(dft_core::grace_plot::Color::BLUE, twf_rep_ds);
  g.set_line_type(dft_core::grace_plot::LineStyle::D_DOTTEDDASHEDLINE_EN, twf_rep_ds);
  g.set_symbol(dft_core::grace_plot::Symbol::STAR, twf_rep_ds);
  g.set_symbol_size(0.25, twf_rep_ds);
  //endregion
  //endregion

  //region Wang-Ramirez-Dobnikar-Frenkel:
  //region Potential:
  auto wrdf_vector = wrdf(x_vector);
  auto wrdf_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(wrdf_vector));
  g.set_color(dft_core::grace_plot::Color::ORANGE, wrdf_ds);
  //endregion

  //region Minimum:
  auto wrdf_min = g.add_dataset(std::vector<double>{wrdf.r_min()}, std::vector<double>{wrdf.v_min()});
  g.set_line_type(dft_core::grace_plot::LineStyle::NO_LINE, wrdf_min);
  g.set_symbol(dft_core::grace_plot::Symbol::SQUARE, wrdf_min);
  g.set_symbol_fill(dft_core::grace_plot::Color::ORANGE, wrdf_min);
  //endregion

  //region Hard-sphere diameter:
  hs_diameter = wrdf.find_hard_sphere_diameter(1.0);
  hs_x = arma::vec(10, arma::fill::ones); hs_x *= hs_diameter;
  auto wrdf_hs = g.add_dataset(conv_arma_to_vec(hs_x), conv_arma_to_vec(y_vec));
  g.set_color(dft_core::grace_plot::Color::ORANGE, wrdf_hs);
  g.set_line_type(dft_core::grace_plot::LineStyle::DASHEDLINE_EN, wrdf_hs);
  console::info("WRDF hard-sphere diameter (kT = 1.0) = " + std::to_string(hs_diameter));
  //endregion

  //region Pertubation theory:
  auto wrdf_att = wrdf.w_attractive(x_vector);
  auto wrdf_att_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(wrdf_att));
  g.set_color(dft_core::grace_plot::Color::ORANGE, wrdf_att_ds);
  g.set_line_type(dft_core::grace_plot::LineStyle::D_DOTTEDDASHEDLINE_EM, wrdf_att_ds);
  g.set_symbol(dft_core::grace_plot::Symbol::PLUS, wrdf_att_ds);
  g.set_symbol_size(0.25, wrdf_att_ds);

  auto wrdf_rep = wrdf.w_repulsive(x_vector);
  auto wrdf_rep_ds = g.add_dataset(conv_arma_to_vec(x_vector), conv_arma_to_vec(wrdf_rep));
  g.set_color(dft_core::grace_plot::Color::ORANGE, wrdf_rep_ds);
  g.set_line_type(dft_core::grace_plot::LineStyle::D_DOTTEDDASHEDLINE_EN, wrdf_rep_ds);
  g.set_symbol(dft_core::grace_plot::Symbol::STAR, wrdf_rep_ds);
  g.set_symbol_size(0.25, wrdf_rep_ds);
  //endregion
  //endregion

  //region Legend:
  g.set_legend("Lennard-Jones (LJ)", lj_ds);
  g.set_legend("ten Wolde-Frenkel (tWF)", twf_ds);
  g.set_legend("Wang-Ramirez-Dobnikar-Frenkel (WRDF)", wrdf_ds);

  g.set_legend("LJ min", lj_min);
  g.set_legend("LJ attractive", lj_att_ds);
  g.set_legend("LJ repulsive", lj_rep_ds);
  g.set_legend("LJ HS diameter", lj_hs);

  g.set_legend("tWF min", twf_min);
  g.set_legend("tWF HS diameter", twf_hs);
  g.set_legend("tWF attractive", twf_att_ds);
  g.set_legend("tWF repulsive", twf_rep_ds);

  g.set_legend("WRDF min", wrdf_min);
  g.set_legend("WRDF HS diameter", wrdf_hs);
  g.set_legend("WRDF attractive", wrdf_att_ds);
  g.set_legend("WRDF repulsive", wrdf_rep_ds);
  //endregion

  g.redraw_and_wait();

  //g.print_to_file("potentials.png", dft_core::grace_plot::ExportFormat::PNG);
  //g.redraw_and_wait();
}