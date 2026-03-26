#include <cmath>
#include <armadillo>
#include <classicaldft>

int main(int argc, char **argv)
{
  console::info("Initialising Grace...");

  //region Cttor
  auto g = dft_core::grace_plot::Grace();
  const int N_POINTS = 100;
  //endregion

  //region Example of adding points to the default dataset 0
  console::info("Plotting sin(x): x in [0, 2*PI]");

  auto x_vector = arma::linspace(0, 2*M_PI, N_POINTS);
  auto y_vector = arma::sin(x_vector);

  for (auto k = 0; k < x_vector.size(); k++)
  { g.add_point(x_vector[k], y_vector[k]); }

  g.set_x_limits(x_vector.min(), x_vector.max());
  g.set_y_limits(y_vector.min(), y_vector.max());

  g.redraw_and_wait();
  //endregion

  //region Example of adding dataset
  console::info("Adding new dataset");
  arma::vec z_vector = arma::cos(x_vector);

  auto dataset_id = g.add_dataset(
      arma::conv_to<std::vector<double>>::from(x_vector),
      arma::conv_to<std::vector<double>>::from(z_vector)
  );
  console::debug("The new dataset id = " + std::to_string(dataset_id));
  g.set_color(dft_core::grace_plot::Color::RED, dataset_id);
  g.redraw_and_wait();
  //endregion

  //region Example of replacing dataset
  console::info("Example: Replacing existing dataset");
  arma::vec w_vector = arma::tan(x_vector);

  auto existing_dataset_id = g.add_dataset(
      arma::conv_to<std::vector<double>>::from(x_vector),
      arma::conv_to<std::vector<double>>::from(w_vector)
  );
  console::debug("The new dataset id = " + std::to_string(dataset_id));
  g.set_color(dft_core::grace_plot::Color::BLUE, existing_dataset_id);
  g.redraw_and_wait();

  console::warning("Replacing dataset id: " + std::to_string(existing_dataset_id));
  w_vector = arma::sqrt(x_vector)/sqrt(2*M_PI);
  g.replace_dataset(
      arma::conv_to<std::vector<double>>::from(x_vector),
      arma::conv_to<std::vector<double>>::from(w_vector),
      existing_dataset_id
  );
  console::debug("The replaced graph id = " + std::to_string(existing_dataset_id));

  g.set_legend("x\\S1/2\\N", existing_dataset_id);
  //g.set_color(dft_core::grace_plot::MAGENTA, existing_dataset_id);
  g.redraw_and_wait();
  //endregion

  //region Example of axis labels
  console::info("Example: Setting the X and Y labels");
  g.set_label("This is X", dft_core::grace_plot::Axis::X);
  g.set_label("This is Y", dft_core::grace_plot::Axis::Y);
  g.redraw_and_wait();
  //endregion

  //region Example of graph title
  console::info("Example: Setting the Title and Subtitle");
  g.set_title("This is the title");
  g.set_subtitle("And this is the subtitle");
  g.redraw_and_wait();
  //endregion

  //region Example of setting limits
  console::info("Example: Setting the limits");
  g.set_limits(std::vector<double>{ -0.1, 2*M_PI+0.1 }, std::vector<double>{-1.2, 1.2});
  g.redraw_and_wait();
  //endregion

  //region Example of setting ticks
  console::info("Example: Setting the tick spacing");
  g.set_ticks(0.5, 0.1);
  g.redraw_and_wait(false, false);
  //endregion

  //region Example of setting line type
  console::info("Example: Setting the line type");
  g.set_line_type(dft_core::grace_plot::LineStyle::NO_LINE, 0);
  g.redraw_and_wait(false, false);
  g.set_line_type(dft_core::grace_plot::LineStyle::LINE, 0);
  g.redraw_and_wait(false, false);
  g.set_line_type(dft_core::grace_plot::LineStyle::DOTTEDLINE, 0);
  g.redraw_and_wait(false, false);
  g.set_line_type(dft_core::grace_plot::LineStyle::DASHEDLINE_EN, 0);
  g.redraw_and_wait(false, false);
  //endregion

  //region Example of setting symbols and fills
  console::info("Example: Setting symbol and symbol color");
  g.set_symbol(dft_core::grace_plot::Symbol::TRIANGLE_DOWN, 0);
  g.set_symbol_color(dft_core::grace_plot::Color::BLUE, 0);
  g.set_symbol_fill(dft_core::grace_plot::Color::BLUE, 0, 0, 4);

  g.set_symbol(dft_core::grace_plot::Symbol::TRIANGLE_LEFT, 1);
  g.set_symbol_color(dft_core::grace_plot::Color::DARKGREEN, 1);

  g.set_symbol(dft_core::grace_plot::Symbol::DIAMOND, 2);
  g.set_symbol_fill(dft_core::grace_plot::Color::RED, 2);
  g.redraw_and_wait(false, false);
  //endregion

  //region Example of setting symbol size
  console::info("Example: Setting symbol size");
  g.set_symbol_size(0.5,0);
  g.set_symbol_size(1.5,1);
  g.redraw_and_wait(false, false);
  //endregion

  //region Example of Export
  console::info("Example: Saving the result as PNG");
  g.print_to_file("test_graph.png", dft_core::grace_plot::ExportFormat::PNG);
  //g.print_to_file("test_graph.pdf", dft_core::grace_plot::ExportFormat::PDF);
  //g.print_to_file("test_graph.jpg", dft_core::grace_plot::ExportFormat::JPG);
  //g.print_to_file("test_graph.eps", dft_core::grace_plot::ExportFormat::EPS1);
  //g.print_to_file("test_graph.ps", dft_core::grace_plot::ExportFormat::PS);
  g.redraw_and_wait(false, false);
  //endregion
}