// The order in which headers are imported follows Google's C++ style guide:
// link: https://google.github.io/styleguide/cppguide.html
#include "classicaldft_bits/graph/grace.h"

#include "classicaldft_bits/exceptions/grace.h"
#include "classicaldft_bits/io/console.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace dft_core::grace_plot::command {
  std::string arrange_command(const int& number_of_rows, const int& number_of_columns, const float& offset,
                              const float& horizontal_gap, const float& vertical_gap) {
    std::string cmd = "ARRANGE(" + std::to_string(number_of_rows) + ", " + std::to_string(number_of_columns) + ", " +
                      std::to_string(offset) + ", " + std::to_string(horizontal_gap) + ", " +
                      std::to_string(vertical_gap) + ")";
    return cmd;
  }
  std::string set_x_min_command(const double& x_min) {
    auto x_rounded = static_cast<float>(x_min);
    std::string cmd = "WORLD XMIN " + std::to_string(x_rounded);
    return cmd;
  }
  std::string set_x_max_command(const double& x_max) {
    auto x_rounded = static_cast<float>(x_max);
    std::string cmd = "WORLD XMAX " + std::to_string(x_rounded);
    return cmd;
  }
  std::string set_y_min_command(const double& y_min) {
    auto y_rounded = static_cast<float>(y_min);
    std::string cmd = "WORLD YMIN " + std::to_string(y_rounded);
    return cmd;
  }
  std::string set_y_max_command(const double& y_max) {
    auto y_rounded = static_cast<float>(y_max);
    std::string cmd = "WORLD YMAX " + std::to_string(y_rounded);
    return cmd;
  }
  std::string add_point_command(const double& x, const double& y, const int& dataset_id, const int& graph_id) {
    std::string cmd = "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " POINT " +
                      std::to_string(x) + "," + std::to_string(y);
    return cmd;
  }
  std::string redraw_command() {
    std::string cmd = "REDRAW";
    return cmd;
  }
  std::string auto_scale_command() {
    std::string cmd = "AUTOSCALE";
    return cmd;
  }
  std::string auto_ticks_command() {
    std::string cmd = "AUTOTICKS";
    return cmd;
  }
  std::string focus_command(const int& graph_id) {
    std::string cmd = "FOCUS G" + std::to_string(graph_id);
    return cmd;
  }
  std::string kill_set_command(const int& dataset_id, const int& graph_id) {
    std::string cmd = "KILL G" + std::to_string(graph_id) + "." + "S" + std::to_string(dataset_id);
    return cmd;
  }
  std::string set_legend_command(const std::string& legend, const int& dataset_id, const int& graph_id) {
    std::string cmd = "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " LEGEND \"" + legend + "\"";
    return cmd;
  }
  std::string set_line_color_command(const grace_plot::Color& color, const int& dataset_id, const int& graph_id) {
    std::string cmd = "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " LINE COLOR " +
                      std::to_string(static_cast<int>(color));
    return cmd;
  }
  std::string set_symbol_color_command(const grace_plot::Color& color, const int& dataset_id, const int& graph_id) {
    std::string cmd = "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " SYMBOL COLOR " +
                      std::to_string(static_cast<int>(color));
    return cmd;
  }
  std::string set_symbol_color_fill_command(const grace_plot::Color& color, const int& dataset_id,
                                            const int& graph_id) {
    std::string cmd = "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " SYMBOL FILL COLOR " +
                      std::to_string(static_cast<int>(color));
    return cmd;
  }
  std::string set_symbol_color_fill_pattern_command(const int& pattern_id, const int& dataset_id, const int& graph_id) {
    std::string cmd = "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " SYMBOL FILL PATTERN " +
                      std::to_string(pattern_id);
    return cmd;
  }
  std::string set_symbol_size_command(const double& size, const int& dataset_id, const int& graph_id) {
    auto size_f = static_cast<float>(size);
    std::string cmd =
        "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " SYMBOL SIZE " + std::to_string(size_f);
    return cmd;
  }
  std::string set_axis_label_command(const std::string& label, const grace_plot::Axis& axis) {
    std::string axis_string = (axis == Axis::X ? "X" : "Y");
    std::string cmd = axis_string + "AXIS LABEL \"" + label + "\"";
    return cmd;
  }
  std::string set_title_command(const std::string& title) {
    std::string cmd = "TITLE \"" + title + "\"";
    return cmd;
  }
  std::string set_subtitle_command(const std::string& subtitle) {
    std::string cmd = "SUBTITLE \"" + subtitle + "\"";
    return cmd;
  }
  std::string set_ticks_command(const double& tick_sep, const Axis& axis) {
    std::string axis_string = (axis == Axis::X ? "X" : "Y");
    auto tick_sep_f = static_cast<float>(tick_sep);
    std::string cmd = axis_string + "AXIS TICK MAJOR " + std::to_string(tick_sep_f);
    return cmd;
  }
  std::string set_symbol_command(const Symbol& symbol_id, const int& dataset_id, const int& graph_id) {
    std::string cmd = "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " SYMBOL " +
                      std::to_string(static_cast<int>(symbol_id));
    return cmd;
  }
  std::string set_line_style_command(const LineStyle& line_type, const int& dataset_id, const int& graph_id) {
    std::string cmd = "G" + std::to_string(graph_id) + ".S" + std::to_string(dataset_id) + " LINE LINESTYLE " +
                      std::to_string(static_cast<int>(line_type));
    return cmd;
  }
  std::string set_format_command(const ExportFormat& format) {
    std::string cmd = "HARDCOPY DEVICE ";
    std::string format_s = "PNG";

    if (ExportFormat::PNG == format) {
      format_s = "PNG";
    } else if (ExportFormat::PDF == format) {
      format_s = "PDF";
    } else if (ExportFormat::JPG == format) {
      format_s = "JPG";
    } else if (ExportFormat::PS == format) {
      format_s = "PS";
    } else if (ExportFormat::EPS1 == format) {
      format_s = "EPS1";
    } else {
      console::warning("The format specified is not yet implemented. Exporting as PNG...");
      format_s = "PNG";
    }

    cmd += "\"" + format_s + "\"";
    return cmd;
  }
  std::string print_to_file_command(const std::string& file_path) {
    std::string cmd = "PRINT TO \"" + file_path + "\"";
    return cmd;
  }
  std::string print_command() {
    std::string cmd = "PRINT";
    return cmd;
  }
}  // namespace dft_core::grace_plot::command

namespace dft_core::grace_plot {
  // region static methods

  static void add_dataset_to_grace_object(const Grace* g, std::vector<double> const& x, std::vector<double> const& y,
                                          const int& dataset_id, const int& graph_id) {
    auto length = x.size();
    for (size_t k = 0; k < length; k++) {
      g->add_point(x[k], y[k], dataset_id, graph_id);
    }
  }

  static bool check_equal_length(std::vector<double> const& x, std::vector<double> const& y) {
    if (x.size() != y.size()) {
      throw exception::GraceException(
          "The dataset must have equal-size X and Y:"
          " x_size = " +
          std::to_string(x.size()) + "!=" + " y_size = " + std::to_string(y.size()));
    }
    return true;
  }

  static bool check_graph_id_in_bounds(const int& graph_id, const int& total_number_of_graphs) {
    if ((graph_id > (total_number_of_graphs - 1)) || (graph_id < 0)) {
      throw exception::GraceException("The graph id is out of bounds: Min id = 0; Max id =" +
                                      std::to_string(total_number_of_graphs));
    }
    return true;
  }

  static bool check_dataset_in_bounds(const int& dataset_id, const int& last_dataset_id) {
    if (dataset_id < 0 || dataset_id > last_dataset_id) {
      throw exception::GraceException("The dataset id is out of bounds: Min id = 0; Last dataset id =" +
                                      std::to_string(last_dataset_id));
    }
    return true;
  }

  // endregion

  // region General methods

  void send_command(const std::string& cmd) {
    if (GraceIsOpen()) {
      try {
        GracePrintf(cmd.c_str());  // NOLINT(cppcoreguidelines-pro-type-vararg)
      } catch (...) {
        throw dft_core::exception::GraceCommunicationFailedException();
      }
    } else {
      throw dft_core::exception::GraceNotOpenedException();
    }
  }

  void error_parsing_function(const char* msg) {
    std::cout << "Grace library message: \"" << msg << "\"" << std::endl;
  }

  void register_grace_error_function() {
    try {
      GraceRegisterErrorFunction(error_parsing_function);
    } catch (const std::exception& e) {
      throw dft_core::exception::GraceException("error_parsing_function could not be registered by xmgrace");
    }
  }

  void start_grace_communication(const int& x_size, const int& y_size, int buffer_size) {
    if ((x_size <= 0) || (y_size <= 0)) {
      throw dft_core::exception::GraceException("The xmgrace-canvas dimensions cannot be negative!");
    }
    if (buffer_size <= 0) {
      throw dft_core::exception::GraceException("The communication buffer size cannot be negative!");
    }

    char grace_name[] = "xmgrace";  // NOLINT(modernize-avoid-c-arrays)

    // Start Grace with a buffer size and open the pipe
    std::string geometry_spec = std::to_string(x_size) + "x" + std::to_string(y_size);
    auto response =
        GraceOpenVA(grace_name, buffer_size, option::FREE.c_str(),  // NOLINT(cppcoreguidelines-pro-type-vararg)
                    option::NO_SAFE.c_str(), option::GEOMETRY.c_str(), geometry_spec.c_str(), NULL);

    // Throwing exception in case of wrong communication, to keep track of what's happening
    if (-1 == response) {
      throw dft_core::exception::GraceCommunicationFailedException();
    }
  }

  int get_number_of_rows(const int& number_of_graphs) {
    if (number_of_graphs < 1) {
      throw dft_core::exception::GraceException("Number of graphs cannot be lesser than one");
    }
    int n_rows = (number_of_graphs <= 2 ? 1 : 2);
    return n_rows;
  }

  int get_number_of_columns(const int& number_of_graphs, const int& number_of_rows) {
    if (number_of_graphs < 1) {
      throw dft_core::exception::GraceException("Number of graphs cannot be lesser than one");
    } else if (number_of_rows < 1) {
      throw dft_core::exception::GraceException("Number of rows cannot be lesser than one");
    }

    int number_of_columns = number_of_graphs / number_of_rows;
    number_of_columns += (number_of_graphs - number_of_rows * number_of_columns);

    return number_of_columns;
  }
  // endregion

  // region Grace

  /// The configuration command which eventually sets up the graph
  void setup_grace(const double& x_min, const double& x_max, const double& y_min, const double& y_max,
                   const int& number_of_graphs, const float& offset, const float& hspace, float const& vspace) {
    if (number_of_graphs > 1) {
      int number_of_rows = get_number_of_rows(number_of_graphs);
      int number_of_columns = get_number_of_columns(number_of_graphs, number_of_rows);
      send_command(command::arrange_command(number_of_rows, number_of_columns, offset, hspace, vspace));
    }

    send_command(command::set_x_min_command(x_min));
    send_command(command::set_x_max_command(x_max));

    // tick major must be set up befor minor to avoid glitch
    send_command("XAXIS TICK MAJOR 5");
    send_command("XAXIS TICK MINOR 1");

    send_command("WORLD YMIN " + std::to_string(y_min));
    send_command("WORLD YMAX " + std::to_string(y_max));

    // tick major must be set up befor minor to avoid glitch
    send_command("YAXIS TICK MAJOR " + std::to_string(static_cast<int>(y_max)));
    send_command("YAXIS TICK MINOR " + std::to_string(static_cast<int>(y_max / 2)));

    send_command("AUTOSCALE ONREAD XYAXES");
  }

  Grace::Grace(int x_size, int y_size, int n_graph, bool show)
      : x_min_(DEFAULT_MIN_AXIS_VALUE),
        x_max_(DEFAULT_MAX_AXIS_VALUE),
        y_min_(DEFAULT_MIN_AXIS_VALUE),
        y_max_(DEFAULT_MAX_AXIS_VALUE),
        offset_(DEFAULT_OFFSET),
        horizontal_space_(DEFAULT_HORIZONTAL_SPACE),
        vertical_space_(DEFAULT_VERTICAL_SPACE),
        last_dataset_id_(DEFAULT_DATASET_ID),
        number_of_graphs_(n_graph),
        show_(show) {
    if (show_) {
      register_grace_error_function();
      start_grace_communication(x_size, y_size);
      setup_grace(x_min_, x_max_, y_min_, y_max_, number_of_graphs_, offset_, horizontal_space_, vertical_space_);
    }
  }

  void Grace::set_x_min(const double& value) {
    x_min_ = value;
  }

  void Grace::set_x_max(const double& value) {
    x_max_ = value;
  }

  void Grace::set_y_min(const double& value) {
    y_min_ = value;
  }

  void Grace::set_y_max(const double& value) {
    y_max_ = value;
  }

  void Grace::set_x_limits(const double& x_min, const double& x_max) {
    if (x_min > x_max) {
      throw exception::GraceException("Lower limit cannot be greater than upper limit!");
    }

    this->set_x_min(x_min);
    this->set_x_max(x_max);

    if (this->show_) {
      send_command(command::set_x_min_command(this->x_min()));
      send_command(command::set_x_max_command(this->x_max()));
    }
  }

  void Grace::set_y_limits(const double& y_min, const double& y_max) {
    if (y_min > y_max) {
      throw exception::GraceException("Lower limit cannot be greater than upper limit!");
    }

    this->set_y_min(y_min);
    this->set_y_max(y_max);

    if (this->show_) {
      send_command(command::set_y_min_command(this->y_min()));
      send_command(command::set_y_max_command(this->y_max()));
    }
  }

  void Grace::set_limits(const double& x_min, const double& x_max, const double& y_min, const double& y_max) {
    this->set_x_limits(x_min, x_max);
    this->set_y_limits(y_min, y_max);
  }

  void Grace::set_limits(const std::vector<double>& x_limits, const std::vector<double>& y_limits) {
    this->set_limits(x_limits.front(), x_limits.back(), y_limits.front(), y_limits.back());
  }

  void Grace::close() const {
    if (this->show_) {
      GraceClose();
    }
  }

  void Grace::add_point(const double& x, const double& y, const int& dataset_id, const int& graph_id) const {
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());

    if (this->show_) {
      send_command(command::add_point_command(x, y, dataset_id, graph_id));
    }
  }

  void Grace::increase_last_dataset_id() {
    this->last_dataset_id_ += 1;
  }

  void Grace::decrease_last_dataset_id() {
    this->last_dataset_id_ -= 1;
  }

  int Grace::add_dataset(std::vector<double> const& x, std::vector<double> const& y, const int& graph_id) {
    check_equal_length(x, y);
    this->increase_last_dataset_id();
    add_dataset_to_grace_object(this, x, y, this->last_dataset_id(), graph_id);
    return last_dataset_id_;
  }

  void Grace::delete_dataset(const int& dataset_id, const int& graph_id) {
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());

    send_command(command::kill_set_command(dataset_id, graph_id));
    // this->decrease_last_dataset_id();

    this->set_color(static_cast<Color>((this->last_dataset_id() % 10) + 1), dataset_id, graph_id);
  }

  void Grace::replace_dataset(std::vector<double> const& x, std::vector<double> const& y, const int& dataset_id,
                              const int& graph_id) {
    check_equal_length(x, y);
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());

    this->delete_dataset(dataset_id, graph_id);
    add_dataset_to_grace_object(this, x, y, dataset_id, graph_id);
  }

  void Grace::redraw(const bool& auto_scale, const bool& auto_ticks, const int& graph_id) const {
    if (this->show_) {
      if (check_graph_id_in_bounds(graph_id, this->number_of_graphs())) {
        send_command(command::focus_command(graph_id));
      }

      if (auto_scale) {
        send_command(command::auto_scale_command());
      }

      if (auto_ticks) {
        send_command(command::auto_ticks_command());
      }

      send_command(command::redraw_command());
    }
  }

  void Grace::wait() const {
    console::wait();
  }

  void Grace::redraw_and_wait(const bool& auto_scale, const bool& auto_ticks, const int& graph_id) const {
    this->redraw(auto_scale, auto_ticks, graph_id);
    this->wait();
  }

  void Grace::set_legend(const std::string& legend, const int& dataset_id, const int& graph_id) const {
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());

    send_command(command::set_legend_command(legend, dataset_id, graph_id));
  }

  void Grace::set_color(const Color& color, const int& dataset_id, const int& graph_id) const {
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());

    send_command(command::set_line_color_command(color, dataset_id, graph_id));
    send_command(command::set_symbol_color_command(color, dataset_id, graph_id));
  }

  void Grace::set_label(const std::string& label, const Axis& axis, const int& graph_id) const {
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());

    if (graph_id > 0) {
      send_command(command::focus_command(graph_id));
    }
    send_command(command::set_axis_label_command(label, axis));
  }

  void Grace::set_title(const std::string& title) const {
    send_command(command::set_title_command(title));
  }

  void Grace::set_subtitle(const std::string& subtitle) const {
    send_command(command::set_subtitle_command(subtitle));
  }

  void Grace::set_ticks(const double& dx, const double& dy, const int& graph_id) const {
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());

    if (dx <= 0 || dy <= 0) {
      std::string msg = "Minimum tick-size is 0: dx = " + std::to_string(dx) + " dy = " + std::to_string(dy);
      throw exception::GraceException(msg);
    }

    send_command(command::focus_command(graph_id));
    send_command(command::set_ticks_command(dx, Axis::X));
    send_command(command::set_ticks_command(dy, Axis::Y));
  }

  void Grace::set_symbol(const Symbol& symbol, const int& dataset_id, const int& graph_id) const {
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());

    send_command(command::set_symbol_command(symbol, dataset_id, graph_id));
  }

  void Grace::set_symbol_color(const Color& color, const int& dataset_id, const int& graph_id) const {
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());

    send_command(command::set_symbol_color_command(color, dataset_id, graph_id));
  }

  void Grace::set_symbol_fill(const Color& color, const int& dataset_id, const int& graph_id,
                              const int& pattern_id) const {
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());

    send_command(command::set_symbol_color_fill_pattern_command(pattern_id, dataset_id, graph_id));
    send_command(command::set_symbol_color_fill_command(color, dataset_id, graph_id));
  }

  void Grace::set_symbol_size(const double& size, const int& dataset_id, const int& graph_id) const {
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());

    send_command(command::set_symbol_size_command(size, dataset_id, graph_id));
  }

  void Grace::set_line_type(const LineStyle& line_type, const int& dataset_id, const int& graph_id) const {
    check_graph_id_in_bounds(graph_id, this->number_of_graphs());
    check_dataset_in_bounds(dataset_id, this->last_dataset_id());

    send_command(command::set_line_style_command(line_type, dataset_id, graph_id));
  }

  void Grace::print_to_file(const std::string& file_path, const ExportFormat& format) const {
    send_command(command::set_format_command(format));
    send_command(command::print_to_file_command(file_path));
    send_command(command::print_command());
  }
  // endregion
}  // namespace dft_core::grace_plot
