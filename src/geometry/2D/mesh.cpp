#include "dft/geometry/2D/mesh.h"

#include "dft/geometry/2D/element.h"
#ifdef DFT_HAS_GRACE
#include "dft/plotting/grace.h"
#endif
#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"
#endif

#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace dft::geometry::two_dimensional {

  void SUQMesh::initialise(double dx) {
    // region Vertices init:
    auto vertex_index = 0;
    auto element_index = 0;

    auto x = origin_[0];
    auto y = origin_[1];
    vertices_raw_ = vertex_vec(number_vertices_);
    elements_raw_ = sqbox_vec(number_elements_);

    for (auto i_idx = 0; i_idx < shape_[0]; i_idx++) {
      for (auto j_idx = 0; j_idx < shape_[1]; j_idx++) {
        vertices_raw_[vertex_index] = Vertex({x, y});
        vertices_.insert({vertex_index, std::ref(vertices_raw_[vertex_index])});

        if ((i_idx < idx_max_[0]) && (j_idx < idx_max_[1])) {
          elements_raw_[element_index] = SquareBox(dx, {x, y});
          elements_.insert({element_index, std::ref(elements_raw_[element_index])});
          element_index += 1;
        }

        vertex_index += 1;
        y += dx;
      }
      y = 0;
      x += dx;
    }
    // endregion
  }

  SUQMesh::SUQMesh(double dx, std::vector<double>& dimensions, std::vector<double>& origin)
      : dft::geometry::SUQMesh(dx, dimensions, origin) {
    this->initialise(dx);
  }

  void SUQMesh::plot(const std::string& path, const bool interactive) const {
#ifdef DFT_HAS_MATPLOTLIB
    namespace plt = matplotlibcpp;
    if (!interactive && path.empty()) {
      throw std::invalid_argument("plot: non-interactive mode requires a file path");
    }
    if (!interactive) {
      plt::backend("Agg");
    }

    std::vector<double> xs(vertices_raw_.size()), ys(vertices_raw_.size());
    for (size_t i = 0; i < vertices_raw_.size(); ++i) {
      xs[i] = vertices_raw_[i].coordinates()[0];
      ys[i] = vertices_raw_[i].coordinates()[1];
    }

    plt::figure_size(700, 700);
    plt::named_plot("Vertices", xs, ys, "bs");
    double pad_x = 0.1 * dimensions_[0];
    double pad_y = 0.1 * dimensions_[1];
    plt::xlim(origin_[0] - pad_x, dimensions_[0] + origin_[0] + pad_x);
    plt::ylim(origin_[1] - pad_y, dimensions_[1] + origin_[1] + pad_y);
    plt::xlabel(R"($x$)");
    plt::ylabel(R"($y$)");
    plt::title("2D mesh (" + std::to_string(shape_[0]) + "x" + std::to_string(shape_[1]) + " vertices)");
    plt::grid(true);
    plt::tight_layout();

    if (!path.empty()) {
      std::filesystem::create_directories(std::filesystem::path(path).parent_path());
      plt::save(path);
      std::cout << "Plot saved: " << std::filesystem::absolute(path) << std::endl;
    }
    if (interactive) {
      plt::show();
    }
    plt::close();
#elif defined(DFT_HAS_GRACE)
    auto g = dft::plotting::Grace();
    for (const auto& v : vertices_raw_) {
      g.add_point(v.coordinates()[0], v.coordinates()[1]);
    }

    auto dx = std::vector<double>{0.1 * dimensions_[0], 0.1 * dimensions_[1]};

    g.set_limits(
        {origin_[0] - dx[0], (dimensions_[0] + origin_[0]) + dx[0]},
        {origin_[1] - dx[1], (dimensions_[1] + origin_[1]) + dx[1]}
    );

    g.set_line_type(dft::plotting::LineStyle::NO_LINE, 0);
    g.set_symbol(dft::plotting::Symbol::SQUARE, 0);
    g.set_symbol_color(dft::plotting::Color::BLUE, 0);
    g.set_symbol_fill(dft::plotting::Color::DARKGREEN, 0);

    if (!path.empty()) {
      g.print_to_file(path);
    }
    if (interactive) {
      g.redraw_and_wait();
    }
#else
    throw std::runtime_error("No plotting backend available: build with -DDFT_USE_MATPLOTLIB=ON or -DDFT_USE_GRACE=ON");
#endif
  }

  const std::vector<SquareBox>& SUQMesh::elements() const {
    return elements_raw_;
  }

  double SUQMesh::element_volume() const {
    return elements().front().volume();
  }

}  // namespace dft::geometry::two_dimensional