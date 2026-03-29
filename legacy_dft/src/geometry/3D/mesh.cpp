#include "classicaldft_bits/geometry/3D/mesh.h"

#include "classicaldft_bits/geometry/3D/element.h"

#ifdef DFT_HAS_MATPLOTLIB
#include "matplotlibcpp.h"

#include <filesystem>
#include <iostream>
#endif

#include <numeric>

namespace dft::geometry::three_dimensional {

  void SUQMesh::initialise(double dx) {
    // region Vertices init:
    auto vertex_index = 0;
    auto element_index = 0;

    auto x = origin_[0];
    auto y = origin_[1];
    auto z = origin_[2];
    vertices_raw_ = vertex_vec(number_vertices_);
    elements_raw_ = sqbox_vec(number_elements_);

    for (auto i_idx = 0; i_idx < shape_[0]; i_idx++) {
      for (auto j_idx = 0; j_idx < shape_[1]; j_idx++) {
        for (auto k_idx = 0; k_idx < shape_[2]; k_idx++) {
          vertices_raw_[vertex_index] = Vertex({x, y, z});
          vertices_.insert({vertex_index, std::ref(vertices_raw_[vertex_index])});

          if ((i_idx < idx_max_[0]) && (j_idx < idx_max_[1]) && (k_idx < idx_max_[2])) {
            elements_raw_[element_index] = SquareBox(dx, {x, y, z});
            elements_.insert({element_index, std::ref(elements_raw_[element_index])});
            element_index += 1;
          }

          vertex_index += 1;
          z += dx;
        }
        z = 0;
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
    if (!interactive) {
      plt::backend("Agg");
    }

    // Project 3D vertices onto XY plane (slice at z=0)
    std::vector<double> xs, ys;
    double z0 = origin_[2];
    double tol = 1e-10;
    for (const auto& v : vertices_raw_) {
      if (std::abs(v.coordinates()[2] - z0) < tol) {
        xs.push_back(v.coordinates()[0]);
        ys.push_back(v.coordinates()[1]);
      }
    }

    plt::figure_size(700, 700);
    plt::named_plot("Vertices (z=0 slice)", xs, ys, "bs");
    double pad_x = 0.1 * dimensions_[0];
    double pad_y = 0.1 * dimensions_[1];
    plt::xlim(origin_[0] - pad_x, dimensions_[0] + origin_[0] + pad_x);
    plt::ylim(origin_[1] - pad_y, dimensions_[1] + origin_[1] + pad_y);
    plt::xlabel(R"($x$)");
    plt::ylabel(R"($y$)");
    plt::title(
        "3D mesh XY slice (" + std::to_string(shape_[0]) + "x" + std::to_string(shape_[1]) + "x" +
        std::to_string(shape_[2]) + " vertices)"
    );
    plt::grid(true);
    plt::tight_layout();

    if (interactive) {
      plt::show();
    } else {
      std::filesystem::create_directories(std::filesystem::path(path).parent_path());
      plt::save(path);
      plt::close();
      std::cout << "Plot saved: " << std::filesystem::absolute(path) << std::endl;
    }
#else
    throw std::runtime_error("No plotting backend available: build with -DDFT_USE_MATPLOTLIB=ON");
#endif
  }

  const std::vector<SquareBox>& SUQMesh::elements() const {
    return elements_raw_;
  }

  double SUQMesh::element_volume() const {
    return elements().front().volume();
  }

}  // namespace dft::geometry::three_dimensional