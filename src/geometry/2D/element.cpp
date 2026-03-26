#include "dft/geometry/2D/element.h"

#include "dft/geometry/base/mesh.h"

namespace dft::geometry::two_dimensional {

  vertex_vec SquareBox::generate_vertices(double dx, const std::vector<double>& x0) {
    // bottom-up anti-clock-wise:
    // [3] - - - - [2]
    //  |           |
    //  |           |
    //  |           |
    // [0] - - - - [1]

    auto x_idx = static_cast<unsigned long>(Direction::X);
    auto y_idx = static_cast<unsigned long>(Direction::Y);

    return vertex_vec{
        Vertex(x0),
        Vertex({x0.at(x_idx) + dx, x0.at(y_idx)}),
        Vertex({x0.at(x_idx) + dx, x0.at(y_idx) + dx}),
        Vertex({x0.at(x_idx), x0.at(y_idx) + dx}),
    };
  }

  SquareBox::SquareBox() {
    origin_ = std::vector<double>{0, 0};
    length_ = DEFAULT_SQUAREBOX_LENGTH;
    vertices_raw_ = generate_vertices(length_, origin_);
    initialise(length_, origin_);
  }

  SquareBox::SquareBox(double length, const std::vector<double>& origin) {
    vertices_raw_ = generate_vertices(length, origin);
    initialise(length, origin);
  }

  SquareBox::SquareBox(vertex_vec&& vertices) {
    if (vertices.size() == 4) {
      vertices_raw_ = std::move(vertices);
      initialise_element();
    } else {
      throw std::runtime_error("2D square-box needs 4 vertices to be initialised");
    }
  }

}  // namespace dft::geometry::two_dimensional