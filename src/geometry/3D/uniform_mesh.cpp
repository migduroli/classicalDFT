#include "classicaldft_bits/geometry/3D/uniform_mesh.h"

#include <cmath>

namespace dft_core::geometry::three_dimensional {

  UniformMesh::UniformMesh(double dx, std::vector<double>& dimensions, std::vector<double>& origin)
      : SUQMesh(dx, dimensions, origin), dx_(dx) {}

  double UniformMesh::spacing() const {
    return dx_;
  }

  Vertex UniformMesh::wrap(const Vertex& position) const {
    auto coords = position.coordinates();
    for (size_t d = 0; d < coords.size(); ++d) {
      coords[d] = std::fmod(coords[d], dimensions()[d]);
      if (coords[d] < 0.0) {
        coords[d] += dimensions()[d];
      }
    }
    return Vertex(std::move(coords));
  }

}  // namespace dft_core::geometry::three_dimensional
