#ifndef DFT_GEOMETRY_MESH_HPP
#define DFT_GEOMETRY_MESH_HPP

#include "dft/geometry/element.hpp"

#include <cmath>
#include <stdexcept>
#include <variant>
#include <vector>

namespace dft::geometry {

  struct UniformMesh2D {
    double dx;
    std::vector<double> dimensions;
    std::vector<double> origin;
    std::vector<long> shape;
    std::vector<Vertex> vertices;
    std::vector<SquareBox2D> elements;
  };

  struct UniformMesh3D {
    double dx;
    std::vector<double> dimensions;
    std::vector<double> origin;
    std::vector<long> shape;
    std::vector<Vertex> vertices;
    std::vector<SquareBox3D> elements;
  };

  using Mesh = std::variant<UniformMesh2D, UniformMesh3D>;

  // Mesh index helpers (dimension-independent)

  namespace detail {

    constexpr double SCALING = 1e-8;

    [[nodiscard]] inline auto flat_index_impl(const std::vector<long>& idxs, const std::vector<long>& shape) -> long {
      long v = idxs[0];
      for (size_t k = 0; k < idxs.size() - 1; ++k) {
        v = idxs[k + 1] + shape[k + 1] * v;
      }
      return v;
    }

    [[nodiscard]] inline auto cartesian_index_impl(long pos, const std::vector<long>& shape) -> std::vector<long> {
      std::vector<long> result(shape.size());
      for (auto k = static_cast<int>(shape.size()) - 1; k > 0; --k) {
        result[static_cast<size_t>(k)] = pos % shape[static_cast<size_t>(k)];
        pos /= shape[static_cast<size_t>(k)];
      }
      result[0] = pos;
      return result;
    }

    inline void validate_indices(std::vector<long>& idxs, const std::vector<long>& shape) {
      if (idxs.size() != shape.size()) {
        throw std::invalid_argument("Index dimension does not match mesh dimension");
      }
      for (size_t i = 0; i < idxs.size(); ++i) {
        if (idxs[i] < 0) {
          idxs[i] += shape[i];
        }
        if (idxs[i] < 0 || idxs[i] >= shape[i]) {
          throw std::out_of_range("Index out of bounds in mesh indexer");
        }
      }
    }

  }  // namespace detail

  // Factory functions

  [[nodiscard]] inline auto uniform_mesh_2d(double dx, std::vector<double> dims, std::vector<double> orig)
      -> UniformMesh2D {
    auto nx = static_cast<long>((dims[0] + detail::SCALING * dx) / dx) + 1;
    auto ny = static_cast<long>((dims[1] + detail::SCALING * dx) / dx) + 1;

    std::vector<long> shape = {nx, ny};
    auto n_vertices = static_cast<size_t>(nx * ny);
    auto n_elements = static_cast<size_t>((nx - 1) * (ny - 1));

    std::vector<Vertex> vertices(n_vertices);
    std::vector<SquareBox2D> elements;
    elements.reserve(n_elements);

    size_t vertex_index = 0;
    auto x = orig[0];
    for (long i = 0; i < nx; ++i) {
      auto y = orig[1];
      for (long j = 0; j < ny; ++j) {
        vertices[vertex_index] = Vertex{{x, y}};
        if (i < nx - 1 && j < ny - 1) {
          elements.push_back(make_square_box_2d(dx, {x, y}));
        }
        ++vertex_index;
        y += dx;
      }
      x += dx;
    }

    return UniformMesh2D{
        .dx = dx,
        .dimensions = std::move(dims),
        .origin = std::move(orig),
        .shape = std::move(shape),
        .vertices = std::move(vertices),
        .elements = std::move(elements),
    };
  }

  [[nodiscard]] inline auto uniform_mesh_3d(double dx, std::vector<double> dims, std::vector<double> orig)
      -> UniformMesh3D {
    auto nx = static_cast<long>((dims[0] + detail::SCALING * dx) / dx) + 1;
    auto ny = static_cast<long>((dims[1] + detail::SCALING * dx) / dx) + 1;
    auto nz = static_cast<long>((dims[2] + detail::SCALING * dx) / dx) + 1;

    std::vector<long> shape = {nx, ny, nz};
    auto n_vertices = static_cast<size_t>(nx * ny * nz);
    auto n_elements = static_cast<size_t>((nx - 1) * (ny - 1) * (nz - 1));

    std::vector<Vertex> vertices(n_vertices);
    std::vector<SquareBox3D> elements;
    elements.reserve(n_elements);

    size_t vertex_index = 0;
    auto x = orig[0];
    for (long i = 0; i < nx; ++i) {
      auto y = orig[1];
      for (long j = 0; j < ny; ++j) {
        auto z = orig[2];
        for (long k = 0; k < nz; ++k) {
          vertices[vertex_index] = Vertex{{x, y, z}};
          if (i < nx - 1 && j < ny - 1 && k < nz - 1) {
            elements.push_back(make_square_box_3d(dx, {x, y, z}));
          }
          ++vertex_index;
          z += dx;
        }
        y += dx;
      }
      x += dx;
    }

    return UniformMesh3D{
        .dx = dx,
        .dimensions = std::move(dims),
        .origin = std::move(orig),
        .shape = std::move(shape),
        .vertices = std::move(vertices),
        .elements = std::move(elements),
    };
  }

  // Free functions on concrete mesh types

  [[nodiscard]] inline auto volume(const UniformMesh2D& mesh) -> double {
    return mesh.dimensions[0] * mesh.dimensions[1];
  }

  [[nodiscard]] inline auto volume(const UniformMesh3D& mesh) -> double {
    return mesh.dimensions[0] * mesh.dimensions[1] * mesh.dimensions[2];
  }

  [[nodiscard]] inline auto element_volume(const UniformMesh2D& mesh) -> double { return mesh.dx * mesh.dx; }

  [[nodiscard]] inline auto element_volume(const UniformMesh3D& mesh) -> double {
    return mesh.dx * mesh.dx * mesh.dx;
  }

  [[nodiscard]] inline auto spacing(const UniformMesh2D& mesh) -> double { return mesh.dx; }

  [[nodiscard]] inline auto spacing(const UniformMesh3D& mesh) -> double { return mesh.dx; }

  // Free functions on variant Mesh

  [[nodiscard]] inline auto volume(const Mesh& mesh) -> double {
    return std::visit(
        [](const auto& m) -> double {
          using T = std::decay_t<decltype(m)>;
          if constexpr (std::is_same_v<T, UniformMesh2D>) {
            return m.dimensions[0] * m.dimensions[1];
          } else {
            return m.dimensions[0] * m.dimensions[1] * m.dimensions[2];
          }
        },
        mesh
    );
  }

  [[nodiscard]] inline auto element_volume(const Mesh& mesh) -> double {
    return std::visit(
        [](const auto& m) -> double {
          using T = std::decay_t<decltype(m)>;
          if constexpr (std::is_same_v<T, UniformMesh2D>) {
            return m.dx * m.dx;
          } else {
            return m.dx * m.dx * m.dx;
          }
        },
        mesh
    );
  }

  [[nodiscard]] inline auto spacing(const Mesh& mesh) -> double {
    return std::visit([](const auto& m) { return m.dx; }, mesh);
  }

  [[nodiscard]] inline auto flat_index(const Mesh& mesh, std::vector<long> idx) -> long {
    return std::visit(
        [&idx](const auto& m) -> long {
          detail::validate_indices(idx, m.shape);
          return detail::flat_index_impl(idx, m.shape);
        },
        mesh
    );
  }

  [[nodiscard]] inline auto cartesian_index(const Mesh& mesh, long idx) -> std::vector<long> {
    return std::visit(
        [idx](const auto& m) { return detail::cartesian_index_impl(idx, m.shape); }, mesh
    );
  }

  [[nodiscard]] inline auto vertex(const Mesh& mesh, std::vector<long> idx) -> const Vertex& {
    return std::visit(
        [&idx](const auto& m) -> const Vertex& {
          detail::validate_indices(idx, m.shape);
          auto flat = detail::flat_index_impl(idx, m.shape);
          return m.vertices.at(static_cast<size_t>(flat));
        },
        mesh
    );
  }

  [[nodiscard]] inline auto wrap(const Mesh& mesh, const Vertex& position) -> Vertex {
    return std::visit(
        [&position](const auto& m) -> Vertex {
          auto coords = position.coordinates;
          for (size_t d = 0; d < coords.size(); ++d) {
            coords[d] = std::fmod(coords[d], m.dimensions[d]);
            if (coords[d] < 0.0) {
              coords[d] += m.dimensions[d];
            }
          }
          return Vertex{std::move(coords)};
        },
        mesh
    );
  }

}  // namespace dft::geometry

#endif  // DFT_GEOMETRY_MESH_HPP
