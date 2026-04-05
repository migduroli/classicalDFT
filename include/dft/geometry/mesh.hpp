#ifndef DFT_GEOMETRY_MESH_HPP
#define DFT_GEOMETRY_MESH_HPP

#include "dft/geometry/element.hpp"

#include <cmath>
#include <ranges>
#include <stdexcept>
#include <variant>
#include <vector>

namespace dft::geometry {

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

    [[nodiscard]] inline auto validate_indices(std::vector<long> idxs, const std::vector<long>& shape)
        -> std::vector<long> {
      if (idxs.size() != shape.size()) {
        throw std::invalid_argument("Index dimension does not match mesh dimension");
      }
      for (auto [idx, dim] : std::views::zip(idxs, shape)) {
        if (idx < 0) {
          idx += dim;
        }
        if (idx < 0 || idx >= dim) {
          throw std::out_of_range("Index out of bounds in mesh indexer");
        }
      }
      return idxs;
    }

  } // namespace detail

  struct UniformMesh2D {
    double dx;
    std::vector<double> dimensions;
    std::vector<double> origin;
    std::vector<long> shape;
    std::vector<Vertex> vertices;
    std::vector<SquareBox2D> elements;

    [[nodiscard]] auto volume() const -> double { return dimensions[0] * dimensions[1]; }

    [[nodiscard]] auto element_volume() const -> double { return dx * dx; }

    [[nodiscard]] auto spacing() const -> double { return dx; }

    [[nodiscard]] auto flat_index(std::vector<long> idx) const -> long {
      auto wrapped = detail::validate_indices(std::move(idx), shape);
      return detail::flat_index_impl(wrapped, shape);
    }

    [[nodiscard]] auto cartesian_index(long idx) const -> std::vector<long> {
      return detail::cartesian_index_impl(idx, shape);
    }

    [[nodiscard]] auto vertex(std::vector<long> idx) const -> const Vertex& {
      auto wrapped = detail::validate_indices(std::move(idx), shape);
      auto flat = detail::flat_index_impl(wrapped, shape);
      return vertices.at(static_cast<size_t>(flat));
    }

    [[nodiscard]] auto wrap(const Vertex& position) const -> Vertex {
      auto coords = position.coordinates;
      for (auto [coord, dim] : std::views::zip(coords, dimensions)) {
        coord = std::fmod(coord, dim);
        if (coord < 0.0) {
          coord += dim;
        }
      }
      return Vertex{std::move(coords)};
    }
  };

  struct UniformMesh3D {
    double dx;
    std::vector<double> dimensions;
    std::vector<double> origin;
    std::vector<long> shape;
    std::vector<Vertex> vertices;
    std::vector<SquareBox3D> elements;

    [[nodiscard]] auto volume() const -> double { return dimensions[0] * dimensions[1] * dimensions[2]; }

    [[nodiscard]] auto element_volume() const -> double { return dx * dx * dx; }

    [[nodiscard]] auto spacing() const -> double { return dx; }

    [[nodiscard]] auto flat_index(std::vector<long> idx) const -> long {
      auto wrapped = detail::validate_indices(std::move(idx), shape);
      return detail::flat_index_impl(wrapped, shape);
    }

    [[nodiscard]] auto cartesian_index(long idx) const -> std::vector<long> {
      return detail::cartesian_index_impl(idx, shape);
    }

    [[nodiscard]] auto vertex(std::vector<long> idx) const -> const Vertex& {
      auto wrapped = detail::validate_indices(std::move(idx), shape);
      auto flat = detail::flat_index_impl(wrapped, shape);
      return vertices.at(static_cast<size_t>(flat));
    }

    [[nodiscard]] auto wrap(const Vertex& position) const -> Vertex {
      auto coords = position.coordinates;
      for (auto [coord, dim] : std::views::zip(coords, dimensions)) {
        coord = std::fmod(coord, dim);
        if (coord < 0.0) {
          coord += dim;
        }
      }
      return Vertex{std::move(coords)};
    }
  };

  using Mesh = std::variant<UniformMesh2D, UniformMesh3D>;

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

} // namespace dft::geometry

#endif // DFT_GEOMETRY_MESH_HPP
