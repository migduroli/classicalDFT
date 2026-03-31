#ifndef DFT_GEOMETRY_VERTEX_HPP
#define DFT_GEOMETRY_VERTEX_HPP

#include <ostream>
#include <stdexcept>
#include <vector>

namespace dft::geometry {

  struct Vertex {
    std::vector<double> coordinates;

    [[nodiscard]] auto operator[](int k) const -> const double& {
      return coordinates.at(static_cast<size_t>(k));
    }

    auto operator[](int k) -> double& {
      return coordinates.at(static_cast<size_t>(k));
    }
  };

  [[nodiscard]] inline auto dimension(const Vertex& v) -> int {
    return static_cast<int>(v.coordinates.size());
  }

  [[nodiscard]] inline auto operator+(const Vertex& a, const Vertex& b) -> Vertex {
    if (dimension(a) != dimension(b)) {
      throw std::invalid_argument("Cannot add vertices of different dimensions");
    }
    std::vector<double> result;
    result.reserve(a.coordinates.size());
    for (size_t i = 0; i < a.coordinates.size(); ++i) {
      result.push_back(a.coordinates[i] + b.coordinates[i]);
    }
    return Vertex{std::move(result)};
  }

  [[nodiscard]] inline auto operator-(const Vertex& a, const Vertex& b) -> Vertex {
    if (dimension(a) != dimension(b)) {
      throw std::invalid_argument("Cannot subtract vertices of different dimensions");
    }
    std::vector<double> result;
    result.reserve(a.coordinates.size());
    for (size_t i = 0; i < a.coordinates.size(); ++i) {
      result.push_back(a.coordinates[i] - b.coordinates[i]);
    }
    return Vertex{std::move(result)};
  }

  inline auto operator<<(std::ostream& os, const Vertex& v) -> std::ostream& {
    os << "(";
    for (size_t i = 0; i < v.coordinates.size(); ++i) {
      if (i > 0) os << ", ";
      os << v.coordinates[i];
    }
    os << ")";
    return os;
  }

}  // namespace dft::geometry

#endif  // DFT_GEOMETRY_VERTEX_HPP
