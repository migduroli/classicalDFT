#ifndef DFT_GEOMETRY_VERTEX_HPP
#define DFT_GEOMETRY_VERTEX_HPP

#include <ostream>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace dft::geometry {

  struct Vertex {
    std::vector<double> coordinates;

    [[nodiscard]] auto operator[](int k) const -> const double& { return coordinates.at(static_cast<size_t>(k)); }

    auto operator[](int k) -> double& { return coordinates.at(static_cast<size_t>(k)); }

    [[nodiscard]] auto dimension() const -> int { return static_cast<int>(coordinates.size()); }
  };

  [[nodiscard]] inline auto operator+(const Vertex& a, const Vertex& b) -> Vertex {
    if (a.dimension() != b.dimension()) {
      throw std::invalid_argument("Cannot add vertices of different dimensions");
    }
    std::vector<double> result;
    result.reserve(a.coordinates.size());
    for (auto [ai, bi] : std::views::zip(a.coordinates, b.coordinates)) {
      result.push_back(ai + bi);
    }
    return Vertex{ std::move(result) };
  }

  [[nodiscard]] inline auto operator-(const Vertex& a, const Vertex& b) -> Vertex {
    if (a.dimension() != b.dimension()) {
      throw std::invalid_argument("Cannot subtract vertices of different dimensions");
    }
    std::vector<double> result;
    result.reserve(a.coordinates.size());
    for (auto [ai, bi] : std::views::zip(a.coordinates, b.coordinates)) {
      result.push_back(ai - bi);
    }
    return Vertex{ std::move(result) };
  }

  inline auto operator<<(std::ostream& os, const Vertex& v) -> std::ostream& {
    os << "(";
    for (size_t i = 0; i < v.coordinates.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << v.coordinates[i];
    }
    os << ")";
    return os;
  }

}  // namespace dft::geometry

#endif  // DFT_GEOMETRY_VERTEX_HPP
