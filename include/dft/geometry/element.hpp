#ifndef DFT_GEOMETRY_ELEMENT_HPP
#define DFT_GEOMETRY_ELEMENT_HPP

#include "dft/geometry/vertex.hpp"

#include <variant>
#include <vector>

namespace dft::geometry {

  enum class Direction { X = 0, Y = 1, Z = 2 };

  struct Element {
    std::vector<Vertex> vertices;
  };

  struct SquareBox2D {
    double length;
    std::vector<double> origin;
    std::vector<Vertex> vertices;
  };

  struct SquareBox3D {
    double length;
    std::vector<double> origin;
    std::vector<Vertex> vertices;
  };

  using ElementVariant = std::variant<Element, SquareBox2D, SquareBox3D>;

  [[nodiscard]] inline auto make_square_box_2d(double length, const std::vector<double>& origin) -> SquareBox2D {
    auto x = origin.at(0);
    auto y = origin.at(1);
    return SquareBox2D{
        .length = length,
        .origin = origin,
        .vertices =
            {
                Vertex{{x, y}},
                Vertex{{x + length, y}},
                Vertex{{x + length, y + length}},
                Vertex{{x, y + length}},
            },
    };
  }

  [[nodiscard]] inline auto make_square_box_3d(double length, const std::vector<double>& origin) -> SquareBox3D {
    auto x = origin.at(0);
    auto y = origin.at(1);
    auto z = origin.at(2);
    return SquareBox3D{
        .length = length,
        .origin = origin,
        .vertices =
            {
                Vertex{{x, y, z}},
                Vertex{{x + length, y, z}},
                Vertex{{x + length, y + length, z}},
                Vertex{{x, y + length, z}},
                Vertex{{x, y + length, z + length}},
                Vertex{{x + length, y + length, z + length}},
                Vertex{{x + length, y, z + length}},
                Vertex{{x, y, z + length}},
            },
    };
  }

  [[nodiscard]] inline auto volume(const SquareBox2D& box) -> double { return box.length * box.length; }

  [[nodiscard]] inline auto volume(const SquareBox3D& box) -> double {
    return box.length * box.length * box.length;
  }

  [[nodiscard]] inline auto volume(const ElementVariant& element) -> double {
    return std::visit(
        [](const auto& e) -> double {
          using T = std::decay_t<decltype(e)>;
          if constexpr (std::is_same_v<T, Element>) {
            return 0.0;
          } else if constexpr (std::is_same_v<T, SquareBox2D>) {
            return e.length * e.length;
          } else {
            return e.length * e.length * e.length;
          }
        },
        element
    );
  }

  [[nodiscard]] inline auto dimension(const ElementVariant& element) -> int {
    return std::visit(
        [](const auto& e) -> int {
          using T = std::decay_t<decltype(e)>;
          if constexpr (std::is_same_v<T, Element>) {
            return e.vertices.empty() ? 0 : dimension(e.vertices.front());
          } else if constexpr (std::is_same_v<T, SquareBox2D>) {
            return 2;
          } else {
            return 3;
          }
        },
        element
    );
  }

}  // namespace dft::geometry

#endif  // DFT_GEOMETRY_ELEMENT_HPP
