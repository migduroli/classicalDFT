#ifndef DFT_GEOMETRY_ELEMENT_HPP
#define DFT_GEOMETRY_ELEMENT_HPP

#include "dft/geometry/vertex.hpp"

#include <variant>
#include <vector>

namespace dft::geometry {

  enum class Direction {
    X = 0,
    Y = 1,
    Z = 2
  };

  struct Element {
    std::vector<Vertex> vertices;
  };

  struct SquareBox2D {
    double length;
    std::vector<double> origin;
    std::vector<Vertex> vertices;

    [[nodiscard]] auto volume() const -> double { return length * length; }
  };

  struct SquareBox3D {
    double length;
    std::vector<double> origin;
    std::vector<Vertex> vertices;

    [[nodiscard]] auto volume() const -> double { return length * length * length; }
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

}  // namespace dft::geometry

#endif  // DFT_GEOMETRY_ELEMENT_HPP
