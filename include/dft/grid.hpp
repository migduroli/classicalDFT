#ifndef DFT_GRID_HPP
#define DFT_GRID_HPP

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dft {

  // Lightweight grid description for DFT on a 3D periodic uniform mesh.
  // Points per axis: N_i = floor(L_i / dx). Flat indexing is row-major
  // with z varying fastest: index = iz + Nz*(iy + Ny*ix).
  struct Grid {
    double dx;
    std::array<double, 3> box_size;
    std::array<long, 3> shape;

    [[nodiscard]] constexpr auto cell_volume() const noexcept -> double { return dx * dx * dx; }

    [[nodiscard]] constexpr auto total_points() const noexcept -> long { return shape[0] * shape[1] * shape[2]; }

    [[nodiscard]] constexpr auto flat_index(long ix, long iy, long iz) const noexcept -> long {
      return iz + shape[2] * (iy + shape[1] * ix);
    }
  };

  // Validated factory. Checks that each box dimension is commensurate with dx
  // (integer multiple within a small tolerance).
  [[nodiscard]] inline auto make_grid(double dx, std::array<double, 3> box) -> Grid {
    if (dx <= 0.0) {
      throw std::invalid_argument("make_grid: dx must be positive, got " + std::to_string(dx));
    }

    std::array<long, 3> shape{};
    for (int d = 0; d < 3; ++d) {
      if (box[d] <= 0.0) {
        throw std::invalid_argument("make_grid: box dimension must be positive, got " + std::to_string(box[d]));
      }
      double n = box[d] / dx;
      long ni = static_cast<long>(std::round(n));
      if (std::abs(n - static_cast<double>(ni)) > 1e-10) {
        throw std::invalid_argument(
            "make_grid: box dimension " + std::to_string(box[d]) + " is not commensurate with dx=" + std::to_string(dx)
        );
      }
      shape[d] = ni;
    }

    return Grid{.dx = dx, .box_size = box, .shape = shape};
  }

}  // namespace dft

#endif  // DFT_GRID_HPP
