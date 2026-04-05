#ifndef DFT_GRID_HPP
#define DFT_GRID_HPP

#include <armadillo>
#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace dft {

  // A single mode in the half-complex Fourier grid.
  struct Wavevector {
    long idx;
    std::array<double, 3> k;

    [[nodiscard]] auto norm2() const noexcept -> double { return k[0] * k[0] + k[1] * k[1] + k[2] * k[2]; }

    [[nodiscard]] auto norm() const noexcept -> double { return std::sqrt(norm2()); }
  };

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

    // Iterate over the half-complex Fourier grid, calling fn(Wavevector)
    // for each mode. Encapsulates index layout and frequency wrapping.

    template <typename F> void for_each_wavevector(F&& fn) const {
      long nx = shape[0];
      long ny = shape[1];
      long nz = shape[2];
      long nz_half = nz / 2 + 1;

      double dk_x = 2.0 * std::numbers::pi / (static_cast<double>(nx) * dx);
      double dk_y = 2.0 * std::numbers::pi / (static_cast<double>(ny) * dx);
      double dk_z = 2.0 * std::numbers::pi / (static_cast<double>(nz) * dx);

      for (long ix = 0; ix < nx; ++ix) {
        double kx = dk_x * static_cast<double>(ix <= nx / 2 ? ix : ix - nx);
        for (long iy = 0; iy < ny; ++iy) {
          double ky = dk_y * static_cast<double>(iy <= ny / 2 ? iy : iy - ny);
          for (long iz = 0; iz < nz_half; ++iz) {
            double kz = dk_z * static_cast<double>(iz);
            long idx = iz + nz_half * (iy + ny * ix);
            fn(Wavevector{.idx = idx, .k = {kx, ky, kz}});
          }
        }
      }
    }

    // Boundary mask: 1 for points on any face of the box, 0 for interior.

    [[nodiscard]] auto boundary_mask() const -> arma::uvec {
      auto n = static_cast<arma::uword>(total_points());
      arma::uvec mask(n, arma::fill::zeros);
      for (long ix = 0; ix < shape[0]; ++ix) {
        for (long iy = 0; iy < shape[1]; ++iy) {
          for (long iz = 0; iz < shape[2]; ++iz) {
            bool on_face =
                (ix == 0 || ix == shape[0] - 1 || iy == 0 || iy == shape[1] - 1 || iz == 0 || iz == shape[2] - 1);
            if (on_face) {
              mask(static_cast<arma::uword>(flat_index(ix, iy, iz))) = 1;
            }
          }
        }
      }
      return mask;
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

  // Homogeneous boundary: return forces with boundary points
  // replaced by their average.

  [[nodiscard]] inline auto homogeneous_boundary(const arma::vec& forces, const arma::uvec& mask) -> arma::vec {
    double sum = 0.0;
    arma::uword count = 0;
    for (arma::uword i = 0; i < forces.n_elem; ++i) {
      if (mask(i)) {
        sum += forces(i);
        count++;
      }
    }
    arma::vec result = forces;
    if (count > 0) {
      double avg = sum / static_cast<double>(count);
      for (arma::uword i = 0; i < result.n_elem; ++i) {
        if (mask(i))
          result(i) = avg;
      }
    }
    return result;
  }

  // Fixed boundary: return forces with boundary points set to zero.

  [[nodiscard]] inline auto fixed_boundary(const arma::vec& forces, const arma::uvec& mask) -> arma::vec {
    arma::vec result = forces;
    for (arma::uword i = 0; i < result.n_elem; ++i) {
      if (mask(i))
        result(i) = 0.0;
    }
    return result;
  }

} // namespace dft

#endif // DFT_GRID_HPP
