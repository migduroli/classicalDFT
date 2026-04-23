#ifndef DFT_GRID_HPP
#define DFT_GRID_HPP

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace dft {

  // A single mode in the half-complex Fourier grid.
  struct Wavevector {
    long idx;
    std::array<double, 3> k;

    [[nodiscard]] auto norm2() const noexcept -> double { return k[0] * k[0] + k[1] * k[1] + k[2] * k[2]; }

    [[nodiscard]] auto norm() const noexcept -> double { return std::sqrt(norm2()); }
  };

  // A 1D line slice through a 3D field on the grid.

  struct Slice1D {
    double time{0.0};
    std::string axis{"x"};
    std::vector<double> x;
    std::vector<double> values;
  };

  // A radially-averaged profile of a 3D field.

  struct RadialProfile {
    std::vector<double> r;
    std::vector<double> values;

    // Equimolar radius: where the radially-averaged profile crosses the
    // midpoint between the center density and the far-field density.

    [[nodiscard]] auto equimolar_radius() const -> double {
      if (r.size() < 4)
        return 0.0;

      double rho_in = values.front();

      std::size_t n = r.size();
      std::size_t n_far = std::max<std::size_t>(n / 4, 1);
      double rho_out = 0.0;
      for (std::size_t i = n - n_far; i < n; ++i)
        rho_out += values[i];
      rho_out /= static_cast<double>(n_far);

      if (rho_in - rho_out < 0.01)
        return 0.0;

      double rho_half = 0.5 * (rho_in + rho_out);

      for (std::size_t i = 1; i < n; ++i) {
        if (values[i] <= rho_half && values[i - 1] > rho_half) {
          double f = (values[i - 1] - rho_half) / (values[i - 1] - values[i]);
          return r[i - 1] + f * (r[i] - r[i - 1]);
        }
      }

      return 0.0;
    }
  };

  // A 2D cross-section slice through a 3D field on the grid.

  struct Slice2D {
    double time{0.0};
    long nx{0};
    long ny{0};
    std::string x_label{"x"};
    std::string y_label{"y"};
    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;
    std::vector<std::vector<double>> z;
  };

  // Lightweight grid description for DFT on a 3D uniform mesh.
  // Each axis can be periodic or non-periodic. Non-periodic axes use
  // zero-padded convolutions (doubled size) and Neumann-BC diffusion.
  // Points per axis: N_i = floor(L_i / dx). Flat indexing is row-major
  // with z varying fastest: index = iz + Nz*(iy + Ny*ix).
  struct Grid {
    double dx;
    std::array<double, 3> box_size;
    std::array<long, 3> shape;
    std::array<bool, 3> periodic{true, true, true};

    [[nodiscard]] constexpr auto cell_volume() const noexcept -> double { return dx * dx * dx; }

    [[nodiscard]] constexpr auto total_points() const noexcept -> long { return shape[0] * shape[1] * shape[2]; }

    [[nodiscard]] constexpr auto flat_index(long ix, long iy, long iz) const noexcept -> long {
      return iz + shape[2] * (iy + shape[1] * ix);
    }

    [[nodiscard]] constexpr auto is_fully_periodic() const noexcept -> bool {
      return periodic[0] && periodic[1] && periodic[2];
    }

    // Shape used for FFT convolutions: doubled in non-periodic directions
    // to convert circular convolution into linear convolution.

    [[nodiscard]] constexpr auto convolution_shape() const noexcept -> std::array<long, 3> {
      return {
          periodic[0] ? shape[0] : 2 * shape[0],
          periodic[1] ? shape[1] : 2 * shape[1],
          periodic[2] ? shape[2] : 2 * shape[2],
      };
    }

    // A fully-periodic grid with the convolution shape, for use as the
    // target grid when generating FFT weights.

    [[nodiscard]] auto convolution_grid() const -> Grid {
      auto cs = convolution_shape();
      return Grid{
          .dx = dx,
          .box_size =
              {static_cast<double>(cs[0]) * dx, static_cast<double>(cs[1]) * dx, static_cast<double>(cs[2]) * dx},
          .shape = cs,
      };
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

    // Face mask: 1 for points on a single face of the box, 0 elsewhere.

    [[nodiscard]] auto face_mask(int axis, bool lower) const -> arma::uvec {
      arma::uvec mask(static_cast<arma::uword>(total_points()), arma::fill::zeros);
      long face_index = lower ? 0 : shape[axis] - 1;
      for (long ix = 0; ix < shape[0]; ++ix) {
        for (long iy = 0; iy < shape[1]; ++iy) {
          for (long iz = 0; iz < shape[2]; ++iz) {
            std::array<long, 3> idx = {ix, iy, iz};
            if (idx[axis] == face_index) {
              mask(static_cast<arma::uword>(flat_index(ix, iy, iz))) = 1;
            }
          }
        }
      }
      return mask;
    }

    // Radial distances from a center point to every grid node.

    [[nodiscard]] auto radial_distances(const std::array<double, 3>& center) const -> arma::vec {
      auto n = static_cast<arma::uword>(total_points());
      arma::vec r2(n, arma::fill::zeros);
      for (int d = 0; d < 3; ++d) {
        long nd = shape[d];
        arma::vec coord(static_cast<arma::uword>(nd));
        for (long i = 0; i < nd; ++i) {
          coord(static_cast<arma::uword>(i)) = i * dx - center[d];
        }
        arma::uword stride = 1;
        for (int dd = d + 1; dd < 3; ++dd) {
          stride *= static_cast<arma::uword>(shape[dd]);
        }
        arma::uword repeat = n / (static_cast<arma::uword>(nd) * stride);
        for (arma::uword rep = 0; rep < repeat; ++rep) {
          for (arma::uword i = 0; i < static_cast<arma::uword>(nd); ++i) {
            arma::uword offset = rep * static_cast<arma::uword>(nd) * stride + i * stride;
            r2.subvec(offset, offset + stride - 1) += coord(i) * coord(i);
          }
        }
      }
      return arma::sqrt(r2);
    }

    // Radial distances from the box center.

    [[nodiscard]] auto radial_distances() const -> arma::vec {
      return radial_distances({box_size[0] / 2.0, box_size[1] / 2.0, box_size[2] / 2.0});
    }

    // Zero-pad a physical-grid field to convolution-grid size.

    [[nodiscard]] auto pad(const arma::vec& field) const -> arma::vec {
      if (is_fully_periodic())
        return field;
      auto cs = convolution_shape();
      long nx = shape[0], ny = shape[1], nz = shape[2];
      long pz = cs[2], py = cs[1];
      arma::vec padded(static_cast<arma::uword>(cs[0] * cs[1] * cs[2]), arma::fill::zeros);
      for (long ix = 0; ix < nx; ++ix) {
        for (long iy = 0; iy < ny; ++iy) {
          for (long iz = 0; iz < nz; ++iz) {
            padded(static_cast<arma::uword>(iz + pz * (iy + py * ix))
            ) = field(static_cast<arma::uword>(iz + nz * (iy + ny * ix)));
          }
        }
      }
      return padded;
    }

    // Extract the physical-grid portion from a convolution-grid field.

    [[nodiscard]] auto unpad(const arma::vec& padded) const -> arma::vec {
      if (is_fully_periodic())
        return padded;
      auto cs = convolution_shape();
      long nx = shape[0], ny = shape[1], nz = shape[2];
      long pz = cs[2], py = cs[1];
      arma::vec result(static_cast<arma::uword>(total_points()));
      for (long ix = 0; ix < nx; ++ix) {
        for (long iy = 0; iy < ny; ++iy) {
          for (long iz = 0; iz < nz; ++iz) {
            result(static_cast<arma::uword>(iz + nz * (iy + ny * ix))
            ) = padded(static_cast<arma::uword>(iz + pz * (iy + py * ix)));
          }
        }
      }
      return result;
    }

    // Even-extend (mirror) a physical-grid field to convolution-grid size.

    [[nodiscard]] auto mirror_extend(const arma::vec& field) const -> arma::vec {
      if (is_fully_periodic())
        return field;
      arma::vec result = pad(field);
      auto cs = convolution_shape();
      long nx = shape[0], ny = shape[1], nz = shape[2];
      long pz = cs[2], py = cs[1];
      if (!periodic[2]) {
        for (long ix = 0; ix < nx; ++ix) {
          for (long iy = 0; iy < ny; ++iy) {
            for (long iz = 0; iz < nz; ++iz) {
              auto src = static_cast<arma::uword>(iz + pz * (iy + py * ix));
              auto dst = static_cast<arma::uword>((2 * nz - 1 - iz) + pz * (iy + py * ix));
              result(dst) = result(src);
            }
          }
        }
      }
      if (!periodic[1]) {
        for (long ix = 0; ix < nx; ++ix) {
          for (long iy = 0; iy < ny; ++iy) {
            long miy = 2 * ny - 1 - iy;
            for (long iz = 0; iz < cs[2]; ++iz) {
              auto src = static_cast<arma::uword>(iz + pz * (iy + py * ix));
              auto dst = static_cast<arma::uword>(iz + pz * (miy + py * ix));
              result(dst) = result(src);
            }
          }
        }
      }
      if (!periodic[0]) {
        for (long ix = 0; ix < nx; ++ix) {
          long mix = 2 * nx - 1 - ix;
          for (long iy = 0; iy < cs[1]; ++iy) {
            for (long iz = 0; iz < cs[2]; ++iz) {
              auto src = static_cast<arma::uword>(iz + pz * (iy + py * ix));
              auto dst = static_cast<arma::uword>(iz + pz * (iy + py * mix));
              result(dst) = result(src);
            }
          }
        }
      }
      return result;
    }

    // Wrap a coordinate into [0, box_length).

    [[nodiscard]] static auto wrap_periodic(double x, double box_length) -> double {
      double wrapped = std::fmod(x, box_length);
      return (wrapped < 0.0) ? wrapped + box_length : wrapped;
    }

    // Minimum image convention for a coordinate difference.

    [[nodiscard]] static auto minimum_image(double delta, double box_length) -> double {
      double wrapped = std::fmod(delta, box_length);
      if (wrapped > 0.5 * box_length)
        wrapped -= box_length;
      if (wrapped < -0.5 * box_length)
        wrapped += box_length;
      return wrapped;
    }

    // Axis name as string.

    [[nodiscard]] static auto axis_name(int axis) -> std::string {
      if (axis == 0)
        return "x";
      if (axis == 1)
        return "y";
      if (axis == 2)
        return "z";
      throw std::runtime_error("Invalid axis index");
    }

    // Extract a 2D plane slice through the grid at a fixed index along the
    // third axis. axes = {horizontal, vertical}, e.g. {0,2} for XZ plane.

    [[nodiscard]] auto plane_slice(
        const arma::vec& field,
        const std::array<int, 2>& axes,
        std::optional<long> fixed_index = std::nullopt
    ) const -> Slice2D {
      if (axes[0] == axes[1])
        throw std::runtime_error("Plane axes must be distinct");

      int fixed_axis = 3 - axes[0] - axes[1];
      std::array<long, 3> index = {shape[0] / 2, shape[1] / 2, shape[2] / 2};
      long fixed = fixed_index.value_or(shape[fixed_axis] / 2);
      fixed = std::clamp(fixed, 0L, shape[fixed_axis] - 1);
      long n_x = shape[axes[0]];
      long n_y = shape[axes[1]];
      double cx = box_size[axes[0]] / 2.0;
      double cy = box_size[axes[1]] / 2.0;

      auto sny = static_cast<std::size_t>(n_y);
      auto snx = static_cast<std::size_t>(n_x);
      std::vector<std::vector<double>> xg(sny, std::vector<double>(snx));
      std::vector<std::vector<double>> yg(sny, std::vector<double>(snx));
      std::vector<std::vector<double>> zg(sny, std::vector<double>(snx));

      for (long iy = 0; iy < n_y; ++iy) {
        index[axes[1]] = iy;
        for (long ix = 0; ix < n_x; ++ix) {
          index[axes[0]] = ix;
          index[fixed_axis] = fixed;
          xg[static_cast<std::size_t>(iy)][static_cast<std::size_t>(ix)] = ix * dx - cx;
          yg[static_cast<std::size_t>(iy)][static_cast<std::size_t>(ix)] = iy * dx - cy;
          zg[static_cast<std::size_t>(iy)][static_cast<std::size_t>(ix)] =
              field(static_cast<arma::uword>(flat_index(index[0], index[1], index[2])));
        }
      }

      return {
          .time = 0.0,
          .nx = n_x,
          .ny = n_y,
          .x_label = axis_name(axes[0]),
          .y_label = axis_name(axes[1]),
          .x = std::move(xg),
          .y = std::move(yg),
          .z = std::move(zg),
      };
    }

    // Convenience shortcuts for common planes.

    [[nodiscard]] auto xy_slice(const arma::vec& field, std::optional<long> fixed_z = std::nullopt) const -> Slice2D {
      return plane_slice(field, {0, 1}, fixed_z);
    }

    [[nodiscard]] auto xz_slice(const arma::vec& field, std::optional<long> fixed_y = std::nullopt) const -> Slice2D {
      return plane_slice(field, {0, 2}, fixed_y);
    }

    [[nodiscard]] auto yz_slice(const arma::vec& field, std::optional<long> fixed_x = std::nullopt) const -> Slice2D {
      return plane_slice(field, {1, 2}, fixed_x);
    }

    // Extract a 1D line slice through the box center along one axis.

    [[nodiscard]] auto line_slice(const arma::vec& field, int axis) const -> Slice1D {
      std::array<long, 3> index = {shape[0] / 2, shape[1] / 2, shape[2] / 2};
      long n = shape[axis];
      double center = box_size[axis] / 2.0;

      std::vector<double> x_out(static_cast<std::size_t>(n));
      std::vector<double> v_out(static_cast<std::size_t>(n));
      for (long i = 0; i < n; ++i) {
        index[axis] = i;
        x_out[static_cast<std::size_t>(i)] = i * dx - center;
        v_out[static_cast<std::size_t>(i)] = field(static_cast<arma::uword>(flat_index(index[0], index[1], index[2])));
      }
      return {.time = 0.0, .axis = axis_name(axis), .x = x_out, .values = v_out};
    }

    [[nodiscard]] auto x_line(const arma::vec& field) const -> Slice1D { return line_slice(field, 0); }

    [[nodiscard]] auto y_line(const arma::vec& field) const -> Slice1D { return line_slice(field, 1); }

    [[nodiscard]] auto z_line(const arma::vec& field) const -> Slice1D { return line_slice(field, 2); }

    // Radially-averaged profile of a field around a center point.

    [[nodiscard]] auto radial_profile(const arma::vec& field, const std::array<double, 3>& center) const
        -> RadialProfile {
      arma::vec r = radial_distances(center);
      double r_max = std::min({box_size[0], box_size[1], box_size[2]}) / 2.0;
      auto n_bins = static_cast<arma::uword>(r_max / dx);

      arma::vec bin_sum(n_bins, arma::fill::zeros);
      arma::uvec bin_count(n_bins, arma::fill::zeros);
      arma::uvec bins = arma::conv_to<arma::uvec>::from(arma::floor(r / dx));

      for (arma::uword i = 0; i < field.n_elem; ++i) {
        if (bins(i) < n_bins) {
          bin_sum(bins(i)) += field(i);
          bin_count(bins(i)) += 1;
        }
      }

      std::vector<double> r_out, v_out;
      for (arma::uword i = 0; i < n_bins; ++i) {
        if (bin_count(i) > 0) {
          r_out.push_back((i + 0.5) * dx);
          v_out.push_back(bin_sum(i) / static_cast<double>(bin_count(i)));
        }
      }

      return {.r = r_out, .values = v_out};
    }

    [[nodiscard]] auto radial_profile(const arma::vec& field) const -> RadialProfile {
      return radial_profile(field, {box_size[0] / 2.0, box_size[1] / 2.0, box_size[2] / 2.0});
    }

    // Field value at the box center.

    [[nodiscard]] auto center_value(const arma::vec& field) const -> double {
      return field(static_cast<arma::uword>(flat_index(shape[0] / 2, shape[1] / 2, shape[2] / 2)));
    }

    // Field value at the grid point nearest to a given coordinate.

    [[nodiscard]] auto value_at(const arma::vec& field, const std::array<double, 3>& point) const -> double {
      long ix = std::clamp(static_cast<long>(std::llround(point[0] / dx)), 0L, shape[0] - 1);
      long iy = std::clamp(static_cast<long>(std::llround(point[1] / dx)), 0L, shape[1] - 1);
      long iz = std::clamp(static_cast<long>(std::llround(point[2] / dx)), 0L, shape[2] - 1);
      return field(static_cast<arma::uword>(flat_index(ix, iy, iz)));
    }

    // Average of field values at points selected by a mask.

    [[nodiscard]] static auto face_average(const arma::vec& field, const arma::uvec& mask) -> double {
      double sum = 0.0;
      arma::uword count = 0;
      for (arma::uword i = 0; i < field.n_elem; ++i) {
        if (mask(i)) {
          sum += field(i);
          ++count;
        }
      }
      return (count > 0) ? sum / static_cast<double>(count) : 0.0;
    }
  };

  // Validated factory. Checks that each box dimension is commensurate with dx
  // (integer multiple within a small tolerance).
  [[nodiscard]] inline auto
  make_grid(double dx, std::array<double, 3> box, std::array<bool, 3> periodic = {true, true, true}) -> Grid {
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

    return Grid{.dx = dx, .box_size = box, .shape = shape, .periodic = periodic};
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

  // Zero-pad a physical-grid field to convolution-grid size.
  // Retained for backward compatibility; prefer grid.pad(field).

  [[nodiscard]] inline auto pad(const arma::vec& field, const Grid& grid) -> arma::vec {
    return grid.pad(field);
  }

  // Extract the physical-grid portion from a convolution-grid field.
  // Retained for backward compatibility; prefer grid.unpad(padded).

  [[nodiscard]] inline auto unpad(const arma::vec& padded, const Grid& grid) -> arma::vec {
    return grid.unpad(padded);
  }

  // Even-extend (mirror) a physical-grid field to convolution-grid size.
  // Retained for backward compatibility; prefer grid.mirror_extend(field).

  [[nodiscard]] inline auto mirror_extend(const arma::vec& field, const Grid& grid) -> arma::vec {
    return grid.mirror_extend(field);
  }

} // namespace dft

#endif // DFT_GRID_HPP
