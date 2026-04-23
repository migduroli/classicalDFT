#ifndef DFT_PHYSICS_WALLS_HPP
#define DFT_PHYSICS_WALLS_HPP

#include "dft/grid.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <variant>

namespace dft::physics::walls {

  // Concrete wall potential: Lennard-Jones 9-3 integrated planar wall.
  // V(z) = (2π/3) ρ_w ε σ³ [ (2/15)(σ/z)⁹ − (σ/z)³ ]

  struct LJ93 {
    double sigma{1.0};
    double epsilon{1.0};
    double density{1.0};
    double cutoff{5.0};

    [[nodiscard]] auto energy(double distance) const -> double {
      double ratio = sigma / distance;
      double ratio3 = ratio * ratio * ratio;
      double prefactor = (2.0 * std::numbers::pi / 3.0) * density * epsilon * std::pow(sigma, 3);
      return prefactor * ((2.0 / 15.0) * ratio3 * ratio3 * ratio3 - ratio3);
    }

    [[nodiscard]] auto field(const Grid& grid, int axis, bool lower) const -> arma::vec {
      arma::vec result(static_cast<arma::uword>(grid.total_points()), arma::fill::zeros);
      double r_cut = (cutoff > 0.0) ? cutoff : 0.5 * grid.box_size[axis];
      double v_cut = energy(r_cut);

      for (long ix = 0; ix < grid.shape[0]; ++ix) {
        for (long iy = 0; iy < grid.shape[1]; ++iy) {
          for (long iz = 0; iz < grid.shape[2]; ++iz) {
            std::array<long, 3> index = {ix, iy, iz};
            double distance = lower ? (static_cast<double>(index[axis]) + 0.5) * grid.dx
                                    : (static_cast<double>(grid.shape[axis] - index[axis]) - 0.5) * grid.dx;
            double value = (distance < r_cut) ? energy(distance) - v_cut : 0.0;
            result(static_cast<arma::uword>(grid.flat_index(ix, iy, iz))) = value;
          }
        }
      }
      return result;
    }

    [[nodiscard]] auto attachment_distance() const -> double { return sigma * std::pow(2.0 / 5.0, 1.0 / 6.0); }
  };

  // Variant wrapper class: hides the concrete wall type, exposes unified interface.
  // std::monostate represents "no wall" (inactive).

  class WallPotential {
   public:
    int axis{2};
    bool lower{true};

    WallPotential() = default;

    template <typename T>
      requires(!std::is_same_v<std::decay_t<T>, WallPotential>)
    WallPotential(T concrete, int wall_axis, bool wall_lower)
        : axis(wall_axis), lower(wall_lower), data_(std::move(concrete)) {}

    [[nodiscard]] auto is_active() const -> bool { return !std::holds_alternative<std::monostate>(data_); }

    [[nodiscard]] auto energy(double distance) const -> double {
      return std::visit(
          [distance](const auto& w) -> double {
            if constexpr (std::is_same_v<std::decay_t<decltype(w)>, std::monostate>) {
              return 0.0;
            } else {
              return w.energy(distance);
            }
          },
          data_
      );
    }

    [[nodiscard]] auto field(const Grid& grid) const -> arma::vec {
      return std::visit(
          [&](const auto& w) -> arma::vec {
            if constexpr (std::is_same_v<std::decay_t<decltype(w)>, std::monostate>) {
              return arma::vec(static_cast<arma::uword>(grid.total_points()), arma::fill::zeros);
            } else {
              return w.field(grid, axis, lower);
            }
          },
          data_
      );
    }

    [[nodiscard]] auto distance(const Grid& grid, long index) const -> double {
      return lower ? (static_cast<double>(index) + 0.5) * grid.dx
                   : (static_cast<double>(grid.shape[axis] - index) - 0.5) * grid.dx;
    }

    [[nodiscard]] auto attachment_distance(const Grid& grid) const -> double {
      double d = std::visit(
          [](const auto& w) -> double {
            if constexpr (std::is_same_v<std::decay_t<decltype(w)>, std::monostate>) {
              return 0.0;
            } else {
              return w.attachment_distance();
            }
          },
          data_
      );
      return std::max(d, 0.5 * grid.dx);
    }

    [[nodiscard]] auto reservoir_mask(const Grid& grid) const -> arma::uvec {
      if (!is_active())
        return grid.boundary_mask();
      return grid.face_mask(axis, !lower);
    }

    // Replace density with background values in repulsive wall regions.

    [[nodiscard]] auto suppress_excess(arma::vec rho, const arma::vec& background, const Grid& grid) const
        -> arma::vec {
      if (!is_active())
        return rho;
      arma::uvec repulsive = arma::find(field(grid) > 0.0);
      rho.elem(repulsive) = background.elem(repulsive);
      return rho;
    }

    using VariantType = std::variant<std::monostate, LJ93>;

    [[nodiscard]] auto variant() const -> const VariantType& { return data_; }

   private:
    VariantType data_;
  };

} // namespace dft::physics::walls

namespace dft::physics {

  // Gravitational external field: V_g(z) = g* · z.

  struct Gravity {
    double strength{0.0};

    [[nodiscard]] auto is_active() const -> bool { return std::abs(strength) > 1e-30; }

    [[nodiscard]] auto field(const Grid& grid, const walls::WallPotential& wall) const -> arma::vec {
      arma::vec result(static_cast<arma::uword>(grid.total_points()), arma::fill::zeros);
      if (!is_active())
        return result;

      int grav_axis = wall.is_active() ? wall.axis : 2;

      for (long ix = 0; ix < grid.shape[0]; ++ix) {
        for (long iy = 0; iy < grid.shape[1]; ++iy) {
          for (long iz = 0; iz < grid.shape[2]; ++iz) {
            std::array<long, 3> index = {ix, iy, iz};
            double z = (static_cast<double>(index[grav_axis]) + 0.5) * grid.dx;
            result(static_cast<arma::uword>(grid.flat_index(ix, iy, iz))) = strength * z;
          }
        }
      }
      return result;
    }
  };

} // namespace dft::physics

#endif // DFT_PHYSICS_WALLS_HPP
