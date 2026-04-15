#ifndef DFT_FIELDS_HPP
#define DFT_FIELDS_HPP

#include "dft/grid.hpp"

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <numbers>

namespace dft {

  // A sharp radial step profile: rho_in inside radius, rho_out outside.

  struct StepProfile {
    double radius;
    double rho_in;
    double rho_out;

    [[nodiscard]] auto operator()(const arma::vec& distances) const -> arma::vec {
      arma::vec field(distances.n_elem, arma::fill::value(rho_out));
      field.elem(arma::find(distances < radius)).fill(rho_in);
      return field;
    }
  };

  // Smooth ellipsoidal envelope with tanh interface.

  struct EllipsoidalEnvelope {
    std::array<double, 3> radii;
    double interface_width{0.25};

    [[nodiscard]] auto operator()(const Grid& grid, const std::array<double, 3>& center) const -> arma::vec {
      auto n = static_cast<arma::uword>(grid.total_points());
      arma::vec envelope(n, arma::fill::zeros);

      double r_min = std::min({radii[0], radii[1], radii[2]});
      double width = std::max(interface_width, 0.5 * grid.dx);

      for (long ix = 0; ix < grid.shape[0]; ++ix) {
        double dx = ix * grid.dx - center[0];
        if (grid.periodic[0])
          dx = Grid::minimum_image(dx, grid.box_size[0]);
        for (long iy = 0; iy < grid.shape[1]; ++iy) {
          double dy = iy * grid.dx - center[1];
          if (grid.periodic[1])
            dy = Grid::minimum_image(dy, grid.box_size[1]);
          for (long iz = 0; iz < grid.shape[2]; ++iz) {
            double dz = iz * grid.dx - center[2];
            if (grid.periodic[2])
              dz = Grid::minimum_image(dz, grid.box_size[2]);
            double ell = std::sqrt(
                (dx * dx) / (radii[0] * radii[0]) + (dy * dy) / (radii[1] * radii[1])
                + (dz * dz) / (radii[2] * radii[2])
            );
            double signed_distance = (ell - 1.0) * r_min;
            envelope(static_cast<arma::uword>(grid.flat_index(ix, iy, iz))) = 0.5
                * (1.0 - std::tanh(signed_distance / width));
          }
        }
      }

      return envelope;
    }
  };

  // Rescale a field so its total integral equals a target mass.

  [[nodiscard]] inline auto
  rescale_mass(const arma::vec& field, double target_mass, double cell_volume, double floor = 1e-18) -> arma::vec {
    double mass = arma::accu(field) * cell_volume;
    if (mass <= 0.0)
      return field;
    return arma::clamp(field * (target_mass / mass), floor, arma::datum::inf);
  }

  // Rescale only the excess (field - background) to match a target excess mass.

  [[nodiscard]] inline auto rescale_excess_mass(
      const arma::vec& field,
      const arma::vec& background,
      double target_excess_mass,
      double cell_volume,
      double floor = 1e-18
  ) -> arma::vec {
    arma::vec excess = arma::clamp(field - background, 0.0, arma::datum::inf);
    double excess_mass = arma::accu(excess) * cell_volume;
    if (excess_mass <= 0.0) {
      return arma::clamp(background, floor, arma::datum::inf);
    }
    return arma::clamp(background + excess * (target_excess_mass / excess_mass), floor, arma::datum::inf);
  }

  // Effective spherical radius from excess particle number:
  // R = (3 Delta_N / (4 pi delta_rho))^(1/3)

  [[nodiscard]] inline auto
  effective_radius(const arma::vec& field, double background, double delta_rho, double cell_volume) -> double {
    double delta_N = (arma::accu(field) - static_cast<double>(field.n_elem) * background) * cell_volume;
    if (delta_N <= 0.0)
      return 0.0;
    return std::cbrt(3.0 * delta_N / (4.0 * std::numbers::pi * delta_rho));
  }

  // Moment-based cluster radius (Lutsko, Sci. Adv. 2019, SI Eq. 51):
  // R^2 = (5/3) * int(Delta_n * r^2 dV) / int(Delta_n dV)
  // Reduces to the sphere radius for a uniform step profile.
  // Requires the radial distance field from grid.radial_distances().

  [[nodiscard]] inline auto
  moment_radius(const arma::vec& field, double background, const arma::vec& radial_distances, double cell_volume)
      -> double {
    arma::vec delta_n = field - background;
    double m0 = arma::accu(delta_n) * cell_volume;
    if (m0 <= 0.0)
      return 0.0;
    arma::vec r2 = radial_distances % radial_distances;
    double m2 = arma::accu(delta_n % r2) * cell_volume;
    double R2 = (5.0 / 3.0) * m2 / m0;
    return (R2 > 0.0) ? std::sqrt(R2) : 0.0;
  }

  // Crystallinity diagnostic: count density peaks (local maxima in 3D)
  // above a threshold, and compute the contrast ratio rho_max/rho_mean
  // within the cluster envelope. A liquid droplet has ~0 peaks and
  // contrast ~1; an FCC crystal with N unit cells has ~4N³ peaks and
  // contrast >> 1.

  struct CrystallinityMetrics {
    int n_peaks{0};
    double contrast{0.0};
    double rho_max{0.0};
    double rho_mean_cluster{0.0};
  };

  [[nodiscard]] inline auto
  crystallinity(const arma::vec& field, const Grid& grid, double background, double threshold_factor = 2.0)
      -> CrystallinityMetrics {
    double threshold = threshold_factor * background;
    long nx = grid.shape[0];
    long ny = grid.shape[1];
    long nz = grid.shape[2];

    int peaks = 0;
    double rho_max = 0.0;
    double sum_cluster = 0.0;
    int n_cluster = 0;

    for (long ix = 0; ix < nx; ++ix) {
      for (long iy = 0; iy < ny; ++iy) {
        for (long iz = 0; iz < nz; ++iz) {
          auto idx = static_cast<arma::uword>(grid.flat_index(ix, iy, iz));
          double val = field(idx);
          if (val > threshold) {
            sum_cluster += val;
            n_cluster++;
            if (val > rho_max)
              rho_max = val;

            bool is_max = true;
            for (int dx = -1; dx <= 1 && is_max; ++dx) {
              for (int dy = -1; dy <= 1 && is_max; ++dy) {
                for (int dz = -1; dz <= 1 && is_max; ++dz) {
                  if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                  long jx = (ix + dx + nx) % nx;
                  long jy = (iy + dy + ny) % ny;
                  long jz = (iz + dz + nz) % nz;
                  auto jdx = static_cast<arma::uword>(grid.flat_index(jx, jy, jz));
                  if (field(jdx) > val)
                    is_max = false;
                }
              }
            }
            if (is_max)
              peaks++;
          }
        }
      }
    }

    double mean = (n_cluster > 0) ? sum_cluster / static_cast<double>(n_cluster) : 0.0;
    return {
        .n_peaks = peaks,
        .contrast = (mean > 0.0) ? rho_max / mean : 0.0,
        .rho_max = rho_max,
        .rho_mean_cluster = mean,
    };
  }

  // Average density inside a sphere of given radius.

  [[nodiscard]] inline auto
  cluster_average_density(const arma::vec& field, const arma::vec& radial_distances, double radius) -> double {
    if (radius <= 0.0)
      return 0.0;
    arma::uvec inside = arma::find(radial_distances < radius);
    if (inside.is_empty())
      return 0.0;
    return arma::mean(field.elem(inside));
  }

} // namespace dft

#endif // DFT_FIELDS_HPP
