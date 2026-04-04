#pragma once

#include "dft/algorithms/simulate.hpp"
#include "dft/grid.hpp"
#include "dft/math/spline.hpp"

#include <armadillo>
#include <cmath>
#include <numbers>
#include <vector>

namespace nucleation {

  struct RadialSnapshot {
    double time{0.0};
    std::vector<double> r;
    std::vector<double> rho;
  };

  struct SliceSnapshot {
    double time{0.0};
    std::vector<double> x;
    std::vector<double> rho;
  };

  struct PathwayPoint {
    double radius;
    double energy;
  };

  struct DynamicsResult {
    std::vector<SliceSnapshot> profiles;
    std::vector<PathwayPoint> pathway;
  };

  [[nodiscard]] inline auto radial_distances(const dft::Grid& grid) -> arma::vec {
    auto n = static_cast<arma::uword>(grid.total_points());
    arma::vec r2(n, arma::fill::zeros);

    for (int d = 0; d < 3; ++d) {
      double centre = grid.box_size[d] / 2.0;
      long nd = grid.shape[d];
      arma::vec coord(static_cast<arma::uword>(nd));
      for (long i = 0; i < nd; ++i) {
        coord(static_cast<arma::uword>(i)) = i * grid.dx - centre;
      }

      arma::uword stride = 1;
      for (int dd = d + 1; dd < 3; ++dd) {
        stride *= static_cast<arma::uword>(grid.shape[dd]);
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

  // Step function: rho_in inside radius R, rho_out outside.

  [[nodiscard]] inline auto step_function(
      const arma::vec& r, double R, double rho_in, double rho_out
  ) -> arma::vec {
    arma::vec rho(r.n_elem);
    rho.fill(rho_out);
    rho.elem(arma::find(r < R)).fill(rho_in);
    return rho;
  }

  // Effective droplet radius from excess particle number:
  // R_eff = (3 Delta_N / (4 pi delta_rho))^(1/3)

  [[nodiscard]] inline auto effective_radius(
      const arma::vec& rho, double rho_background, double delta_rho, double dv
  ) -> double {
    double delta_N = (arma::accu(rho) - static_cast<double>(rho.n_elem) * rho_background) * dv;
    if (delta_N <= 0.0) return 0.0;
    return std::cbrt(3.0 * delta_N / (4.0 * std::numbers::pi * delta_rho));
  }

  [[nodiscard]] inline auto extract_radial(
      const arma::vec& rho, const dft::Grid& grid, const arma::vec& r
  ) -> RadialSnapshot {
    double r_max = std::min({grid.box_size[0], grid.box_size[1], grid.box_size[2]}) / 2.0;
    auto n_bins = static_cast<arma::uword>(r_max / grid.dx);

    arma::vec bin_sum(n_bins, arma::fill::zeros);
    arma::uvec bin_count(n_bins, arma::fill::zeros);
    arma::uvec bins = arma::conv_to<arma::uvec>::from(arma::floor(r / grid.dx));

    for (arma::uword i = 0; i < rho.n_elem; ++i) {
      if (bins(i) < n_bins) {
        bin_sum(bins(i)) += rho(i);
        bin_count(bins(i)) += 1;
      }
    }

    std::vector<double> r_out, rho_out;
    for (arma::uword i = 0; i < n_bins; ++i) {
      if (bin_count(i) > 0) {
        r_out.push_back((i + 0.5) * grid.dx);
        rho_out.push_back(bin_sum(i) / static_cast<double>(bin_count(i)));
      }
    }

    return {.time = 0.0, .r = r_out, .rho = rho_out};
  }

  // Extract an x-axis slice through the centre of the box (iy=Ny/2, iz=Nz/2).

  [[nodiscard]] inline auto extract_x_slice(
      const arma::vec& rho, const dft::Grid& grid
  ) -> SliceSnapshot {
    long nx = grid.shape[0];
    long iy = grid.shape[1] / 2;
    long iz = grid.shape[2] / 2;
    double centre = grid.box_size[0] / 2.0;

    std::vector<double> x_out(static_cast<std::size_t>(nx));
    std::vector<double> rho_out(static_cast<std::size_t>(nx));
    for (long ix = 0; ix < nx; ++ix) {
      x_out[static_cast<std::size_t>(ix)] = ix * grid.dx - centre;
      rho_out[static_cast<std::size_t>(ix)] = rho(static_cast<arma::uword>(grid.flat_index(ix, iy, iz)));
    }
    return {.time = 0.0, .x = x_out, .rho = rho_out};
  }

  // Extract x-slice snapshots and (R_eff, Omega) pathway from a DDFT simulation result.

  [[nodiscard]] inline auto extract_dynamics(
      const dft::algorithms::ddft::SimulationResult& sim,
      const dft::Grid& grid, const arma::vec& r,
      double rho_background, double delta_rho
  ) -> DynamicsResult {
    double dv = grid.cell_volume();
    DynamicsResult result;
    for (const auto& snap : sim.snapshots) {
      auto prof = extract_x_slice(snap.densities[0], grid);
      prof.time = snap.time;
      result.profiles.push_back(std::move(prof));
      result.pathway.push_back({
          .radius = effective_radius(snap.densities[0], rho_background, delta_rho, dv),
          .energy = snap.energy,
      });
    }
    return result;
  }

  // Perturb the critical cluster by rescaling both radius and amplitude.
  // Works additively on the original 3D field to preserve far-field density:
  //   rho_new(i) = rho_orig(i) + [perturbed_profile(r_i) - original_profile(r_i)]
  // where:
  //   perturbed_profile(r) = rho_bg + amp * (spline(r / rad_factor) - rho_bg)
  //
  // rad_factor < 1 shrinks the cluster, > 1 grows it.
  // amp_factor < 1 dilutes, > 1 densifies.

  [[nodiscard]] inline auto perturb_cluster(
      const arma::vec& rho, const dft::Grid& grid, const arma::vec& r,
      double rad_factor, double amp_factor, double rho_background
  ) -> arma::vec {
    auto profile = extract_radial(rho, grid, r);
    dft::math::CubicSpline spline(profile.r, profile.rho);
    double r_max = profile.r.back();
    double rho_edge = profile.rho.back();

    arma::vec result = rho;
    for (arma::uword i = 0; i < rho.n_elem; ++i) {
      double ri = r(i);
      double orig = (ri <= r_max) ? spline(ri) : rho_edge;
      double ri_s = ri / rad_factor;
      double stretched = (ri_s <= r_max) ? spline(ri_s) : rho_edge;
      double perturbed = rho_background + amp_factor * (stretched - rho_background);
      result(i) += (perturbed - orig);
    }
    return arma::clamp(result, 1e-18, arma::datum::inf);
  }

}  // namespace nucleation
