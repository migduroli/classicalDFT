#ifndef DFT_FUNCTIONALS_MEAN_FIELD_HPP
#define DFT_FUNCTIONALS_MEAN_FIELD_HPP

#include "dft/functionals/ideal_gas.hpp"
#include "dft/grid.hpp"
#include "dft/math/convolution.hpp"
#include "dft/math/fourier.hpp"
#include "dft/physics/interactions.hpp"
#include "dft/physics/potentials.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <complex>
#include <numbers>
#include <span>
#include <vector>

namespace dft::functionals {

  // Precomputed Fourier-space weight for a single pair interaction.
  // Immutable after construction.

  struct InteractionWeight {
    int species_i{0};
    int species_j{0};
    math::FourierTransform weight;
    double a_vdw{0.0};
  };

  // Precomputed weights for all pair interactions.

  struct MeanFieldWeights {
    std::vector<InteractionWeight> interactions;
  };

  namespace detail {

    // Compute the cell weight at a single grid displacement using the
    // attractive part of the potential divided by kT.

    [[nodiscard]] inline auto cell_weight_zero(
        const physics::potentials::Potential& pot, physics::potentials::SplitScheme split,
        double kT, double dx, long sx, long sy, long sz
    ) -> double {
      double r2 = static_cast<double>(sx * sx + sy * sy + sz * sz) * dx * dx;
      double r = std::sqrt(r2);
      return physics::potentials::attractive(pot, r, split) / kT;
    }

    // Linear interpolation: average over the 8 corners of the cubic cell.

    [[nodiscard]] inline auto cell_weight_linear(
        const physics::potentials::Potential& pot, physics::potentials::SplitScheme split,
        double kT, double dx, long sx, long sy, long sz
    ) -> double {
      double sum = 0.0;
      for (int di = 0; di <= 1; ++di) {
        for (int dj = 0; dj <= 1; ++dj) {
          for (int dk = 0; dk <= 1; ++dk) {
            double x = (static_cast<double>(sx) - 0.5 + static_cast<double>(di)) * dx;
            double y = (static_cast<double>(sy) - 0.5 + static_cast<double>(dj)) * dx;
            double z = (static_cast<double>(sz) - 0.5 + static_cast<double>(dk)) * dx;
            double r = std::sqrt(x * x + y * y + z * z);
            sum += physics::potentials::attractive(pot, r, split) / kT;
          }
        }
      }
      return sum / 8.0;
    }

    // Quadratic force-route (QF): 3 points per axis at offsets [-0.5, 0, +0.5]
    // with equal weights 1/3 (27-point rule).  Matches Lutsko's
    // Interaction_Interpolation_QF exactly.

    [[nodiscard]] inline auto cell_weight_quadratic_f(
        const physics::potentials::Potential& pot, physics::potentials::SplitScheme split,
        double kT, double dx, long sx, long sy, long sz
    ) -> double {
      static constexpr double vv[] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
      static constexpr double pt[] = {-0.5, 0.0, 0.5};

      double sum = 0.0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 3; ++k) {
            double rx = (static_cast<double>(sx) + pt[i]) * dx;
            double ry = (static_cast<double>(sy) + pt[j]) * dx;
            double rz = (static_cast<double>(sz) + pt[k]) * dx;
            double r = std::sqrt(rx * rx + ry * ry + rz * rz);
            sum += vv[i] * vv[j] * vv[k] * physics::potentials::attractive(pot, r, split) / kT;
          }
        }
      }
      return sum;
    }

    // Compute the cell weight for a given displacement using the specified scheme.

    [[nodiscard]] inline auto cell_weight(
        const physics::Interaction& inter, double kT, double dx, long sx, long sy, long sz
    ) -> double {
      using physics::WeightScheme;
      switch (inter.weight_scheme) {
        case WeightScheme::InterpolationZero:
          return cell_weight_zero(inter.potential, inter.split, kT, dx, sx, sy, sz);
        case WeightScheme::InterpolationLinearE:
        case WeightScheme::InterpolationLinearF:
          return cell_weight_linear(inter.potential, inter.split, kT, dx, sx, sy, sz);
        case WeightScheme::InterpolationQuadraticF:
          return cell_weight_quadratic_f(inter.potential, inter.split, kT, dx, sx, sy, sz);
        case WeightScheme::GaussE:
        case WeightScheme::GaussF:
          return cell_weight_linear(inter.potential, inter.split, kT, dx, sx, sy, sz);
      }
      return 0.0;
    }

    // Generate the real-space weight array for a single interaction
    // and FFT it. Returns the FourierTransform with the weight in k-space
    // and the van der Waals parameter a_vdw = sum(w * dV).

    struct WeightResult {
      math::FourierTransform ft;
      double a_vdw;
    };

    [[nodiscard]] inline auto generate_interaction_weight(
        const physics::Interaction& inter, const Grid& grid, double kT
    ) -> WeightResult {
      std::vector<long> shape(grid.shape.begin(), grid.shape.end());
      long nx = shape[0];
      long ny = shape[1];
      long nz = shape[2];
      double dx = grid.dx;
      double dv = grid.cell_volume();
      double inv_n = 1.0 / static_cast<double>(grid.total_points());

      double r_cutoff = std::visit(
          [](const auto& p) -> double { return p.r_cutoff; }, inter.potential
      );
      // Expand the early-skip radius to account for sub-cell quadrature
      // offsets.  The farthest sub-cell point from the cell center lies
      // at offset sqrt(3) * 0.5 * dx (half the cell diagonal).  Any cell
      // whose center is within r_cutoff + margin may have a sub-cell
      // point inside r_cutoff.
      double margin = std::sqrt(3.0) * 0.5 * dx;
      double r_skip = r_cutoff + margin;
      double r_skip_sq = r_skip * r_skip;

      math::FourierTransform ft(shape);
      auto real = ft.real();
      std::fill(real.begin(), real.end(), 0.0);

      double weight_sum = 0.0;

      for (long ix = 0; ix < nx; ++ix) {
        long sx = (ix <= nx / 2) ? ix : ix - nx;
        for (long iy = 0; iy < ny; ++iy) {
          long sy = (iy <= ny / 2) ? iy : iy - ny;
          for (long iz = 0; iz < nz; ++iz) {
            long sz = (iz <= nz / 2) ? iz : iz - nz;

            double dist2 = static_cast<double>(sx * sx + sy * sy + sz * sz) * dx * dx;
            if (r_cutoff > 0.0 && dist2 > r_skip_sq) {
              continue;
            }

            double w = cell_weight(inter, kT, dx, sx, sy, sz);

            auto idx = static_cast<std::size_t>(iz + nz * (iy + ny * ix));
            // Bake in dV/N so that convolve() (unnormalized IFFT) gives
            // the correctly scaled discrete convolution.
            real[idx] = w * dv * inv_n;
            weight_sum += w * dv;
          }
        }
      }

      ft.forward();
      return {.ft = std::move(ft), .a_vdw = weight_sum};
    }

  }  // namespace detail

  // Generate the Fourier-space weights for all interactions.

  [[nodiscard]] inline auto make_mean_field_weights(
      const Grid& grid, const std::vector<physics::Interaction>& interactions, double kT
  ) -> MeanFieldWeights {
    MeanFieldWeights w;
    w.interactions.reserve(interactions.size());

    for (const auto& inter : interactions) {
      auto [ft, a_vdw] = detail::generate_interaction_weight(inter, grid, kT);
      w.interactions.push_back(InteractionWeight{
          .species_i = inter.species_i,
          .species_j = inter.species_j,
          .weight = std::move(ft),
          .a_vdw = a_vdw,
      });
    }

    return w;
  }

  // Evaluate the mean-field functional for all interactions.
  //
  // Free energy:
  //   F_mf = (1/2) sum_{pairs} sum_r rho_i(r) [w * rho_j](r) dV
  //
  // Functional derivative (force per grid point):
  //   dF/d rho_i(r) = sum_{j in partners} [w_ij * rho_j](r) dV
  //
  // For self-interactions (i == j), the factor 1/2 cancels with the
  // double counting in the derivative. For cross-interactions (i != j),
  // each species gets half the convolution.

  [[nodiscard]] inline auto mean_field(
      const Grid& grid, const State& state,
      const std::vector<Species>& species,
      const MeanFieldWeights& weights
  ) -> Contribution {
    auto n_species = species.size();
    auto n_points = static_cast<arma::uword>(grid.total_points());
    double dv = grid.cell_volume();
    std::vector<long> shape(grid.shape.begin(), grid.shape.end());

    // FFT all density profiles
    std::vector<math::FourierTransform> rho_ft;
    rho_ft.reserve(n_species);
    for (std::size_t s = 0; s < n_species; ++s) {
      rho_ft.emplace_back(shape);
      rho_ft.back().set_real(state.species[s].density.values);
      rho_ft.back().forward();
    }

    // Accumulate per-species forces
    std::vector<arma::vec> forces(n_species, arma::zeros(n_points));
    double free_energy = 0.0;

    for (const auto& iw : weights.interactions) {
      auto i = static_cast<std::size_t>(iw.species_i);
      auto j = static_cast<std::size_t>(iw.species_j);

      // Convolve weight with rho_j to get [w * rho_j](r)
      arma::vec conv_j = math::convolve(iw.weight.fourier(), rho_ft[j].fourier(), shape);

      if (i == j) {
        // Self-interaction: F = (1/2) sum rho_i * conv_j * dV
        free_energy += 0.5 * arma::dot(state.species[i].density.values, conv_j) * dv;

        // dF/d rho_i = conv_j * dV (the 1/2 cancels with double counting)
        forces[i] += conv_j * dv;
      } else {
        // Cross-interaction: F = (1/2) sum rho_i * conv_j * dV
        free_energy += 0.5 * arma::dot(state.species[i].density.values, conv_j) * dv;

        // dF/d rho_i = (1/2) [w * rho_j] * dV
        forces[i] += 0.5 * conv_j * dv;

        // dF/d rho_j = (1/2) [w * rho_i] * dV
        arma::vec conv_i = math::convolve(iw.weight.fourier(), rho_ft[i].fourier(), shape);
        forces[j] += 0.5 * conv_i * dv;
      }
    }

    return Contribution{.free_energy = free_energy, .forces = std::move(forces)};
  }

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_MEAN_FIELD_HPP
