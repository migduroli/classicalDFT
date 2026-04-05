#ifndef DFT_FUNCTIONALS_HARD_SPHERE_HPP
#define DFT_FUNCTIONALS_HARD_SPHERE_HPP

#include "dft/functionals/fmt/measures.hpp"
#include "dft/functionals/fmt/models.hpp"
#include "dft/functionals/fmt/weights.hpp"
#include "dft/functionals/types.hpp"
#include "dft/grid.hpp"
#include "dft/math/convolution.hpp"
#include "dft/math/fourier.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <numbers>
#include <vector>

namespace dft::functionals {

  // Pre-computed FMT weight functions for all species.
  // Immutable after construction — pass by const ref.

  struct FMTWeights {
    std::vector<fmt::WeightSet> per_species;
  };

  // Generate the Fourier-space weight functions for each species
  // on the given grid.

  [[nodiscard]] inline auto make_fmt_weights(
      const Grid& grid, const std::vector<Species>& species
  ) -> FMTWeights {
    FMTWeights w;
    w.per_species.reserve(species.size());
    for (const auto& sp : species) {
      w.per_species.push_back(fmt::generate_weights(sp.hard_sphere_diameter, grid));
    }
    return w;
  }

  // Convolve a density (already in Fourier space) with all weight channels
  // to produce the weighted density arrays for one species.

  namespace detail {

    struct WeightedDensityArrays {
      arma::vec eta;
      arma::vec n2;
      std::array<arma::vec, 3> nv2;
      std::array<std::array<arma::vec, 3>, 3> nT;
    };

    [[nodiscard]] inline auto convolve_weights(
        const fmt::WeightSet& ws, std::span<const std::complex<double>> rho_k,
        const std::vector<long>& shape
    ) -> WeightedDensityArrays {
      WeightedDensityArrays wd;
      wd.eta = math::convolve(ws.w3.fourier(), rho_k, shape);
      wd.n2 = math::convolve(ws.w2.fourier(), rho_k, shape);
      for (int a = 0; a < 3; ++a) {
        wd.nv2[a] = math::convolve(ws.wv2[a].fourier(), rho_k, shape);
      }
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          wd.nT[i][j] = math::convolve(ws.tensor(i, j).fourier(), rho_k, shape);
        }
      }
      return wd;
    }

    // Assemble a Measures struct at a single grid point from the
    // weighted density arrays. n1 and n0 are recovered from n2 via
    // Rosenfeld scaling: n1 = n2/(4piR), n0 = n2/(4piR^2).

    [[nodiscard]] inline auto assemble_measures(
        const WeightedDensityArrays& wd, arma::uword idx, double R
    ) -> fmt::Measures {
      double four_pi_R = 4.0 * std::numbers::pi * R;

      fmt::Measures m;
      m.eta = wd.eta(idx);
      m.n2 = wd.n2(idx);
      m.n1 = wd.n2(idx) / four_pi_R;
      m.n0 = wd.n2(idx) / (four_pi_R * R);

      for (int a = 0; a < 3; ++a) {
        m.v1(a) = wd.nv2[a](idx);
        m.v0(a) = wd.nv2[a](idx) / four_pi_R;
      }

      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          m.T(i, j) = wd.nT[i][j](idx);
          m.T(j, i) = m.T(i, j);
        }
      }

      m.products = m.inner_products();
      return m;
    }

    // Back-convolution: compute the Fourier-space force contribution
    // from all weight channels for one species.  Returns the sum of
    // all back-convolved channels as a complex vector.

    [[nodiscard]] inline auto compute_force_k(
        const fmt::WeightSet& ws, double R, const arma::vec& d_eta, const arma::vec& d_n2,
        const arma::vec& d_n1, const arma::vec& d_n0, const std::array<arma::vec, 3>& d_nv2,
        const std::array<arma::vec, 3>& d_nv0,
        const std::array<std::array<arma::vec, 3>, 3>& d_nT,
        const std::vector<long>& shape
    ) -> arma::cx_vec {
      double four_pi_R = 4.0 * std::numbers::pi * R;

      // w3 channel: dPhi/deta
      arma::cx_vec force_k = math::back_convolve(ws.w3.fourier(), d_eta, shape);

      // w2 channel: dPhi/dn2 + dPhi/dn1 / (4piR) + dPhi/dn0 / (4piR^2)
      force_k += math::back_convolve(ws.w2.fourier(), d_n2 + d_n1 / four_pi_R + d_n0 / (four_pi_R * R), shape);

      // wv2 channels (parity-odd, conjugate = true)
      for (int a = 0; a < 3; ++a) {
        force_k += math::back_convolve(ws.wv2[a].fourier(), d_nv2[a] + d_nv0[a] / four_pi_R, shape, true);
      }

      // wT channels
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          double sym = (i == j) ? 1.0 : 2.0;
          force_k += math::back_convolve(ws.tensor(i, j).fourier(), sym * d_nT[i][j], shape);
        }
      }

      return force_k;
    }

  }  // namespace detail

  namespace _internal {

    struct DerivativeArrays {
      arma::vec d_eta;
      arma::vec d_n2;
      arma::vec d_n1;
      arma::vec d_n0;
      std::array<arma::vec, 3> d_nv2;
      std::array<arma::vec, 3> d_nv0;
      std::array<std::array<arma::vec, 3>, 3> d_nT;
    };

    [[nodiscard]] inline auto make_derivative_arrays(
        std::size_t n_species, arma::uword n_points
    ) -> std::vector<DerivativeArrays> {
      std::vector<DerivativeArrays> derivs(n_species);
      for (auto& da : derivs) {
        da.d_eta = arma::zeros(n_points);
        da.d_n2 = arma::zeros(n_points);
        da.d_n1 = arma::zeros(n_points);
        da.d_n0 = arma::zeros(n_points);
        for (int a = 0; a < 3; ++a) {
          da.d_nv2[a] = arma::zeros(n_points);
          da.d_nv0[a] = arma::zeros(n_points);
        }
        for (int i = 0; i < 3; ++i) {
          for (int j = i; j < 3; ++j) {
            da.d_nT[i][j] = arma::zeros(n_points);
          }
        }
      }
      return derivs;
    }

    [[nodiscard]] inline auto accumulate_derivatives(
        const fmt::FMTModel& model,
        const std::vector<detail::WeightedDensityArrays>& wd,
        const std::vector<Species>& species,
        arma::uword n_points, double dv
    ) -> std::pair<double, std::vector<DerivativeArrays>> {
      auto n_species = species.size();
      auto derivs = make_derivative_arrays(n_species, n_points);
      double free_energy = 0.0;

      for (arma::uword idx = 0; idx < n_points; ++idx) {
        fmt::Measures total;
        for (std::size_t s = 0; s < n_species; ++s) {
          double R = 0.5 * species[s].hard_sphere_diameter;
          auto m = detail::assemble_measures(wd[s], idx, R);
          total.eta += m.eta;
          total.n0 += m.n0;
          total.n1 += m.n1;
          total.n2 += m.n2;
          total.v0 += m.v0;
          total.v1 += m.v1;
          total.T += m.T;
        }

        if (total.eta < 1e-30) continue;

        total.products = total.inner_products();
        free_energy += model.phi(total) * dv;

        auto dm = model.d_phi(total);
        for (std::size_t s = 0; s < n_species; ++s) {
          derivs[s].d_eta(idx) = dm.d_eta;
          derivs[s].d_n2(idx) = dm.d_n2;
          derivs[s].d_n1(idx) = dm.d_n1;
          derivs[s].d_n0(idx) = dm.d_n0;
          for (int a = 0; a < 3; ++a) {
            derivs[s].d_nv2[a](idx) = dm.d_v1(a);
            derivs[s].d_nv0[a](idx) = dm.d_v0(a);
          }
          for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) {
              derivs[s].d_nT[i][j](idx) = dm.d_T(i, j);
            }
          }
        }
      }

      return {free_energy, std::move(derivs)};
    }

    [[nodiscard]] inline auto back_convolve_forces(
        const std::vector<DerivativeArrays>& derivs,
        const FMTWeights& weights,
        const std::vector<Species>& species,
        const std::vector<long>& shape, double dv
    ) -> std::vector<arma::vec> {
      std::vector<arma::vec> forces;
      forces.reserve(species.size());

      for (std::size_t s = 0; s < species.size(); ++s) {
        double R = 0.5 * species[s].hard_sphere_diameter;
        auto force_k = detail::compute_force_k(
            weights.per_species[s], R,
            derivs[s].d_eta, derivs[s].d_n2, derivs[s].d_n1, derivs[s].d_n0,
            derivs[s].d_nv2, derivs[s].d_nv0, derivs[s].d_nT, shape
        );
        math::FourierTransform force_ft(shape);
        force_ft.set_fourier(force_k);
        force_ft.backward();
        forces.push_back(force_ft.real_vec() * dv);
      }

      return forces;
    }

  }  // namespace _internal

  // Evaluate the FMT hard-sphere functional for all species.

  [[nodiscard]] inline auto hard_sphere(
      const fmt::FMTModel& model, const Grid& grid, const State& state,
      const std::vector<Species>& species, const FMTWeights& weights
  ) -> Contribution {
    auto n_species = species.size();
    auto n_points = static_cast<arma::uword>(grid.total_points());
    double dv = grid.cell_volume();
    std::vector<long> shape(grid.shape.begin(), grid.shape.end());

    // FFT all density profiles.
    std::vector<math::FourierTransform> rho_ft;
    rho_ft.reserve(n_species);
    for (std::size_t s = 0; s < n_species; ++s) {
      rho_ft.emplace_back(shape);
      rho_ft.back().set_real(state.species[s].density.values);
      rho_ft.back().forward();
    }

    // Convolve each species density with its weights.
    std::vector<detail::WeightedDensityArrays> wd;
    wd.reserve(n_species);
    for (std::size_t s = 0; s < n_species; ++s) {
      wd.push_back(detail::convolve_weights(weights.per_species[s], rho_ft[s].fourier(), shape));
    }

    // Accumulate energy and derivatives at each grid point.
    auto [free_energy, derivs] = _internal::accumulate_derivatives(model, wd, species, n_points, dv);

    // Back-convolve derivatives with weights to get forces.
    auto forces = _internal::back_convolve_forces(derivs, weights, species, shape, dv);

    return Contribution{.free_energy = free_energy, .forces = std::move(forces)};
  }

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_HARD_SPHERE_HPP
