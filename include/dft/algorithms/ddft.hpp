#ifndef DFT_ALGORITHMS_DDFT_HPP
#define DFT_ALGORITHMS_DDFT_HPP

#include "dft/grid.hpp"
#include "dft/math/fourier.hpp"

#include <armadillo>
#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

namespace dft::algorithms::ddft {

  struct DdftConfig {
    double dt{1e-4};
    double diffusion_coefficient{1.0};
    double min_density{1e-18};
    int cn_max_iterations{5};
    double cn_tolerance{1e-10};
  };

  // Precomputed k^2 values for the half-complex FFT grid.
  // Build once, reuse for all timesteps.

  [[nodiscard]] inline auto compute_k_squared(const Grid& grid) -> arma::vec {
    long nz_half = grid.shape[2] / 2 + 1;
    long fourier_total = grid.shape[0] * grid.shape[1] * nz_half;
    arma::vec k2(static_cast<arma::uword>(fourier_total));

    for_each_wavevector(grid, [&](const Wavevector& wv) {
      k2(static_cast<arma::uword>(wv.idx)) = wv.norm2();
    });

    return k2;
  }

  // Diffusion propagator: exp(-D * k^2 * dt) for each Fourier mode.

  [[nodiscard]] inline auto diffusion_propagator(
      const arma::vec& k_squared, double D, double dt
  ) -> arma::vec {
    return arma::exp(-D * k_squared * dt);
  }

  // The excess force callback computes functional derivatives at
  // the current density, returning the per-species forces
  // (delta_F / delta_rho - mu, in real space).

  using ForceCallback = std::function<std::pair<double, std::vector<arma::vec>>(
      const std::vector<arma::vec>& densities
  )>;

  // Result of a single DDFT timestep.

  struct DdftResult {
    std::vector<arma::vec> densities;
    double energy;
  };

  // Split-operator DDFT step (first-order, explicit for excess).
  //
  // 1. Apply ideal diffusion propagator in Fourier space
  // 2. Compute excess nonlinear term and apply via forward Euler
  //
  // The DDFT equation is:
  //   d rho / dt = D * div(rho * grad(delta F / delta rho))
  //             = D * laplacian(rho) + D * div(rho * grad(c_ex))
  //
  // The first term is solved exactly in Fourier space.
  // The second is treated explicitly.

  [[nodiscard]] inline auto split_operator_step(
      const std::vector<arma::vec>& densities, const Grid& grid,
      const arma::vec& k_squared, const arma::vec& propagator,
      const ForceCallback& compute, const DdftConfig& config
  ) -> DdftResult {
    auto n_species = densities.size();
    auto shape_vec = std::vector<long>(grid.shape.begin(), grid.shape.end());
    double n_total = static_cast<double>(grid.total_points());

    // Step 1: Apply ideal diffusion propagator.
    std::vector<arma::vec> rho_new(n_species);
    for (std::size_t s = 0; s < n_species; ++s) {
      math::FourierTransform ft(shape_vec);
      ft.set_real(densities[s]);
      ft.forward();
      arma::cx_vec rho_k = ft.fourier_vec();
      arma::vec prop_real = propagator;
      arma::cx_vec prop_cx(prop_real, arma::zeros(prop_real.n_elem));
      rho_k %= prop_cx;
      ft.set_fourier(rho_k);
      ft.backward();
      rho_new[s] = ft.real_vec() / n_total;
    }

    // Step 2: Compute excess forces at the propagated density.
    auto [energy, forces] = compute(rho_new);

    // Step 3: Excess contribution via forward Euler.
    // The excess force for species s is: f_ex = (total_force / dV) - ln(rho)
    // The flux is: rho * f_ex
    // Divergence in Fourier space: -k^2 * FFT(flux)
    double D = config.diffusion_coefficient;
    double dv = grid.cell_volume();

    for (std::size_t s = 0; s < n_species; ++s) {
      arma::vec f_ex = forces[s] / dv - arma::log(arma::clamp(rho_new[s], config.min_density, arma::datum::inf));
      arma::vec flux = rho_new[s] % f_ex;

      math::FourierTransform ft(shape_vec);
      ft.set_real(flux);
      ft.forward();
      arma::cx_vec flux_k = ft.fourier_vec();

      // -k^2 in Fourier space gives the Laplacian of the flux
      arma::cx_vec k2_cx(k_squared, arma::zeros(k_squared.n_elem));
      flux_k %= -k2_cx;

      ft.set_fourier(flux_k);
      ft.backward();
      arma::vec div_flux = ft.real_vec() / n_total;

      rho_new[s] += config.dt * D * div_flux;
      rho_new[s] = arma::clamp(rho_new[s], config.min_density, arma::datum::inf);
    }

    // Final energy at new density.
    auto [e_new, _] = compute(rho_new);

    return DdftResult{.densities = std::move(rho_new), .energy = e_new};
  }

  // Crank-Nicholson DDFT step (second-order, implicit for excess).
  //
  // Uses the integrating-factor approach:
  //   rho_prop = exp(-D k^2 dt) * rho^n  (ideal propagation)
  // Then fixed-point iteration for the nonlinear term:
  //   rho^{n+1}_k = rho_prop + (dt/2) * (N^n + exp(-D k^2 dt/2) * N^{n+1})
  // where N = -D k^2 * FFT(rho * c_ex).

  [[nodiscard]] inline auto crank_nicholson_step(
      const std::vector<arma::vec>& densities, const Grid& grid,
      const arma::vec& k_squared, const ForceCallback& compute,
      const DdftConfig& config
  ) -> DdftResult {
    auto n_species = densities.size();
    auto shape_vec = std::vector<long>(grid.shape.begin(), grid.shape.end());
    double n_total = static_cast<double>(grid.total_points());
    double D = config.diffusion_coefficient;
    double dv = grid.cell_volume();

    arma::vec prop_full = diffusion_propagator(k_squared, D, config.dt);
    arma::vec prop_half = diffusion_propagator(k_squared, D, config.dt / 2.0);

    // Compute N^n (nonlinear term at current density).
    auto nonlinear_term = [&](const std::vector<arma::vec>& rho,
                              const std::vector<arma::vec>& forces) -> std::vector<arma::cx_vec> {
      std::vector<arma::cx_vec> N(n_species);
      arma::cx_vec k2_cx(k_squared, arma::zeros(k_squared.n_elem));
      for (std::size_t s = 0; s < n_species; ++s) {
        arma::vec f_ex = forces[s] / dv - arma::log(arma::clamp(rho[s], config.min_density, arma::datum::inf));
        arma::vec flux = rho[s] % f_ex;

        math::FourierTransform ft(shape_vec);
        ft.set_real(flux);
        ft.forward();
        N[s] = -D * k2_cx % ft.fourier_vec();
      }
      return N;
    };

    auto [energy_n, forces_n] = compute(densities);
    auto N_n = nonlinear_term(densities, forces_n);

    // Propagated density in Fourier space.
    std::vector<arma::cx_vec> rho_prop_k(n_species);
    for (std::size_t s = 0; s < n_species; ++s) {
      math::FourierTransform ft(shape_vec);
      ft.set_real(densities[s]);
      ft.forward();
      arma::cx_vec prop_cx(prop_full, arma::zeros(prop_full.n_elem));
      rho_prop_k[s] = prop_cx % ft.fourier_vec();
    }

    // Fixed-point iteration.
    std::vector<arma::cx_vec> rho_k(n_species);
    for (std::size_t s = 0; s < n_species; ++s) {
      arma::cx_vec prop_half_cx(prop_half, arma::zeros(prop_half.n_elem));
      rho_k[s] = rho_prop_k[s] + (config.dt / 2.0) * N_n[s];
    }

    for (int iter = 0; iter < config.cn_max_iterations; ++iter) {
      // Convert current guess to real space.
      std::vector<arma::vec> rho_guess(n_species);
      for (std::size_t s = 0; s < n_species; ++s) {
        math::FourierTransform ft(shape_vec);
        ft.set_fourier(rho_k[s]);
        ft.backward();
        rho_guess[s] = arma::clamp(ft.real_vec() / n_total, config.min_density, arma::datum::inf);
      }

      auto [e_guess, f_guess] = compute(rho_guess);
      auto N_new = nonlinear_term(rho_guess, f_guess);

      // Update: rho_k = rho_prop + (dt/2) * (N^n + prop_half * N^{n+1})
      std::vector<arma::cx_vec> rho_k_new(n_species);
      double max_change = 0.0;
      arma::cx_vec prop_half_cx(prop_half, arma::zeros(prop_half.n_elem));

      for (std::size_t s = 0; s < n_species; ++s) {
        rho_k_new[s] = rho_prop_k[s] + (config.dt / 2.0) * (N_n[s] + prop_half_cx % N_new[s]);
        double change = arma::norm(rho_k_new[s] - rho_k[s]);
        double scale = std::max(arma::norm(rho_k[s]), 1e-30);
        max_change = std::max(max_change, change / scale);
      }

      rho_k = std::move(rho_k_new);

      if (max_change < config.cn_tolerance) {
        break;
      }
    }

    // Final inverse FFT.
    std::vector<arma::vec> rho_new(n_species);
    for (std::size_t s = 0; s < n_species; ++s) {
      math::FourierTransform ft(shape_vec);
      ft.set_fourier(rho_k[s]);
      ft.backward();
      rho_new[s] = arma::clamp(ft.real_vec() / n_total, config.min_density, arma::datum::inf);
    }

    auto [e_final, _] = compute(rho_new);

    return DdftResult{.densities = std::move(rho_new), .energy = e_final};
  }

}  // namespace dft::algorithms::ddft

#endif  // DFT_ALGORITHMS_DDFT_HPP
