#include "dft/dynamics/integrator.h"

#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace dft::dynamics {

  Integrator::Integrator(Solver& solver, IntegratorConfig config)
      : Minimizer(solver, config.force_limit, config.min_density), config_(config) {
    if (solver.num_species() < 1) {
      throw std::invalid_argument("Integrator: solver must have at least one species");
    }

    // Precompute k^2
    k_squared_ = compute_k_squared();

    // Initialize working FFT buffer with the same grid shape
    work_fft_ = math::fourier::FourierTransform(solver.density(0).shape());

    // Allocate Crank-Nicholson FFT buffer only when needed
    if (config_.scheme == IntegrationScheme::CrankNicholson) {
      cn_fft_ = math::fourier::FourierTransform(solver.density(0).shape());
    }

    // Compute initial forces (density-space, no alias conversion)
    (void)compute_density_forces();
  }

  void Integrator::do_reset() {
    (void)compute_density_forces();
  }

  double Integrator::do_step() {
    switch (config_.scheme) {
      case IntegrationScheme::SplitOperator:
        return do_split_operator_step();
      case IntegrationScheme::CrankNicholson:
        return do_crank_nicholson_step();
    }
    // All enum values handled; unreachable.
    __builtin_unreachable();
  }

  // ── Split-operator step ─────────────────────────────────────────────────

  double Integrator::do_split_operator_step() {
    double dt = config_.dt;
    double d = config_.diffusion_coefficient;
    int n_species = mutable_solver().num_species();

    for (int s = 0; s < n_species; ++s) {
      auto& density = mutable_solver().species(s).density();
      const arma::vec& force = mutable_solver().species(s).force();
      arma::uword n = density.size();
      double d_v = density.cell_volume();

      // Apply exact diffusion propagator for the ideal part
      apply_diffusion_propagator(s, dt);

      // Excess contribution via forward Euler.
      // The excess force per lattice point (dF_excess/drho), excluding ideal:
      const arma::vec& rho = density.values();
      arma::vec excess_force = force / d_v - arma::log(arma::clamp(rho, 1e-30, arma::datum::inf));

      // Copy excess flux into FFT buffer: rho * excess_force
      auto real_span = work_fft_.real();
      for (arma::uword i = 0; i < n; ++i) {
        real_span[i] = rho(i) * excess_force(i);
      }
      work_fft_.forward();

      // In Fourier space: multiply by -k^2 for the divergence
      auto fourier_span = work_fft_.fourier();
      long n_fourier = work_fft_.fourier_total();
      for (long i = 0; i < n_fourier; ++i) {
        fourier_span[i] *= -k_squared_(static_cast<arma::uword>(i));
      }

      // Transform back
      work_fft_.backward();
      double n_total = static_cast<double>(work_fft_.total());

      // Update density with excess contribution
      arma::vec& rho_mut = density.values();
      auto result_span = work_fft_.real();
      for (arma::uword i = 0; i < n; ++i) {
        rho_mut(i) += dt * d * result_span[i] / n_total;
      }

      // Enforce minimum density
      rho_mut = arma::clamp(rho_mut, species::Species::RHO_MIN, arma::datum::inf);
    }

    // Update aliases from new densities
    for (int s = 0; s < n_species; ++s) {
      aliases()[static_cast<size_t>(s)] = mutable_solver().species(s).density_alias();
    }

    return compute_density_forces();
  }

  // ── Crank-Nicholson step ────────────────────────────────────────────────

  double Integrator::do_crank_nicholson_step() {
    double dt = config_.dt;
    double d = config_.diffusion_coefficient;
    int n_species = mutable_solver().num_species();

    // Compute excess forces at current state
    (void)compute_density_forces();

    for (int s = 0; s < n_species; ++s) {
      auto& density = mutable_solver().species(s).density();
      arma::uword n = density.size();
      long n_fourier = cn_fft_.fourier_total();

      // Step 1: Compute N[rho^n] = nonlinear term at current state
      arma::cx_vec n_current = compute_nonlinear_term(s);

      // Step 2: FFT the current density
      auto real_span = cn_fft_.real();
      const arma::vec& rho = density.values();
      for (arma::uword i = 0; i < n; ++i) {
        real_span[i] = rho(i);
      }
      cn_fft_.forward();

      // Step 3: Apply integrating factor: rho_hat_propagated = exp(-D*k^2*dt) * rho_hat
      auto fourier_span = cn_fft_.fourier();
      arma::cx_vec rho_hat_propagated(static_cast<arma::uword>(n_fourier));
      for (long i = 0; i < n_fourier; ++i) {
        auto ui = static_cast<arma::uword>(i);
        double decay = std::exp(-d * k_squared_(ui) * dt);
        rho_hat_propagated(ui) = decay * fourier_span[i];
      }

      // Step 4: Crank-Nicholson iteration
      arma::cx_vec rho_hat_new = rho_hat_propagated;

      // Precompute half-step propagator
      arma::vec half_propagator(static_cast<arma::uword>(n_fourier));
      for (long i = 0; i < n_fourier; ++i) {
        auto ui = static_cast<arma::uword>(i);
        half_propagator(ui) = std::exp(-d * k_squared_(ui) * 0.5 * dt);
      }

      for (int iter = 0; iter < config_.crank_nicholson_iterations; ++iter) {
        // Convert current guess to real space, update density
        for (long i = 0; i < n_fourier; ++i) {
          fourier_span[i] = rho_hat_new(static_cast<arma::uword>(i));
        }
        cn_fft_.backward();
        double n_total = static_cast<double>(cn_fft_.total());

        arma::vec& rho_mut = density.values();
        for (arma::uword i = 0; i < n; ++i) {
          rho_mut(i) = std::max(real_span[i] / n_total, species::Species::RHO_MIN);
        }

        // Recompute forces at predicted state
        aliases()[static_cast<size_t>(s)] = mutable_solver().species(s).density_alias();
        (void)compute_density_forces();

        // Compute N[rho^{n+1}]
        arma::cx_vec n_new = compute_nonlinear_term(s);

        // Update: rho_hat^{n+1} = rho_hat_propagated + dt/2 * (N^n + half * N^{n+1})
        arma::cx_vec rho_hat_updated(static_cast<arma::uword>(n_fourier));
        for (long i = 0; i < n_fourier; ++i) {
          auto ui = static_cast<arma::uword>(i);
          rho_hat_updated(ui) = rho_hat_propagated(ui) + 0.5 * dt * (n_current(ui) + half_propagator(ui) * n_new(ui));
        }

        // Check convergence
        double diff = arma::norm(rho_hat_updated - rho_hat_new) / std::max(arma::norm(rho_hat_new), 1e-30);
        rho_hat_new = rho_hat_updated;

        if (diff < config_.cn_tolerance) {
          break;
        }
      }

      // Final: set density from converged Fourier coefficients
      for (long i = 0; i < n_fourier; ++i) {
        fourier_span[i] = rho_hat_new(static_cast<arma::uword>(i));
      }
      cn_fft_.backward();
      double n_total = static_cast<double>(cn_fft_.total());

      arma::vec& rho_mut = density.values();
      for (arma::uword i = 0; i < n; ++i) {
        rho_mut(i) = std::max(real_span[i] / n_total, species::Species::RHO_MIN);
      }
    }

    // Update aliases and recompute final energy
    for (int s = 0; s < n_species; ++s) {
      aliases()[static_cast<size_t>(s)] = mutable_solver().species(s).density_alias();
    }

    return compute_density_forces();
  }

  // ── Nonlinear term ──────────────────────────────────────────────────────

  arma::cx_vec Integrator::compute_nonlinear_term(int species_index) {
    auto& density = mutable_solver().species(species_index).density();
    const arma::vec& rho = density.values();
    const arma::vec& force = mutable_solver().species(species_index).force();
    double d_v = density.cell_volume();
    arma::uword n = density.size();
    long n_fourier = cn_fft_.fourier_total();
    double d = config_.diffusion_coefficient;

    // Compute excess force: dF_ex/drho = (total_force/dV) - ln(rho)
    arma::vec excess_force = force / d_v - arma::log(arma::clamp(rho, 1e-30, arma::datum::inf));

    // Compute rho * excess_force in real space
    auto real_span = cn_fft_.real();
    for (arma::uword i = 0; i < n; ++i) {
      real_span[i] = rho(i) * excess_force(i);
    }
    cn_fft_.forward();

    // In Fourier space: N[rho] = -D * k^2 * FT[rho * c_ex]
    arma::cx_vec result(static_cast<arma::uword>(n_fourier));
    auto fourier_span = cn_fft_.fourier();
    for (long i = 0; i < n_fourier; ++i) {
      auto ui = static_cast<arma::uword>(i);
      result(ui) = -d * k_squared_(ui) * fourier_span[i];
    }

    return result;
  }

  // ── Shared Fourier-space helpers ────────────────────────────────────────

  arma::vec Integrator::compute_k_squared() const {
    const auto& shape = solver().density(0).shape();
    const auto& box = solver().density(0).box_size();

    long nx = shape[0];
    long ny = shape[1];
    long nz = shape[2];
    long nz_half = nz / 2 + 1;
    long n_fourier = nx * ny * nz_half;

    arma::vec k2(static_cast<arma::uword>(n_fourier));

    double kx_factor = 2.0 * std::numbers::pi / box(0);
    double ky_factor = 2.0 * std::numbers::pi / box(1);
    double kz_factor = 2.0 * std::numbers::pi / box(2);

    arma::uword idx = 0;
    for (long ix = 0; ix < nx; ++ix) {
      double kx = (ix <= nx / 2) ? static_cast<double>(ix) : static_cast<double>(ix - nx);
      kx *= kx_factor;
      for (long iy = 0; iy < ny; ++iy) {
        double ky = (iy <= ny / 2) ? static_cast<double>(iy) : static_cast<double>(iy - ny);
        ky *= ky_factor;
        for (long iz = 0; iz < nz_half; ++iz) {
          double kz = static_cast<double>(iz) * kz_factor;
          k2(idx) = kx * kx + ky * ky + kz * kz;
          ++idx;
        }
      }
    }

    return k2;
  }

  void Integrator::apply_diffusion_propagator(int species_index, double dt_propagate) {
    auto& density = mutable_solver().species(species_index).density();

    // Copy density into FFT buffer
    auto real_span = work_fft_.real();
    const arma::vec& rho = density.values();
    arma::uword n = density.size();
    for (arma::uword i = 0; i < n; ++i) {
      real_span[i] = rho(i);
    }

    // Forward FFT
    work_fft_.forward();

    // Apply propagator: rho_hat *= exp(-D * k^2 * dt)
    auto fourier_span = work_fft_.fourier();
    long n_fourier = work_fft_.fourier_total();
    double d = config_.diffusion_coefficient;
    for (long i = 0; i < n_fourier; ++i) {
      double decay = std::exp(-d * k_squared_(static_cast<arma::uword>(i)) * dt_propagate);
      fourier_span[i] *= decay;
    }

    // Inverse FFT
    work_fft_.backward();

    // Copy back (FFTW backward is unnormalized: divide by N)
    double n_total = static_cast<double>(work_fft_.total());
    arma::vec& rho_mut = density.values();
    for (arma::uword i = 0; i < n; ++i) {
      rho_mut(i) = real_span[i] / n_total;
    }
  }

}  // namespace dft::dynamics
