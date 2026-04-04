#ifndef DFT_ALGORITHMS_DDFT_HPP
#define DFT_ALGORITHMS_DDFT_HPP

#include "dft/grid.hpp"
#include "dft/math/fourier.hpp"

#include <algorithm>
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
    double dt_max{1.0};
    double fp_tolerance{1e-4};
    int fp_max_iterations{100};
  };

  // The force callback computes functional derivatives at the current
  // density.  Returns (Omega, {dOmega/drho_s * dV}).

  using ForceCallback = std::function<std::pair<double, std::vector<arma::vec>>(
      const std::vector<arma::vec>& densities
  )>;

  // Result of a single DDFT timestep.

  struct DdftResult {
    std::vector<arma::vec> densities;
    double energy;
    double dt_used;
  };

  // Precomputed arrays for the integrating-factor DDFT scheme.
  // Matches Lutsko's DDFT implementation exactly:
  //   Lambda_k = sum_d 2 * (cos(2 pi i_d / N_d) - 1) / dx_d^2
  // These are the eigenvalues of the 3-point central-difference Laplacian
  // on a periodic grid.

  struct DdftState {
    std::vector<long> shape;
    double dx;
    long fourier_total;

    // Per-axis Laplacian eigenvalues (real-space freq index).
    arma::vec lam_x, lam_y, lam_z;

    // dt-dependent integrating factors: exp(lam_d * dt) per axis.
    arma::vec fx, fy, fz;
    double dt;
  };

  [[nodiscard]] inline auto make_ddft_state(const Grid& grid) -> DdftState {
    long nx = grid.shape[0];
    long ny = grid.shape[1];
    long nz = grid.shape[2];
    long nz_half = nz / 2 + 1;
    double dx = grid.dx;
    double Dx = 1.0 / (dx * dx);

    arma::vec lx(static_cast<arma::uword>(nx));
    for (long ix = 0; ix < nx; ++ix) {
      double kx = 2.0 * std::numbers::pi * static_cast<double>(ix) / static_cast<double>(nx);
      lx(static_cast<arma::uword>(ix)) = 2.0 * Dx * (std::cos(kx) - 1.0);
    }

    arma::vec ly(static_cast<arma::uword>(ny));
    for (long iy = 0; iy < ny; ++iy) {
      double ky = 2.0 * std::numbers::pi * static_cast<double>(iy) / static_cast<double>(ny);
      ly(static_cast<arma::uword>(iy)) = 2.0 * Dx * (std::cos(ky) - 1.0);
    }

    arma::vec lz(static_cast<arma::uword>(nz_half));
    for (long iz = 0; iz < nz_half; ++iz) {
      double kz = 2.0 * std::numbers::pi * static_cast<double>(iz) / static_cast<double>(nz);
      lz(static_cast<arma::uword>(iz)) = 2.0 * Dx * (std::cos(kz) - 1.0);
    }

    return DdftState{
        .shape = {nx, ny, nz},
        .dx = dx,
        .fourier_total = nx * ny * nz_half,
        .lam_x = std::move(lx),
        .lam_y = std::move(ly),
        .lam_z = std::move(lz),
        .fx = {},
        .fy = {},
        .fz = {},
        .dt = 0.0,
    };
  }

  // Recompute exp(lam * dt) factors when dt changes.

  inline void update_timestep(DdftState& st, double dt) {
    if (std::abs(st.dt - dt) < 1e-30) return;
    st.dt = dt;
    st.fx = arma::exp(st.lam_x * dt);
    st.fy = arma::exp(st.lam_y * dt);
    st.fz = arma::exp(st.lam_z * dt);
  }

  // ──────────────────────────────────────────────────────────────────
  // Finite-difference operators (real space, matching Jim exactly).
  // ──────────────────────────────────────────────────────────────────

  // g_dot_x: computes div(rho * grad(x)) in real space using central
  // differences.  Result is stored in gx.
  //   gx[pos] = sum_d D_d * ((rho_{+d} + rho_0)*(x_{+d} - x_0)
  //                        - (rho_0 + rho_{-d})*(x_0 - x_{-d}))
  // where D_d = 1 / (2 * dx_d^2).

  inline void g_dot_x(
      const arma::vec& rho, const arma::vec& x, arma::vec& gx,
      long nx, long ny, long nz, double dx
  ) {
    double D = 0.5 / (dx * dx);
    long nyz = ny * nz;

    for (long ix = 0; ix < nx; ++ix) {
      long ixp = ((ix + 1) % nx) * nyz;
      long ixm = ((ix - 1 + nx) % nx) * nyz;
      long ix0 = ix * nyz;
      for (long iy = 0; iy < ny; ++iy) {
        long iyp = ((iy + 1) % ny) * nz;
        long iym = ((iy - 1 + ny) % ny) * nz;
        long iy0 = iy * nz;
        for (long iz = 0; iz < nz; ++iz) {
          long izp = (iz + 1) % nz;
          long izm = (iz - 1 + nz) % nz;

          long pos = ix0 + iy0 + iz;
          auto p = static_cast<arma::uword>(pos);
          double d0 = rho(p);
          double x0 = x(p);

          auto px = static_cast<arma::uword>(ixp + iy0 + iz);
          auto mx = static_cast<arma::uword>(ixm + iy0 + iz);
          auto py = static_cast<arma::uword>(ix0 + iyp + iz);
          auto my = static_cast<arma::uword>(ix0 + iym + iz);
          auto pz = static_cast<arma::uword>(ix0 + iy0 + izp);
          auto mz = static_cast<arma::uword>(ix0 + iy0 + izm);

          gx(p) = D * ((rho(px) + d0) * (x(px) - x0) - (d0 + rho(mx)) * (x0 - x(mx)))
                 + D * ((rho(py) + d0) * (x(py) - x0) - (d0 + rho(my)) * (x0 - x(my)))
                 + D * ((rho(pz) + d0) * (x(pz) - x0) - (d0 + rho(mz)) * (x0 - x(mz)));
        }
      }
    }
  }

  // subtract_ideal_gas: RHS -= Laplacian(rho) using the same 3-point
  // central-difference stencil whose eigenvalues are Lambda_k.

  inline void subtract_ideal_gas(
      const arma::vec& rho, arma::vec& rhs,
      long nx, long ny, long nz, double dx
  ) {
    double D = 1.0 / (dx * dx);
    long nyz = ny * nz;

    for (long ix = 0; ix < nx; ++ix) {
      long ixp = ((ix + 1) % nx) * nyz;
      long ixm = ((ix - 1 + nx) % nx) * nyz;
      long ix0 = ix * nyz;
      for (long iy = 0; iy < ny; ++iy) {
        long iyp = ((iy + 1) % ny) * nz;
        long iym = ((iy - 1 + ny) % ny) * nz;
        long iy0 = iy * nz;
        for (long iz = 0; iz < nz; ++iz) {
          long izp = (iz + 1) % nz;
          long izm = (iz - 1 + nz) % nz;

          long pos = ix0 + iy0 + iz;
          auto p = static_cast<arma::uword>(pos);
          double d0 = rho(p);

          auto px = static_cast<arma::uword>(ixp + iy0 + iz);
          auto mx = static_cast<arma::uword>(ixm + iy0 + iz);
          auto py = static_cast<arma::uword>(ix0 + iyp + iz);
          auto my = static_cast<arma::uword>(ix0 + iym + iz);
          auto pz = static_cast<arma::uword>(ix0 + iy0 + izp);
          auto mz = static_cast<arma::uword>(ix0 + iy0 + izm);

          rhs(p) -= D * (rho(px) + rho(mx) - 2.0 * d0)
                  + D * (rho(py) + rho(my) - 2.0 * d0)
                  + D * (rho(pz) + rho(mz) - 2.0 * d0);
        }
      }
    }
  }

  // ──────────────────────────────────────────────────────────────────
  // Compute the excess (nonlinear) RHS and FFT it.
  //
  // R(rho) = div(rho * grad(dOmega/drho)) - Laplacian(rho)
  //
  // The forces from the functional are dOmega/drho_i * dV.
  // Dividing by dV gives dOmega/drho_i.  Then g_dot_x computes
  // div(rho * grad(dOmega/drho)).  Subtracting the ideal-gas
  // Laplacian leaves only the excess contribution.
  //
  // Matches Jim's calculate_excess_RHS:
  //   g_dot_x(dF, RHS);            // dF includes dV factor
  //   RHS.MultBy(1/dV);            // strip it => div(rho*grad(dOmega/drho))
  //   subtract_ideal_gas(rho,RHS);

  inline void compute_excess_rhs(
      const arma::vec& rho, const arma::vec& forces, double dv,
      long nx, long ny, long nz, double dx,
      math::FourierTransform& rhs_ft
  ) {
    auto n = static_cast<arma::uword>(nx * ny * nz);
    arma::vec df_drho = forces / dv;
    arma::vec rhs(n);
    g_dot_x(rho, df_drho, rhs, nx, ny, nz, dx);
    subtract_ideal_gas(rho, rhs, nx, ny, nz, dx);
    rhs_ft.set_real(rhs);
    rhs_ft.forward();
  }

  // ──────────────────────────────────────────────────────────────────
  // Apply the diffusion propagator (integrating factor):
  //
  //   d1_k = U0 * d0_k + U1 * R0_k + U2 * (R1_k - R0_k)
  //
  // where U0 = exp(L*dt), U1 = (exp(L*dt)-1)/L, U2 = (exp(L*dt)-1-L*dt)/(L^2*dt)
  //
  // Returns the maximum absolute change |d1_new - d1_old|.
  // ──────────────────────────────────────────────────────────────────

  inline auto apply_propagator(
      const DdftState& st,
      const math::FourierTransform& rho0_ft,
      const math::FourierTransform& rhs0_ft,
      const math::FourierTransform& rhs1_ft,
      arma::vec& d1
  ) -> double {
    long nx = st.shape[0];
    long ny = st.shape[1];
    long nz_half = st.shape[2] / 2 + 1;
    double dt = st.dt;
    long n_total = nx * ny * st.shape[2];

    auto shape_vec = std::vector<long>(st.shape.begin(), st.shape.end());
    math::FourierTransform work(shape_vec);

    auto rho0_k = rho0_ft.fourier();
    auto r0_k = rhs0_ft.fourier();
    auto r1_k = rhs1_ft.fourier();
    auto cwork = work.fourier();

    long pos = 0;
    for (long ix = 0; ix < nx; ++ix) {
      auto ux = static_cast<arma::uword>(ix);
      for (long iy = 0; iy < ny; ++iy) {
        auto uy = static_cast<arma::uword>(iy);
        for (long iz = 0; iz < nz_half; ++iz) {
          auto uz = static_cast<arma::uword>(iz);

          double Lambda = st.lam_x(ux) + st.lam_y(uy) + st.lam_z(uz);
          double exp_dt = st.fx(ux) * st.fy(uy) * st.fz(uz);

          double U0 = exp_dt;
          double U1 = (pos == 0) ? dt : (exp_dt - 1.0) / Lambda;
          double U2 = (pos == 0) ? dt / 2.0 : (exp_dt - 1.0 - dt * Lambda) / (Lambda * Lambda * dt);

          cwork[pos] = U0 * rho0_k[pos] + U1 * r0_k[pos] + U2 * (r1_k[pos] - r0_k[pos]);
          ++pos;
        }
      }
    }

    work.backward();
    auto work_real = work.real();
    double inv_n = 1.0 / static_cast<double>(n_total);

    double max_dev = 0.0;
    for (arma::uword i = 0; i < d1.n_elem; ++i) {
      double new_val = work_real[i] * inv_n;
      double dev = std::abs(d1(i) - new_val);
      if (dev > max_dev) max_dev = dev;
      d1(i) = new_val;
    }

    return max_dev;
  }

  // ──────────────────────────────────────────────────────────────────
  // Integrating-factor DDFT step (Lutsko's scheme).
  //
  // Solves:
  //   d rho / dt = Laplacian(rho) + R(rho)
  //
  // where R(rho) = div(rho * grad(dF/drho)) - Laplacian(rho)
  // is the excess (nonlinear) contribution.
  //
  // The equation is solved with the integrating factor for the
  // exact Laplacian and implicit fixed-point iteration for R.
  // If convergence fails or density goes negative, the timestep
  // is halved and the step is restarted.
  // ──────────────────────────────────────────────────────────────────

  [[nodiscard]] inline auto integrating_factor_step(
      const std::vector<arma::vec>& densities, const Grid& grid,
      DdftState& st, const ForceCallback& compute, DdftConfig& config
  ) -> DdftResult {
    long nx = st.shape[0];
    long ny = st.shape[1];
    long nz = st.shape[2];
    double dx = st.dx;
    double dv = grid.cell_volume();
    auto shape_vec = std::vector<long>(st.shape.begin(), st.shape.end());
    long n_total = nx * ny * nz;

    // Evaluate forces at current density.
    auto [energy, forces] = compute(densities);

    // Compute RHS0 = excess nonlinear term at current density.
    math::FourierTransform rhs0(shape_vec);
    compute_excess_rhs(densities[0], forces[0], dv, nx, ny, nz, dx, rhs0);

    // FFT of current density (needed by propagator).
    math::FourierTransform rho0_ft(shape_vec);
    rho0_ft.set_real(densities[0]);
    rho0_ft.forward();

    // Working density for iteration.
    arma::vec d1 = densities[0];

    bool restart;
    do {
      restart = false;
      update_timestep(st, config.dt);
      d1 = densities[0];

      math::FourierTransform rhs1(shape_vec);
      double deviation = 1e30;

      for (int iter = 0; iter < config.fp_max_iterations; ++iter) {
        // Compute forces at d1.
        auto [e1, f1] = compute({d1});
        compute_excess_rhs(d1, f1[0], dv, nx, ny, nz, dx, rhs1);

        double old_deviation = deviation;
        deviation = apply_propagator(st, rho0_ft, rhs0, rhs1, d1);

        // Check for negative density or divergence.
        if (d1.min() < 0.0 || (iter > 0 && deviation > old_deviation)) {
          restart = true;
          break;
        }

        if (deviation < config.fp_tolerance) break;
      }

      // Non-convergence: also restart with smaller dt.
      if (!restart && deviation > config.fp_tolerance) restart = true;

      if (restart) {
        config.dt /= 10.0;
        d1 = densities[0];
      }
    } while (restart);

    // Final energy evaluation.
    auto [e_final, f_final] = compute({d1});

    return DdftResult{
        .densities = {d1},
        .energy = e_final,
        .dt_used = config.dt,
    };
  }

  // Precomputed k^2 values for the half-complex FFT grid (kept for
  // backward compatibility with any code using split_operator_step).

  [[nodiscard]] inline auto compute_k_squared(const Grid& grid) -> arma::vec {
    long nz_half = grid.shape[2] / 2 + 1;
    long fourier_total = grid.shape[0] * grid.shape[1] * nz_half;
    arma::vec k2(static_cast<arma::uword>(fourier_total));

    for_each_wavevector(grid, [&](const Wavevector& wv) {
      k2(static_cast<arma::uword>(wv.idx)) = wv.norm2();
    });

    return k2;
  }

  [[nodiscard]] inline auto diffusion_propagator(
      const arma::vec& k_squared, double D, double dt
  ) -> arma::vec {
    return arma::exp(-D * k_squared * dt);
  }

  // ──────────────────────────────────────────────────────────────────
  // Compute excess RHS in real space (no FFT).
  //
  // R(rho) = div(rho * grad(dF/drho)) - Laplacian(rho)
  //
  // Same as compute_excess_rhs but returns the real-space vector
  // instead of FFT-ing it.
  // ──────────────────────────────────────────────────────────────────

  [[nodiscard]] inline auto excess_rhs_real(
      const arma::vec& rho, const arma::vec& forces, double dv,
      long nx, long ny, long nz, double dx
  ) -> arma::vec {
    auto n = static_cast<arma::uword>(nx * ny * nz);
    arma::vec df_drho = forces / dv;
    arma::vec rhs(n);
    g_dot_x(rho, df_drho, rhs, nx, ny, nz, dx);
    subtract_ideal_gas(rho, rhs, nx, ny, nz, dx);
    return rhs;
  }

  // ──────────────────────────────────────────────────────────────────
  // Split-operator step (Strang splitting).
  //
  // Decomposes  d rho / dt = Laplacian(rho) + R(rho)  into:
  //
  //   A: d rho / dt = Laplacian(rho)  [diffusion, exact in Fourier space]
  //   B: d rho / dt = R(rho)          [excess nonlinear, forward Euler]
  //
  // Second-order Strang splitting:
  //   A(dt/2)  ->  B(dt)  ->  A(dt/2)
  //
  // The propagator argument is exp(-D * k^2 * dt) for the full step;
  // the half-step uses sqrt(propagator).
  // ──────────────────────────────────────────────────────────────────

  [[nodiscard]] inline auto split_operator_step(
      const std::vector<arma::vec>& densities, const Grid& grid,
      const arma::vec& k_squared, const arma::vec& propagator,
      const ForceCallback& compute, const DdftConfig& config
  ) -> DdftResult {
    auto shape = std::vector<long>(grid.shape.begin(), grid.shape.end());
    long n_total = grid.total_points();
    double dv = grid.cell_volume();
    double inv_n = 1.0 / static_cast<double>(n_total);
    long nx = grid.shape[0], ny = grid.shape[1], nz = grid.shape[2];

    arma::vec half_prop = arma::sqrt(propagator);

    // Step 1: half-step diffusion in Fourier space.
    math::FourierTransform ft(shape);
    ft.set_real(densities[0]);
    ft.forward();
    {
      auto fk = ft.fourier();
      for (arma::uword i = 0; i < half_prop.n_elem; ++i) {
        fk[i] *= half_prop(i);
      }
    }
    ft.backward();
    arma::vec rho_star = ft.real_vec() * inv_n;
    rho_star = arma::clamp(rho_star, config.min_density, arma::datum::inf);

    // Step 2: full nonlinear step (forward Euler).
    auto [energy, forces] = compute({rho_star});
    arma::vec rhs = excess_rhs_real(rho_star, forces[0], dv, nx, ny, nz, grid.dx);
    arma::vec rho_dstar = rho_star + config.dt * rhs;
    rho_dstar = arma::clamp(rho_dstar, config.min_density, arma::datum::inf);

    // Step 3: half-step diffusion again.
    ft.set_real(rho_dstar);
    ft.forward();
    {
      auto fk = ft.fourier();
      for (arma::uword i = 0; i < half_prop.n_elem; ++i) {
        fk[i] *= half_prop(i);
      }
    }
    ft.backward();
    arma::vec d1 = ft.real_vec() * inv_n;
    d1 = arma::clamp(d1, config.min_density, arma::datum::inf);

    auto [e_final, f_final] = compute({d1});
    return DdftResult{
        .densities = {d1},
        .energy = e_final,
        .dt_used = config.dt,
    };
  }

  // ──────────────────────────────────────────────────────────────────
  // Crank-Nicholson step.
  //
  // Semi-implicit discretization of  d rho / dt = -D k^2 rho + R(rho):
  //
  //   rho^{n+1} = [ (1 - a) rho^n + dt/2 (R^n + R^{n+1}) ] / (1 + a)
  //
  // where  a = D k^2 dt / 2.   R^{n+1} depends on rho^{n+1} so the
  // scheme is iterated to self-consistency (fixed-point).
  // ──────────────────────────────────────────────────────────────────

  [[nodiscard]] inline auto crank_nicholson_step(
      const std::vector<arma::vec>& densities, const Grid& grid,
      const arma::vec& k_squared, const ForceCallback& compute,
      const DdftConfig& config
  ) -> DdftResult {
    auto shape = std::vector<long>(grid.shape.begin(), grid.shape.end());
    long n_total = grid.total_points();
    double dv = grid.cell_volume();
    double inv_n = 1.0 / static_cast<double>(n_total);
    long nx = grid.shape[0], ny = grid.shape[1], nz = grid.shape[2];
    double D = config.diffusion_coefficient;
    double dt = config.dt;

    // CN coefficients per Fourier mode.
    arma::vec a_coeff = D * k_squared * dt * 0.5;
    arma::vec cn_minus = 1.0 - a_coeff;  // (1 - a)
    arma::vec cn_plus_inv = 1.0 / (1.0 + a_coeff);  // 1/(1 + a)

    // FFT of current density.
    math::FourierTransform rho0_ft(shape);
    rho0_ft.set_real(densities[0]);
    rho0_ft.forward();

    // Compute R^n.
    auto [energy0, forces0] = compute(densities);
    arma::vec rhs0 = excess_rhs_real(densities[0], forces0[0], dv, nx, ny, nz, grid.dx);
    math::FourierTransform r0_ft(shape);
    r0_ft.set_real(rhs0);
    r0_ft.forward();

    // Iterative solve: start with d1 = current density.
    arma::vec d1 = densities[0];
    math::FourierTransform work(shape);

    for (int iter = 0; iter < config.fp_max_iterations; ++iter) {
      auto [e1, f1] = compute({d1});
      arma::vec rhs1 = excess_rhs_real(d1, f1[0], dv, nx, ny, nz, grid.dx);
      math::FourierTransform r1_ft(shape);
      r1_ft.set_real(rhs1);
      r1_ft.forward();

      auto rho0_k = rho0_ft.fourier();
      auto r0_k = r0_ft.fourier();
      auto r1_k = r1_ft.fourier();
      auto wk = work.fourier();

      for (arma::uword i = 0; i < k_squared.n_elem; ++i) {
        wk[i] = (cn_minus(i) * rho0_k[i] + (dt * 0.5) * (r0_k[i] + r1_k[i])) * cn_plus_inv(i);
      }

      work.backward();
      arma::vec d1_new = work.real_vec() * inv_n;
      d1_new = arma::clamp(d1_new, config.min_density, arma::datum::inf);

      double max_dev = arma::max(arma::abs(d1_new - d1));
      d1 = d1_new;

      if (max_dev < config.fp_tolerance) break;
    }

    auto [e_final, f_final] = compute({d1});
    return DdftResult{
        .densities = {d1},
        .energy = e_final,
        .dt_used = dt,
    };
  }

}  // namespace dft::algorithms::ddft

#endif  // DFT_ALGORITHMS_DDFT_HPP
