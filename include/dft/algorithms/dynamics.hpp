#ifndef DFT_ALGORITHMS_DYNAMICS_HPP
#define DFT_ALGORITHMS_DYNAMICS_HPP

#include "dft/fields.hpp"
#include "dft/grid.hpp"
#include "dft/math/fourier.hpp"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <numbers>
#include <vector>

namespace dft::algorithms::dynamics {

  // The force callback computes functional derivatives at the current
  // density.  Returns (Omega, {dOmega/drho_s * dV}).

  using ForceCallback =
      std::function<std::pair<double, std::vector<arma::vec>>(const std::vector<arma::vec>& densities)>;

  // Configuration for a single DDFT timestep.

  struct StepConfig {
    double dt{1e-4};
    double diffusion_coefficient{1.0};
    double min_density{1e-18};
    double dt_max{1.0};
    double fp_tolerance{1e-4};
    int fp_max_iterations{100};
  };

  // Result of a single DDFT timestep.

  struct StepResult {
    std::vector<arma::vec> densities;
    double energy;
    double dt_used;
  };

  // A single snapshot captured during the simulation.

  struct Snapshot {
    int step;
    double time;
    double energy;
    std::vector<arma::vec> densities;
  };

  // Complete result of a DDFT simulation.

  struct SimulationResult {
    std::vector<arma::vec> densities;
    std::vector<double> times;
    std::vector<double> energies;
    std::vector<Snapshot> snapshots;
    double mass_initial;
    double mass_final;
  };

  namespace _internal {

    // Precomputed arrays for the integrating-factor DDFT scheme.
    // Lambda_k = sum_d 2 * (cos(2 pi i_d / N_d) - 1) / dx_d^2
    // are the eigenvalues of the 3-point central-difference Laplacian.
    // For non-periodic axes the FFT operates on the doubled
    // (mirror-extended) grid, which automatically produces Neumann
    // eigenvalues: lambda_k = 2*(cos(pi*k/N)-1)/dx^2.

    struct IntegratingFactorState {
      Grid grid;
      long fourier_total;
      arma::vec lam_x, lam_y, lam_z;
      arma::vec fx, fy, fz;
      double dt;
      math::FourierTransform rho0_ft;
      math::FourierTransform rhs0_ft;
      math::FourierTransform rhs1_ft;
      math::FourierTransform work;
    };

    [[nodiscard]] inline auto make_if_state(const Grid& grid) -> IntegratingFactorState {
      auto cs = grid.convolution_shape();
      long nx = cs[0];
      long ny = cs[1];
      long nz = cs[2];
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

      return IntegratingFactorState{
          .grid = grid,
          .fourier_total = nx * ny * nz_half,
          .lam_x = std::move(lx),
          .lam_y = std::move(ly),
          .lam_z = std::move(lz),
          .fx = {},
          .fy = {},
          .fz = {},
          .dt = 0.0,
          .rho0_ft = math::FourierTransform({nx, ny, nz}),
          .rhs0_ft = math::FourierTransform({nx, ny, nz}),
          .rhs1_ft = math::FourierTransform({nx, ny, nz}),
          .work = math::FourierTransform({nx, ny, nz}),
      };
    }

    // Recompute exp(lam * dt) factors when dt changes.

    inline void update_timestep(IntegratingFactorState& st, double dt) {
      if (std::abs(st.dt - dt) < 1e-30) {
        return;
      }
      st.dt = dt;
      st.fx = arma::exp(st.lam_x * dt);
      st.fy = arma::exp(st.lam_y * dt);
      st.fz = arma::exp(st.lam_z * dt);
    }

    // Finite-difference operators (real space).
    // For non-periodic axes, Neumann (no-flux) BCs are applied by
    // reflecting the ghost-point index: f[-1] = f[1], f[N] = f[N-2].

    // div(rho * grad(x)) via central differences.

    inline void divergence_flux(
        const arma::vec& rho,
        const arma::vec& x,
        arma::vec& gx,
        long nx,
        long ny,
        long nz,
        double dx,
        const std::array<bool, 3>& periodic = {true, true, true}
    ) {
      double D = 0.5 / (dx * dx);
      long nyz = ny * nz;

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (long ix = 0; ix < nx; ++ix) {
        long ixp = periodic[0] ? ((ix + 1) % nx) * nyz : std::min(ix + 1, nx - 1) * nyz;
        long ixm = periodic[0] ? ((ix - 1 + nx) % nx) * nyz : std::max(ix - 1, 0L) * nyz;
        long ix0 = ix * nyz;
        for (long iy = 0; iy < ny; ++iy) {
          long iyp = periodic[1] ? ((iy + 1) % ny) * nz : std::min(iy + 1, ny - 1) * nz;
          long iym = periodic[1] ? ((iy - 1 + ny) % ny) * nz : std::max(iy - 1, 0L) * nz;
          long iy0 = iy * nz;
          for (long iz = 0; iz < nz; ++iz) {
            long izp = periodic[2] ? (iz + 1) % nz : std::min(iz + 1, nz - 1);
            long izm = periodic[2] ? (iz - 1 + nz) % nz : std::max(iz - 1, 0L);

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

    // Subtract the ideal-gas Laplacian: rhs -= Laplacian(rho).

    inline void subtract_laplacian(
        const arma::vec& rho,
        arma::vec& rhs,
        long nx,
        long ny,
        long nz,
        double dx,
        const std::array<bool, 3>& periodic = {true, true, true}
    ) {
      double D = 1.0 / (dx * dx);
      long nyz = ny * nz;

#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (long ix = 0; ix < nx; ++ix) {
        long ixp = periodic[0] ? ((ix + 1) % nx) * nyz : std::min(ix + 1, nx - 1) * nyz;
        long ixm = periodic[0] ? ((ix - 1 + nx) % nx) * nyz : std::max(ix - 1, 0L) * nyz;
        long ix0 = ix * nyz;
        for (long iy = 0; iy < ny; ++iy) {
          long iyp = periodic[1] ? ((iy + 1) % ny) * nz : std::min(iy + 1, ny - 1) * nz;
          long iym = periodic[1] ? ((iy - 1 + ny) % ny) * nz : std::max(iy - 1, 0L) * nz;
          long iy0 = iy * nz;
          for (long iz = 0; iz < nz; ++iz) {
            long izp = periodic[2] ? (iz + 1) % nz : std::min(iz + 1, nz - 1);
            long izm = periodic[2] ? (iz - 1 + nz) % nz : std::max(iz - 1, 0L);

            long pos = ix0 + iy0 + iz;
            auto p = static_cast<arma::uword>(pos);
            double d0 = rho(p);

            auto px = static_cast<arma::uword>(ixp + iy0 + iz);
            auto mx = static_cast<arma::uword>(ixm + iy0 + iz);
            auto py = static_cast<arma::uword>(ix0 + iyp + iz);
            auto my = static_cast<arma::uword>(ix0 + iym + iz);
            auto pz = static_cast<arma::uword>(ix0 + iy0 + izp);
            auto mz = static_cast<arma::uword>(ix0 + iy0 + izm);

            rhs(p) -= D * (rho(px) + rho(mx) - 2.0 * d0) + D * (rho(py) + rho(my) - 2.0 * d0)
                + D * (rho(pz) + rho(mz) - 2.0 * d0);
          }
        }
      }
    }

    // Compute the excess (nonlinear) RHS and FFT it.
    //
    // R(rho) = div(rho * grad(dOmega/drho)) - Laplacian(rho)
    //
    // The stencils use the grid's periodicity flags. For non-periodic
    // axes the result is mirror-extended before the FFT so that the
    // transform matches the Neumann-eigenvalue basis.

    inline void excess_rhs_fourier(
        const arma::vec& rho,
        const arma::vec& forces,
        double dv,
        const Grid& grid,
        math::FourierTransform& rhs_ft,
        const arma::uvec& frozen_mask = {}
    ) {
      long nx = grid.shape[0], ny = grid.shape[1], nz = grid.shape[2];
      auto n = static_cast<arma::uword>(nx * ny * nz);
      arma::vec df_drho = forces / dv;
      arma::vec rhs(n);
      divergence_flux(rho, df_drho, rhs, nx, ny, nz, grid.dx, grid.periodic);
      subtract_laplacian(rho, rhs, nx, ny, nz, grid.dx, grid.periodic);
      if (!frozen_mask.is_empty()) {
        rhs.elem(arma::find(frozen_mask)).zeros();
      }
      arma::vec rhs_ext = mirror_extend(rhs, grid);
      rhs_ft.set_real(rhs_ext);
      rhs_ft.forward();
    }

    // Compute excess RHS in real space (no FFT).

    [[nodiscard]] inline auto
    excess_rhs_real(const arma::vec& rho, const arma::vec& forces, double dv, const Grid& grid) -> arma::vec {
      long nx = grid.shape[0], ny = grid.shape[1], nz = grid.shape[2];
      auto n = static_cast<arma::uword>(nx * ny * nz);
      arma::vec df_drho = forces / dv;
      arma::vec rhs(n);
      divergence_flux(rho, df_drho, rhs, nx, ny, nz, grid.dx, grid.periodic);
      subtract_laplacian(rho, rhs, nx, ny, nz, grid.dx, grid.periodic);
      return rhs;
    }

    // Apply the diffusion propagator (integrating factor).
    // Returns the maximum absolute change |d1_new - d1_old|.
    // FFT objects operate on the convolution shape (doubled in
    // non-periodic directions); the physical-grid density d1 is
    // obtained by unpacking the first N points.

    inline auto apply_propagator(
        IntegratingFactorState& st,
        const math::FourierTransform& rho0_ft,
        const math::FourierTransform& rhs0_ft,
        const math::FourierTransform& rhs1_ft,
        arma::vec& d1
    ) -> double {
      auto cs = st.grid.convolution_shape();
      long nx = cs[0];
      long ny = cs[1];
      long nz_half = cs[2] / 2 + 1;
      double dt = st.dt;
      long n_fft = cs[0] * cs[1] * cs[2];

      auto rho0_k = rho0_ft.fourier();
      auto r0_k = rhs0_ft.fourier();
      auto r1_k = rhs1_ft.fourier();
      auto cwork = st.work.fourier();

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
      for (long ix = 0; ix < nx; ++ix) {
        auto ux = static_cast<arma::uword>(ix);
        for (long iy = 0; iy < ny; ++iy) {
          auto uy = static_cast<arma::uword>(iy);
          for (long iz = 0; iz < nz_half; ++iz) {
            auto uz = static_cast<arma::uword>(iz);
            long pos = iz + nz_half * (iy + ny * ix);

            double Lambda = st.lam_x(ux) + st.lam_y(uy) + st.lam_z(uz);
            double exp_dt = st.fx(ux) * st.fy(uy) * st.fz(uz);

            double U0 = exp_dt;
            double U1 = (pos == 0) ? dt : (exp_dt - 1.0) / Lambda;
            double U2 = (pos == 0) ? dt / 2.0 : (exp_dt - 1.0 - dt * Lambda) / (Lambda * Lambda * dt);

            cwork[pos] = U0 * rho0_k[pos] + U1 * r0_k[pos] + U2 * (r1_k[pos] - r0_k[pos]);
          }
        }
      }

      st.work.backward();
      arma::vec work_vec = st.work.real_vec();
      double inv_n = 1.0 / static_cast<double>(n_fft);
      work_vec *= inv_n;

      arma::vec phys = unpad(work_vec, st.grid);

      double max_dev = 0.0;
      for (arma::uword i = 0; i < d1.n_elem; ++i) {
        double dev = std::abs(d1(i) - phys(i));
        if (dev > max_dev) {
          max_dev = dev;
        }
        d1(i) = phys(i);
      }

      return max_dev;
    }

  } // namespace _internal

  // Precomputed k^2 values for the half-complex FFT grid.

  [[nodiscard]] inline auto compute_k_squared(const Grid& grid) -> arma::vec {
    long nz_half = grid.shape[2] / 2 + 1;
    long fourier_total = grid.shape[0] * grid.shape[1] * nz_half;
    arma::vec k2(static_cast<arma::uword>(fourier_total));

    grid.for_each_wavevector([&](const Wavevector& wv) { k2(static_cast<arma::uword>(wv.idx)) = wv.norm2(); });

    return k2;
  }

  [[nodiscard]] inline auto diffusion_propagator(const arma::vec& k_squared, double D, double dt) -> arma::vec {
    return arma::exp(-D * k_squared * dt);
  }

  // Integrating-factor DDFT step (Lutsko's scheme).
  //
  // Solves: d rho / dt = Laplacian(rho) + R(rho)
  // where R(rho) = div(rho * grad(dF/drho)) - Laplacian(rho)
  //
  // Uses integrating factor for the exact Laplacian and implicit
  // fixed-point iteration for R. If convergence fails or density
  // goes negative, the timestep is reduced and the step restarts.

  // Boundary condition applied to densities after each DDFT timestep.
  // Empty (default): periodic boundaries, mass is conserved (canonical).
  // For open systems: use reservoir_boundary() or frozen_boundary()
  // to create Dirichlet BCs.

  using BoundaryCondition = std::function<void(std::vector<arma::vec>&)>;

  [[nodiscard]] inline auto integrating_factor_step(
      const std::vector<arma::vec>& densities,
      const Grid& grid,
      _internal::IntegratingFactorState& st,
      const ForceCallback& compute,
      StepConfig& config,
      const BoundaryCondition& boundary = {},
      const arma::uvec& frozen_mask = {}
  ) -> StepResult {
    double dv = grid.cell_volume();

    auto [energy, forces] = compute(densities);

    _internal::excess_rhs_fourier(densities[0], forces[0], dv, grid, st.rhs0_ft, frozen_mask);

    arma::vec rho_ext = mirror_extend(densities[0], grid);
    st.rho0_ft.set_real(rho_ext);
    st.rho0_ft.forward();

    arma::vec d1 = densities[0];

    bool restart;
    int restarts = 0;
    constexpr int max_restarts = 20;
    do {
      restart = false;
      _internal::update_timestep(st, config.dt);
      d1 = densities[0];

      double deviation = 1e30;

      for (int iter = 0; iter < config.fp_max_iterations; ++iter) {
        auto [e1, f1] = compute({d1});
        _internal::excess_rhs_fourier(d1, f1[0], dv, grid, st.rhs1_ft, frozen_mask);

        double old_deviation = deviation;
        deviation = _internal::apply_propagator(st, st.rho0_ft, st.rhs0_ft, st.rhs1_ft, d1);

        if (boundary) {
          std::vector<arma::vec> tmp{d1};
          boundary(tmp);
          d1 = std::move(tmp[0]);
        }

        if (d1.min() < 0.0 || (iter > 0 && deviation > old_deviation)) {
          restart = true;
          break;
        }

        if (deviation < config.fp_tolerance) {
          break;
        }
      }

      if (!restart && deviation > config.fp_tolerance) {
        restart = true;
      }

      if (restart) {
        config.dt /= 10.0;
        d1 = densities[0];
        if (++restarts > max_restarts) {
          break;
        }
      }
    } while (restart);

    auto [e_final, f_final] = compute({d1});

    return StepResult{
        .densities = {d1},
        .energy = e_final,
        .dt_used = config.dt,
    };
  }

  // Split-operator step (Strang splitting).
  //
  // Decomposes  d rho / dt = Laplacian(rho) + R(rho)  into:
  //   A(dt/2) -> B(dt) -> A(dt/2)
  // where A is exact diffusion in Fourier space and B is forward
  // Euler for the excess nonlinear contribution.

  [[nodiscard]] inline auto split_operator_step(
      const std::vector<arma::vec>& densities,
      const Grid& grid,
      const arma::vec& k_squared,
      const arma::vec& propagator,
      const ForceCallback& compute,
      const StepConfig& config
  ) -> StepResult {
    auto shape = std::vector<long>(grid.shape.begin(), grid.shape.end());
    long n_total = grid.total_points();
    double dv = grid.cell_volume();
    double inv_n = 1.0 / static_cast<double>(n_total);
    long nx = grid.shape[0], ny = grid.shape[1], nz = grid.shape[2];

    arma::vec half_prop = arma::sqrt(propagator);

    // Half-step diffusion.
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

    // Full nonlinear step (forward Euler).
    auto [energy, forces] = compute({rho_star});
    arma::vec rhs = _internal::excess_rhs_real(rho_star, forces[0], dv, grid);
    arma::vec rho_dstar = rho_star + config.dt * rhs;
    rho_dstar = arma::clamp(rho_dstar, config.min_density, arma::datum::inf);

    // Half-step diffusion.
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
    return StepResult{
        .densities = {d1},
        .energy = e_final,
        .dt_used = config.dt,
    };
  }

  // Crank-Nicholson step.
  //
  // Semi-implicit: rho^{n+1} = [(1-a) rho^n + dt/2 (R^n + R^{n+1})] / (1+a)
  // where a = D k^2 dt / 2. Iterated to self-consistency.

  [[nodiscard]] inline auto crank_nicholson_step(
      const std::vector<arma::vec>& densities,
      const Grid& grid,
      const arma::vec& k_squared,
      const ForceCallback& compute,
      const StepConfig& config
  ) -> StepResult {
    auto shape = std::vector<long>(grid.shape.begin(), grid.shape.end());
    long n_total = grid.total_points();
    double dv = grid.cell_volume();
    double inv_n = 1.0 / static_cast<double>(n_total);
    long nx = grid.shape[0], ny = grid.shape[1], nz = grid.shape[2];
    double D = config.diffusion_coefficient;
    double dt = config.dt;

    arma::vec a_coeff = D * k_squared * dt * 0.5;
    arma::vec cn_minus = 1.0 - a_coeff;
    arma::vec cn_plus_inv = 1.0 / (1.0 + a_coeff);

    math::FourierTransform rho0_ft(shape);
    rho0_ft.set_real(densities[0]);
    rho0_ft.forward();

    auto [energy0, forces0] = compute(densities);
    arma::vec rhs0 = _internal::excess_rhs_real(densities[0], forces0[0], dv, grid);
    math::FourierTransform r0_ft(shape);
    r0_ft.set_real(rhs0);
    r0_ft.forward();

    arma::vec d1 = densities[0];
    math::FourierTransform work(shape);

    for (int iter = 0; iter < config.fp_max_iterations; ++iter) {
      auto [e1, f1] = compute({d1});
      arma::vec rhs1 = _internal::excess_rhs_real(d1, f1[0], dv, grid);
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

      if (max_dev < config.fp_tolerance) {
        break;
      }
    }

    auto [e_final, f_final] = compute({d1});
    return StepResult{
        .densities = {d1},
        .energy = e_final,
        .dt_used = dt,
    };
  }

  // Dirichlet boundary condition: resets marked cells to a fixed reservoir
  // density after each timestep. This couples the system to an infinite
  // reservoir at fixed chemical potential, allowing mass to flow in/out
  // through the boundary (Lutsko, J. Chem. Phys. 2008; Sci. Adv. 2019).

  [[nodiscard]] inline auto reservoir_boundary(const arma::uvec& mask, double reservoir_density) -> BoundaryCondition {
    return [mask, reservoir_density](std::vector<arma::vec>& densities) {
      for (auto& rho : densities) {
        rho.elem(arma::find(mask)).fill(reservoir_density);
      }
    };
  }

  // Generalised Dirichlet boundary: freezes marked cells to per-point
  // values from a reference density profile. Use this when the frozen
  // region has spatially varying density (e.g. the wall depletion zone
  // whose equilibrium profile is not uniform).

  [[nodiscard]] inline auto frozen_boundary(const arma::uvec& mask, const arma::vec& reference) -> BoundaryCondition {
    arma::uvec idx = arma::find(mask);
    arma::vec vals = reference.elem(idx);
    return [idx, vals](std::vector<arma::vec>& densities) {
      for (auto& rho : densities) {
        rho.elem(idx) = vals;
      }
    };
  }

  // Optional callback checked after each step; return true to stop early.
  using StopCondition = std::function<bool(int step, double time, double energy)>;

  // Configuration for a full DDFT simulation run.

  struct Simulation {
    StepConfig step;
    int n_steps{1000};
    int snapshot_interval{100};
    int log_interval{50};
    double energy_offset{0.0};
    BoundaryCondition boundary{};
    arma::uvec frozen_mask{};
    StopCondition stop_condition{};

    // Run a DDFT simulation using Lutsko's integrating-factor scheme
    // with implicit fixed-point iteration and adaptive timestep.
    // Boundary conditions are applied after each step via the boundary
    // callback (empty = periodic/canonical).

    [[nodiscard]] auto run(std::vector<arma::vec> densities, const Grid& grid, const ForceCallback& force_fn) const
        -> SimulationResult;
  };

  [[nodiscard]] inline auto
  Simulation::run(std::vector<arma::vec> densities, const Grid& grid, const ForceCallback& force_fn) const
      -> SimulationResult {
    double dv = grid.cell_volume();

    // Apply boundary condition to initial state.
    if (boundary) {
      boundary(densities);
    }

    double mass_initial = 0.0;
    for (const auto& rho : densities) {
      mass_initial += arma::accu(rho) * dv;
    }

    auto st = _internal::make_if_state(grid);
    StepConfig step_cfg = step;

    auto [e0, f0] = force_fn(densities);

    SimulationResult result;
    result.times.push_back(0.0);
    result.energies.push_back(e0);
    result.snapshots.push_back(Snapshot{
        .step = 0,
        .time = 0.0,
        .energy = e0,
        .densities = densities,
    });

    if (log_interval > 0) {
      std::cout
          << std::format("  {:>8s}  {:>14s}  {:>12s}  {:>18s}  {:>14s}\n", "step", "time", "dt", "Delta_E", "mass");
      std::cout << "  " << std::string(74, '-') << "\n";
      std::cout << std::format(
          "  {:>8d}  {:>14.6f}  {:>12.6e}  {:>18.6f}  {:>14.6f}\n",
          0,
          0.0,
          step_cfg.dt,
          e0 - energy_offset,
          mass_initial
      );
    }

    int successes = 0;
    double time = 0.0;

    for (int step = 1; step <= n_steps; ++step) {
      double dt_before = step_cfg.dt;
      auto step_result = integrating_factor_step(densities, grid, st, force_fn, step_cfg, boundary, frozen_mask);
      densities = std::move(step_result.densities);
      double e_step = step_result.energy;
      time += step_result.dt_used;

      if (boundary) {
        boundary(densities);
      }

      // Adaptive timestep: increase after 5 consecutive successes.
      if (step_result.dt_used < dt_before) {
        successes = 0;
      } else {
        ++successes;
      }
      if (successes >= 5 && step_cfg.dt < step_cfg.dt_max) {
        step_cfg.dt = std::min(2.0 * step_cfg.dt, step_cfg.dt_max);
        successes = 0;
      }

      if (log_interval > 0 && (step % log_interval == 0 || step == n_steps)) {
        double mass = 0.0;
        for (const auto& rho : densities) {
          mass += arma::accu(rho) * dv;
        }
        result.times.push_back(time);
        result.energies.push_back(e_step);
        std::cout << std::format(
            "  {:>8d}  {:>14.6f}  {:>12.6e}  {:>18.6f}  {:>14.6f}\n",
            step,
            time,
            step_cfg.dt,
            e_step - energy_offset,
            mass
        );
      }

      if (snapshot_interval > 0 && step % snapshot_interval == 0) {
        result.snapshots.push_back(Snapshot{
            .step = step,
            .time = time,
            .energy = e_step,
            .densities = densities,
        });
      }

      if (stop_condition && stop_condition(step, time, e_step)) {
        if (log_interval > 0) {
          std::cout << std::format("  Early stop at step {} (t={:.6f})\n", step, time);
        }
        break;
      }
    }

    result.densities = std::move(densities);
    result.mass_initial = mass_initial;
    result.mass_final = 0.0;
    for (const auto& rho : result.densities) {
      result.mass_final += arma::accu(rho) * dv;
    }

    return result;
  }

  // Post-processing types for dynamics analysis.

  struct PathwayPoint {
    double radius;
    double energy;
    double rho_center;
  };

  struct DynamicsAnalysis {
    std::vector<Slice1D> profiles;
    std::vector<PathwayPoint> pathway;
  };

  // Extract 1D line profiles and a (R_eff, Omega, rho_center) pathway
  // from a DDFT simulation result. The effective radius is computed from
  // the excess particle number relative to the background.

  [[nodiscard]] inline auto analyse_dynamics(
      const SimulationResult& sim,
      const Grid& grid,
      double background,
      double delta_rho,
      int profile_axis = 0
  ) -> DynamicsAnalysis {
    double dv = grid.cell_volume();
    DynamicsAnalysis result;
    for (const auto& snap : sim.snapshots) {
      auto prof = grid.line_slice(snap.densities[0], profile_axis);
      prof.time = snap.time;
      result.profiles.push_back(std::move(prof));
      double R = dft::effective_radius(snap.densities[0], background, delta_rho, dv);
      result.pathway.push_back({
          .radius = R,
          .energy = snap.energy,
          .rho_center = grid.center_value(snap.densities[0]),
      });
    }
    return result;
  }

} // namespace dft::algorithms::dynamics

#endif // DFT_ALGORITHMS_DYNAMICS_HPP
