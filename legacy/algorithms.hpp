// legacy_algorithms.hpp — Verbatim translations of the original algorithm code.
//
// Self-contained header with no dependencies on the new dft library.
// Every function, struct, and constant in this file corresponds to a
// specific routine in the original classicalDFT repository.
//
// Source files from lutsko/classicalDFT/:
//   src/DFT_FFT.cpp, include/DFT_FFT.h             — FFT3D (FFTW3 r2c/c2r)
//   src/DDFT.cpp                                    — DDFT integrating-factor
//   src/Minimizer.cpp, src/Species.cpp              — FIRE2 fixed-mass
//   src/Eigenvalues.cpp                             — eigenvalue via FIRE2

#ifndef DFT_LEGACY_ALGORITHMS_HPP
#define DFT_LEGACY_ALGORITHMS_HPP

#include <armadillo>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <numbers>
#include <span>
#include <vector>

namespace legacy::algorithms {

  using ForceCallback = std::function<std::pair<double, arma::vec>(const arma::vec&)>;

  // FFTW3 RAII wrapper for 3D r2c/c2r transforms.
  // Translation of the original DFT_FFT / DFT_Vec_Complex classes
  // (include/DFT_FFT.h, src/DFT_FFT.cpp).

  class FFT3D {
   public:
    FFT3D() = default;

    explicit FFT3D(long nx, long ny, long nz)
        : nx_(nx), ny_(ny), nz_(nz),
          n_real_(static_cast<std::size_t>(nx * ny * nz)),
          n_complex_(static_cast<std::size_t>(nx * ny * (nz / 2 + 1))) {
      real_ = static_cast<double*>(fftw_malloc(sizeof(double) * n_real_));
      complex_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n_complex_));
      int dims[3] = {static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz)};
      fwd_ = fftw_plan_dft_r2c_3d(dims[0], dims[1], dims[2], real_, complex_, FFTW_MEASURE);
      bwd_ = fftw_plan_dft_c2r_3d(dims[0], dims[1], dims[2], complex_, real_, FFTW_MEASURE);
    }

    ~FFT3D() {
      if (fwd_) fftw_destroy_plan(fwd_);
      if (bwd_) fftw_destroy_plan(bwd_);
      if (real_) fftw_free(real_);
      if (complex_) fftw_free(complex_);
    }

    FFT3D(FFT3D&& o) noexcept
        : nx_(o.nx_), ny_(o.ny_), nz_(o.nz_),
          n_real_(o.n_real_), n_complex_(o.n_complex_),
          real_(o.real_), complex_(o.complex_),
          fwd_(o.fwd_), bwd_(o.bwd_) {
      o.real_ = nullptr; o.complex_ = nullptr;
      o.fwd_ = nullptr; o.bwd_ = nullptr;
    }

    FFT3D& operator=(FFT3D&& o) noexcept {
      if (this != &o) {
        if (fwd_) fftw_destroy_plan(fwd_);
        if (bwd_) fftw_destroy_plan(bwd_);
        if (real_) fftw_free(real_);
        if (complex_) fftw_free(complex_);
        nx_ = o.nx_; ny_ = o.ny_; nz_ = o.nz_;
        n_real_ = o.n_real_; n_complex_ = o.n_complex_;
        real_ = o.real_; complex_ = o.complex_;
        fwd_ = o.fwd_; bwd_ = o.bwd_;
        o.real_ = nullptr; o.complex_ = nullptr;
        o.fwd_ = nullptr; o.bwd_ = nullptr;
      }
      return *this;
    }

    FFT3D(const FFT3D&) = delete;
    FFT3D& operator=(const FFT3D&) = delete;

    void set_real(const arma::vec& v) {
      std::copy(v.memptr(), v.memptr() + n_real_, real_);
    }

    void forward() { fftw_execute(fwd_); }
    void backward() { fftw_execute(bwd_); }

    auto real() -> std::span<double> { return {real_, n_real_}; }
    [[nodiscard]] auto real() const -> std::span<const double> { return {real_, n_real_}; }

    auto fourier() -> std::span<std::complex<double>> {
      return {reinterpret_cast<std::complex<double>*>(complex_), n_complex_};
    }
    [[nodiscard]] auto fourier() const -> std::span<const std::complex<double>> {
      return {reinterpret_cast<const std::complex<double>*>(complex_), n_complex_};
    }

   private:
    long nx_ = 0, ny_ = 0, nz_ = 0;
    std::size_t n_real_ = 0, n_complex_ = 0;
    double* real_ = nullptr;
    fftw_complex* complex_ = nullptr;
    fftw_plan fwd_ = nullptr;
    fftw_plan bwd_ = nullptr;
  };

  // Alias transform: x = sqrt(max(0, rho - DMIN)), rho = DMIN + x^2.
  // From Species.cpp: convert_to_alias / convert_from_alias.

  static constexpr double DMIN = 1e-99;

  inline auto rho_to_alias(const arma::vec& rho) -> arma::vec {
    arma::vec x(rho.n_elem);
    for (arma::uword i = 0; i < rho.n_elem; ++i)
      x(i) = std::sqrt(std::max(0.0, rho(i) - DMIN));
    return x;
  }

  inline auto alias_to_rho(const arma::vec& x) -> arma::vec {
    return DMIN + x % x;
  }

  // Boundary mask: 1 for face points, 0 for interior.
  // From Density class: is_boundary_point().

  inline auto boundary_mask_3d(long nx, long ny, long nz) -> arma::uvec {
    arma::uvec mask(static_cast<arma::uword>(nx * ny * nz), arma::fill::zeros);
    for (long ix = 0; ix < nx; ++ix)
      for (long iy = 0; iy < ny; ++iy)
        for (long iz = 0; iz < nz; ++iz) {
          if (ix == 0 || ix == nx - 1 ||
              iy == 0 || iy == ny - 1 ||
              iz == 0 || iz == nz - 1)
            mask(static_cast<arma::uword>(ix * ny * nz + iy * nz + iz)) = 1;
        }
    return mask;
  }

  // Homogeneous boundary: average forces at face points.
  // From Species.cpp: endForceCalculation() with homogeneousBoundary_.

  inline void homogeneous_boundary(arma::vec& forces, const arma::uvec& mask) {
    double sum = 0.0;
    arma::uword count = 0;
    for (arma::uword i = 0; i < forces.n_elem; ++i) {
      if (mask(i)) { sum += forces(i); count++; }
    }
    if (count > 0) {
      double avg = sum / static_cast<double>(count);
      for (arma::uword i = 0; i < forces.n_elem; ++i) {
        if (mask(i)) forces(i) = avg;
      }
    }
  }

  // Fixed-mass force evaluation.
  // From Minimizer.cpp: getDF_DX() + Species.cpp: convert_to_alias_deriv().

  struct ForceResult {
    double energy;
    arma::vec dF_alias;
    double lambda;
    arma::vec rho;
  };

  inline auto evaluate_fixed_mass(
      const ForceCallback& force_fn, const arma::vec& x,
      double target_mass, double dV, const arma::uvec& bdry_mask
  ) -> ForceResult {
    arma::vec rho = alias_to_rho(x);
    rho *= (target_mass / (arma::accu(rho) * dV));
    auto [F, dF] = force_fn(rho);
    homogeneous_boundary(dF, bdry_mask);
    double lambda = arma::dot(dF, rho) / target_mass;
    dF -= lambda * dV;
    return {F, 2.0 * x % dF, lambda, rho};
  }

  // FIRE2 minimizer for fixed-mass constraint.
  // From Minimizer.cpp: run().

  struct FireConfig {
    double dt{0.1};
    double dt_max{1.0};
    double alpha_start{0.01};
    double alpha_fac{0.99};
    double force_tolerance{1e-4};
    int max_steps{1000000};
    int log_interval{500};
    int N_delay{5};
    double f_dec{0.5};
    double f_inc{1.1};
  };

  struct FireResult {
    arma::vec density;
    double energy;
    double lambda;
    int iterations;
    bool converged;
  };

  [[nodiscard]] inline auto fire_minimize_fixed_mass(
      const ForceCallback& force_fn, arma::vec rho_init,
      double target_mass, double dV,
      long nx, long ny, long nz,
      const FireConfig& config
  ) -> FireResult {
    arma::uword n = rho_init.n_elem;
    double volume = static_cast<double>(n) * dV;
    arma::uvec bdry_mask = boundary_mask_3d(nx, ny, nz);
    rho_init *= (target_mass / (arma::accu(rho_init) * dV));
    arma::vec x = rho_to_alias(rho_init);
    arma::vec v(n, arma::fill::zeros);

    auto fr = evaluate_fixed_mass(force_fn, x, target_mass, dV, bdry_mask);
    double F = fr.energy;
    double F_old = F;
    arma::vec dF = fr.dF_alias;

    double dt = config.dt;
    double alpha = config.alpha_start;
    int N_P_positive = 0;
    double vv = 1.0;
    bool converged = false;
    int it = 0;

    for (it = 1; it <= config.max_steps; ++it) {
      double P = -arma::dot(v, dF);
      if (P > 0 || it == 1) {
        N_P_positive++;
        if (N_P_positive > config.N_delay) {
          dt = std::min(dt * config.f_inc, config.dt_max);
          alpha *= config.alpha_fac;
        }
      } else {
        N_P_positive = 0;
        x -= 0.5 * dt * v;
        v.zeros();
        if (it >= config.N_delay) { dt *= config.f_dec; alpha = config.alpha_start; }
      }

      v -= dt * dF;
      double vnorm = arma::norm(v);
      double fnorm = arma::norm(dF);
      v *= (1.0 - alpha);
      if (fnorm > 0) v -= (alpha * vnorm / fnorm) * dF;
      x += dt * v;

      F_old = F;
      fr = evaluate_fixed_mass(force_fn, x, target_mass, dV, bdry_mask);
      F = fr.energy;
      dF = fr.dF_alias;

      vv = vv * 0.9 + 0.1 * (std::abs(F - F_old) / dt);
      double monitor = std::abs(vv / volume);

      if (config.log_interval > 0 &&
          (it % config.log_interval == 0 || monitor < config.force_tolerance)) {
        std::cout << std::format("  {:>8d}  {:>18.6f}  {:>14.6e}  {:>12.4e}\n",
                                 it, F, monitor, dt);
      }
      if (monitor < config.force_tolerance) { converged = true; break; }
    }
    return {fr.rho, F, fr.lambda, it, converged};
  }

  // Hessian-vector product: H*v = (F(rho+eps*v) - F(rho)) / eps.
  // From Eigenvalues.cpp: calculate_eigenvalue().

  inline auto hessian_vec(
      const ForceCallback& force_fn,
      const arma::vec& rho, const arma::vec& forces,
      const arma::vec& v, double eps
  ) -> arma::vec {
    auto [_, f_shifted] = force_fn(rho + eps * v);
    return (f_shifted - forces) / eps;
  }

  // Eigenvalue via FIRE2 on Rayleigh quotient.
  // From Eigenvalues.cpp: calculate_eigenvector().

  struct EigenConfig {
    double tolerance{1e-4};
    int max_iterations{5000};
    double hessian_eps{1e-6};
    int log_interval{50};
  };

  struct EigenResult {
    arma::vec eigenvector;
    double eigenvalue{0.0};
    int iterations{0};
    bool converged{false};
  };

  [[nodiscard]] inline auto eigenvalue_fire2(
      const ForceCallback& force_fn,
      const arma::vec& rho,
      const arma::uvec& bdry_mask,
      const EigenConfig& config
  ) -> EigenResult {
    auto [omega, forces] = force_fn(rho);
    arma::uword n = rho.n_elem;

    auto Hv = [&](const arma::vec& vec) -> arma::vec {
      arma::vec hv = hessian_vec(force_fn, rho, forces, vec, config.hessian_eps);
      for (arma::uword i = 0; i < n; ++i) {
        if (bdry_mask(i)) hv(i) = 0.0;
      }
      return hv;
    };

    arma::vec ev = arma::randn(n);
    for (arma::uword i = 0; i < n; ++i) {
      if (bdry_mask(i)) ev(i) = 0.0;
    }
    ev /= arma::norm(ev);

    double alpha_start = 1.0;
    double alpha = alpha_start;
    double dt = 0.01;
    double dt_max = 1.0;
    double f_inc = 1.1;
    double f_alf = 0.9;
    double f_dec = 0.1;
    int N_delay = 5;
    int N_pos = 0;
    int N_neg = 0;
    bool initial_delay = true;

    arma::vec vel(n, arma::fill::zeros);
    arma::vec ev_old = ev;

    auto compute_obj_grad = [&](const arma::vec& v)
        -> std::tuple<double, arma::vec, double> {
      arma::vec hv = Hv(v);
      double x2 = arma::dot(v, v);
      double xhx = arma::dot(v, hv);
      double R = xhx / x2;
      arma::vec grad = 2.0 * (hv - R * v) / x2;
      grad += 4.0 * (x2 - 1.0) * v;
      double f = R + (x2 - 1.0) * (x2 - 1.0);
      return {f, grad, R};
    };

    auto [f, df, eigenval] = compute_obj_grad(ev);

    if (config.log_interval > 0) {
      arma::vec res = Hv(ev) - eigenval * ev;
      double res_norm = arma::norm(res) / arma::norm(ev) / (1.0 + std::abs(eigenval));
      std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}\n", "iter", "eigenvalue", "residual");
      std::cout << "  " << std::string(40, '-') << "\n";
      std::cout << std::format("  {:>6d}  {:>14.6e}  {:>14.6e}\n", 0, eigenval, res_norm);
    }

    bool converged = false;
    int it = 0;
    for (it = 1; it <= config.max_iterations; ++it) {
      double P = -arma::dot(vel, df);
      double vnorm = arma::norm(vel);
      double fnorm = arma::norm(df);
      double P_norm = (vnorm > 0 && fnorm > 0) ? P / (vnorm * fnorm) : 0.0;

      if (P >= 0) {
        N_pos++; N_neg = 0;
        if (N_pos > N_delay && P_norm > 0.999) {
          dt = std::min(dt * f_inc, dt_max);
          alpha = std::max(alpha * f_alf, 0.0);
        } else if (N_pos > N_delay && P_norm <= 0.5) {
          dt = std::max(dt / f_inc, 0.0);
          alpha = std::min(alpha / f_alf, alpha_start);
        }
      } else {
        N_neg++; N_pos = 0;
        if (N_neg > 20) break;
        if (!initial_delay || it >= N_delay) {
          dt *= f_dec;
          alpha = alpha_start;
        }
        ev = ev_old;
        vel.zeros();
        auto [f2, df2, e2] = compute_obj_grad(ev);
        f = f2; df = df2; eigenval = e2;
      }

      ev_old = ev;

      vel -= dt * df;
      vnorm = arma::norm(vel);
      fnorm = arma::norm(df);
      vel *= (1.0 - alpha);
      if (fnorm > 0) vel -= (alpha * vnorm / fnorm) * df;
      ev += dt * vel;

      for (arma::uword i = 0; i < n; ++i) {
        if (bdry_mask(i)) ev(i) = 0.0;
      }

      auto [f_new, df_new, eigenval_new] = compute_obj_grad(ev);
      f = f_new; df = df_new; eigenval = eigenval_new;

      arma::vec res = Hv(ev) - eigenval * ev;
      double res_norm = arma::norm(res) / arma::norm(ev) / (1.0 + std::abs(eigenval));

      if (config.log_interval > 0 &&
          (it % config.log_interval == 0 || res_norm < config.tolerance)) {
        std::cout << std::format("  {:>6d}  {:>14.6e}  {:>14.6e}\n", it, eigenval, res_norm);
      }

      if (!initial_delay || it >= N_delay) {
        if (res_norm < config.tolerance) {
          converged = true;
          break;
        }
      }
    }

    ev /= arma::norm(ev);
    arma::vec hev = Hv(ev);
    eigenval = arma::dot(ev, hev);

    return {ev, eigenval, it, converged};
  }

  // DDFT integrating-factor scheme.
  // From DDFT.cpp.

  struct DdftState {
    long nx, ny, nz, nz_half;
    double dx;
    arma::vec lam_x, lam_y, lam_z;
    arma::vec fx, fy, fz;
    double dt = 0.0;
  };

  inline auto make_ddft_state(long nx, long ny, long nz, double dx) -> DdftState {
    long nz_half = nz / 2 + 1;
    double Dx = 1.0 / (dx * dx);

    arma::vec lx(static_cast<arma::uword>(nx));
    for (long i = 0; i < nx; ++i) {
      double k = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(nx);
      lx(static_cast<arma::uword>(i)) = 2.0 * Dx * (std::cos(k) - 1.0);
    }
    arma::vec ly(static_cast<arma::uword>(ny));
    for (long i = 0; i < ny; ++i) {
      double k = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(ny);
      ly(static_cast<arma::uword>(i)) = 2.0 * Dx * (std::cos(k) - 1.0);
    }
    arma::vec lz(static_cast<arma::uword>(nz_half));
    for (long i = 0; i < nz_half; ++i) {
      double k = 2.0 * std::numbers::pi * static_cast<double>(i) / static_cast<double>(nz);
      lz(static_cast<arma::uword>(i)) = 2.0 * Dx * (std::cos(k) - 1.0);
    }
    return {nx, ny, nz, nz_half, dx, lx, ly, lz, {}, {}, {}, 0.0};
  }

  inline void update_timestep(DdftState& st, double dt) {
    if (std::abs(st.dt - dt) < 1e-30) return;
    st.dt = dt;
    st.fx = arma::exp(st.lam_x * dt);
    st.fy = arma::exp(st.lam_y * dt);
    st.fz = arma::exp(st.lam_z * dt);
  }

  // g_dot_x: computes div(rho * grad(x)).
  // From DDFT.cpp: g_dot_x(), FWD1 discretization.

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
          gx(p) = D * ((rho(px)+d0)*(x(px)-x0) - (d0+rho(mx))*(x0-x(mx)))
                 + D * ((rho(py)+d0)*(x(py)-x0) - (d0+rho(my))*(x0-x(my)))
                 + D * ((rho(pz)+d0)*(x(pz)-x0) - (d0+rho(mz))*(x0-x(mz)));
        }
      }
    }
  }

  // subtract_ideal_gas: RHS -= Laplacian(rho).
  // From DDFT.cpp: subtract_ideal_gas().

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
          rhs(p) -= D * (rho(px) + rho(mx) - 2.0*d0)
                  + D * (rho(py) + rho(my) - 2.0*d0)
                  + D * (rho(pz) + rho(mz) - 2.0*d0);
        }
      }
    }
  }

  // Compute the excess (nonlinear) RHS for the integrating-factor scheme.
  // From DDFT.cpp: calculate_excess_RHS().

  inline void compute_excess_rhs(
      const arma::vec& rho, const arma::vec& forces, double dv,
      long nx, long ny, long nz, double dx,
      FFT3D& rhs_ft
  ) {
    auto n = static_cast<arma::uword>(nx * ny * nz);
    arma::vec rhs(n);
    g_dot_x(rho, forces, rhs, nx, ny, nz, dx);
    rhs /= dv;
    subtract_ideal_gas(rho, rhs, nx, ny, nz, dx);
    rhs_ft.set_real(rhs);
    rhs_ft.forward();
  }

  // Apply the diffusion propagator in Fourier space.
  // From DDFT.cpp: apply_diffusion_propagator().

  inline auto apply_propagator(
      const DdftState& st,
      const FFT3D& rho0_ft,
      const FFT3D& rhs0_ft,
      const FFT3D& rhs1_ft,
      arma::vec& d1
  ) -> double {
    long nx = st.nx, ny = st.ny;
    long nz_half = st.nz_half;
    double dt = st.dt;
    long n_total = nx * ny * st.nz;
    FFT3D work(nx, ny, st.nz);

    auto rho0_k = rho0_ft.fourier();
    auto r0_k = rhs0_ft.fourier();
    auto r1_k = rhs1_ft.fourier();
    auto cwork = work.fourier();

    long pos = 0;
    for (long ix = 0; ix < nx; ++ix) {
      for (long iy = 0; iy < ny; ++iy) {
        for (long iz = 0; iz < nz_half; ++iz) {
          double Lambda = st.lam_x(static_cast<arma::uword>(ix))
                        + st.lam_y(static_cast<arma::uword>(iy))
                        + st.lam_z(static_cast<arma::uword>(iz));
          double exp_dt = st.fx(static_cast<arma::uword>(ix))
                        * st.fy(static_cast<arma::uword>(iy))
                        * st.fz(static_cast<arma::uword>(iz));
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

  // Full DDFT step with adaptive restart.
  // From DDFT.cpp: step().

  struct DdftResult {
    arma::vec density;
    double energy;
    double dt_used;
  };

  [[nodiscard]] inline auto ddft_step(
      const arma::vec& rho0, DdftState& st,
      const ForceCallback& force_fn, double dv,
      double fp_tolerance, int fp_max_iterations,
      double& dt, double /*dt_max*/
  ) -> DdftResult {
    long nx = st.nx, ny = st.ny, nz = st.nz;
    double dx = st.dx;

    auto [energy, forces] = force_fn(rho0);

    FFT3D rhs0(nx, ny, nz);
    compute_excess_rhs(rho0, forces, dv, nx, ny, nz, dx, rhs0);

    FFT3D rho0_ft(nx, ny, nz);
    rho0_ft.set_real(rho0);
    rho0_ft.forward();

    arma::vec d1 = rho0;
    bool restart;

    do {
      restart = false;
      update_timestep(st, dt);
      d1 = rho0;

      FFT3D rhs1(nx, ny, nz);
      double deviation = 1e30;

      for (int i = 0; i < fp_max_iterations && deviation > fp_tolerance && !restart; ++i) {
        auto [e1, f1] = force_fn(d1);
        compute_excess_rhs(d1, f1, dv, nx, ny, nz, dx, rhs1);

        double old_deviation = deviation;
        deviation = apply_propagator(st, rho0_ft, rhs0, rhs1, d1);

        if (d1.min() < 0 || (i > 0 && old_deviation < deviation)) {
          restart = true;
        }
      }

      if (restart || deviation > fp_tolerance) {
        dt /= 10.0;
        d1 = rho0;
        restart = true;
      }
    } while (restart);

    auto [e_final, f_final] = force_fn(d1);
    return {d1, e_final, dt};
  }

}  // namespace legacy::algorithms

#endif  // DFT_LEGACY_ALGORITHMS_HPP
