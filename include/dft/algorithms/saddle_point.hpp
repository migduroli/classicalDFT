#ifndef DFT_ALGORITHMS_SADDLE_POINT_HPP
#define DFT_ALGORITHMS_SADDLE_POINT_HPP

#include "dft/algorithms/fire.hpp"
#include "dft/algorithms/minimization.hpp"
#include "dft/fields.hpp"
#include "dft/functionals/bulk/coexistence.hpp"
#include "dft/functionals/functional.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/grid.hpp"
#include "dft/init.hpp"
#include "dft/math/spline.hpp"
#include "dft/physics/walls.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <chrono>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <print>
#include <stdexcept>
#include <vector>

namespace dft::algorithms::saddle_point {

  // Force function signature: rho -> (Omega, dOmega/drho * dV).

  using ForceFunction = std::function<std::pair<double, arma::vec>(const arma::vec&)>;

  namespace _internal {

    // Finite-difference Hessian-vector product:
    //   H * v = (F(rho + eps*v) - F(rho)) / eps
    // Requires one extra force evaluation per call.

    [[nodiscard]] inline auto hessian_times_vector(
        const ForceFunction& force_fn,
        const arma::vec& rho,
        const arma::vec& forces_at_rho,
        const arma::vec& v,
        double eps = 1e-6
    ) -> arma::vec {
      auto [_, forces_shifted] = force_fn(rho + eps * v);
      return (forces_shifted - forces_at_rho) / eps;
    }

    struct EigenvalueConfig {
      double tolerance{1e-4};
      int max_iterations{300};
      double hessian_eps{1e-6};
      int log_interval{0};
    };

    // Find the smallest eigenvalue and its eigenvector of the Hessian
    // using the LOBPCG algorithm.

    [[nodiscard]] inline auto lobpcg(
        const ForceFunction& force_fn,
        const arma::vec& rho,
        const EigenvalueConfig& config = {},
        const arma::vec& initial_guess = {}
    ) -> std::pair<arma::vec, double> {
      auto [omega, forces] = force_fn(rho);
      arma::uword n = rho.n_elem;

      arma::vec v = initial_guess.n_elem == n ? initial_guess : arma::randn(n);
      v /= arma::norm(v);

      if (config.log_interval > 0) {
        std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}\n", "iter", "eigenvalue", "residual");
        std::cout << "  " << std::string(40, '-') << "\n";
      }

      arma::vec hv = hessian_times_vector(force_fn, rho, forces, v, config.hessian_eps);
      double lambda = arma::dot(v, hv);
      arma::vec p_raw;
      arma::vec hp_raw;

      for (int iter = 0; iter < config.max_iterations; ++iter) {
        arma::vec residual = hv - lambda * v;
        double res_norm = arma::norm(residual);

        if (config.log_interval > 0 && (iter % config.log_interval == 0 || res_norm < config.tolerance)) {
          std::cout << std::format("  {:>6d}  {:>14.6e}  {:>14.6e}\n", iter, lambda, res_norm);
        }

        if (res_norm < config.tolerance) {
          break;
        }

        arma::vec q0 = v;
        arma::vec q1 = residual - arma::dot(residual, q0) * q0;
        double q1n = arma::norm(q1);
        if (q1n < 1e-30) {
          break;
        }
        q1 /= q1n;

        arma::vec hq0 = hv;
        arma::vec hq1 = hessian_times_vector(force_fn, rho, forces, q1, config.hessian_eps);

        int dim = 2;
        arma::vec q2;
        arma::vec hq2;

        if (p_raw.n_elem == n) {
          q2 = p_raw - arma::dot(p_raw, q0) * q0 - arma::dot(p_raw, q1) * q1;
          double q2n = arma::norm(q2);
          if (q2n > 1e-10) {
            q2 /= q2n;
            hq2 = hessian_times_vector(force_fn, rho, forces, q2, config.hessian_eps);
            dim = 3;
          }
        }

        arma::mat H_proj(dim, dim);
        H_proj(0, 0) = arma::dot(q0, hq0);
        H_proj(0, 1) = arma::dot(q0, hq1);
        H_proj(1, 0) = H_proj(0, 1);
        H_proj(1, 1) = arma::dot(q1, hq1);

        if (dim == 3) {
          H_proj(0, 2) = arma::dot(q0, hq2);
          H_proj(1, 2) = arma::dot(q1, hq2);
          H_proj(2, 0) = H_proj(0, 2);
          H_proj(2, 1) = H_proj(1, 2);
          H_proj(2, 2) = arma::dot(q2, hq2);
        }

        arma::vec eigvals;
        arma::mat eigvecs;
        arma::eig_sym(eigvals, eigvecs, H_proj);
        arma::vec c = eigvecs.col(0);

        arma::vec v_new = c(0) * q0 + c(1) * q1;
        arma::vec hv_new = c(0) * hq0 + c(1) * hq1;
        if (dim == 3) {
          v_new += c(2) * q2;
          hv_new += c(2) * hq2;
        }

        p_raw = v_new - v;
        hp_raw = hv_new - hv;

        v = v_new / arma::norm(v_new);
        hv = hessian_times_vector(force_fn, rho, forces, v, config.hessian_eps);
        lambda = arma::dot(v, hv);
      }

      return {v, lambda};
    }

  } // namespace _internal

  // Eigenvalue result (public, needed by nucleation example).

  struct EigenvalueResult {
    arma::vec eigenvector;
    double eigenvalue{0.0};
    int iterations{0};
    bool converged{false};
  };

  struct EigenvalueSolver {
    double tolerance{1e-4};
    int max_iterations{300};
    double hessian_eps{1e-6};
    int log_interval{0};
    arma::uvec boundary_mask{};

    // Find the smallest eigenvalue and its eigenvector of the Hessian
    // d^2 Omega / (drho_i drho_j) using LOBPCG.
    //
    // When boundary_mask is set, the eigenvector is projected to zero
    // at masked indices each iteration so the effective operator is
    // P H P (symmetric) rather than P H (non-symmetric).

    [[nodiscard]] auto solve(const ForceFunction& force_fn, const arma::vec& rho, const arma::vec& initial_guess = {})
        const -> EigenvalueResult;
  };

  [[nodiscard]] inline auto
  EigenvalueSolver::solve(const ForceFunction& force_fn, const arma::vec& rho, const arma::vec& initial_guess) const
      -> EigenvalueResult {
    _internal::EigenvalueConfig internal_cfg{
        .tolerance = tolerance,
        .max_iterations = max_iterations,
        .hessian_eps = hessian_eps,
        .log_interval = log_interval,
    };

    auto [omega, forces] = force_fn(rho);
    arma::uword n = rho.n_elem;

    arma::vec v = initial_guess.n_elem == n ? initial_guess : arma::randn(n);
    if (boundary_mask.n_elem == n) {
      v.elem(arma::find(boundary_mask)).zeros();
    }
    v /= arma::norm(v);

    if (log_interval > 0) {
      std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}\n", "iter", "eigenvalue", "residual");
      std::cout << "  " << std::string(40, '-') << "\n";
    }

    arma::vec hv = _internal::hessian_times_vector(force_fn, rho, forces, v, hessian_eps);
    double lambda = arma::dot(v, hv);
    arma::vec p_raw;

    bool converged = false;
    int iter = 0;

    for (; iter < max_iterations; ++iter) {
      arma::vec residual = hv - lambda * v;
      double res_norm = arma::norm(residual);

      if (log_interval > 0 && (iter % log_interval == 0 || res_norm < tolerance)) {
        std::cout << std::format("  {:>6d}  {:>14.6e}  {:>14.6e}\n", iter, lambda, res_norm);
      }

      if (res_norm < tolerance) {
        converged = true;
        break;
      }

      arma::vec q0 = v;
      arma::vec q1 = residual - arma::dot(residual, q0) * q0;
      double q1n = arma::norm(q1);
      if (q1n < 1e-30) {
        break;
      }
      q1 /= q1n;

      arma::vec hq0 = hv;
      arma::vec hq1 = _internal::hessian_times_vector(force_fn, rho, forces, q1, hessian_eps);

      int dim = 2;
      arma::vec q2;
      arma::vec hq2;

      if (p_raw.n_elem == n) {
        q2 = p_raw - arma::dot(p_raw, q0) * q0 - arma::dot(p_raw, q1) * q1;
        double q2n = arma::norm(q2);
        if (q2n > 1e-10) {
          q2 /= q2n;
          hq2 = _internal::hessian_times_vector(force_fn, rho, forces, q2, hessian_eps);
          dim = 3;
        }
      }

      arma::mat H_proj(dim, dim);
      H_proj(0, 0) = arma::dot(q0, hq0);
      H_proj(0, 1) = arma::dot(q0, hq1);
      H_proj(1, 0) = H_proj(0, 1);
      H_proj(1, 1) = arma::dot(q1, hq1);

      if (dim == 3) {
        H_proj(0, 2) = arma::dot(q0, hq2);
        H_proj(1, 2) = arma::dot(q1, hq2);
        H_proj(2, 0) = H_proj(0, 2);
        H_proj(2, 1) = H_proj(1, 2);
        H_proj(2, 2) = arma::dot(q2, hq2);
      }

      arma::vec eigvals;
      arma::mat eigvecs;
      arma::eig_sym(eigvals, eigvecs, H_proj);
      arma::vec c = eigvecs.col(0);

      arma::vec v_new = c(0) * q0 + c(1) * q1;
      arma::vec hv_new = c(0) * hq0 + c(1) * hq1;
      if (dim == 3) {
        v_new += c(2) * q2;
        hv_new += c(2) * hq2;
      }

      p_raw = v_new - v;

      v = v_new / arma::norm(v_new);
      if (boundary_mask.n_elem == n) {
        v.elem(arma::find(boundary_mask)).zeros();
        v /= arma::norm(v);
      }
      hv = _internal::hessian_times_vector(force_fn, rho, forces, v, hessian_eps);
      lambda = arma::dot(v, hv);
    }

    return EigenvalueResult{
        .eigenvector = v,
        .eigenvalue = lambda,
        .iterations = iter,
        .converged = converged,
    };
  }

  namespace _internal {

    // Wrap a force function with a Householder reflection that
    // reverses the force component along one direction per species:
    //   f' = f - 2 (f . d) d
    //
    // This turns a saddle point into an effective minimum. The
    // direction d must be the actual unstable eigenvector of
    // the Hessian d^2 Omega / drho^2.

    using MultiSpeciesForceFunction =
        std::function<std::pair<double, std::vector<arma::vec>>(const std::vector<arma::vec>&)>;

    [[nodiscard]] inline auto reversed_forces(MultiSpeciesForceFunction compute, std::vector<arma::vec> directions)
        -> MultiSpeciesForceFunction {
      for (auto& d : directions) {
        double n = arma::norm(d);
        if (n > 0.0) {
          d /= n;
        }
      }

      return [compute = std::move(compute), directions = std::move(directions)](const std::vector<arma::vec>& densities
             ) -> std::pair<double, std::vector<arma::vec>> {
        auto [energy, forces] = compute(densities);
        for (std::size_t s = 0; s < forces.size() && s < directions.size(); ++s) {
          double proj = arma::dot(directions[s], forces[s]);
          forces[s] -= 2.0 * proj * directions[s];
        }
        return {energy, std::move(forces)};
      };
    }

  } // namespace _internal

  struct Result {
    arma::vec density;
    arma::vec eigenvector;
    double grand_potential{0.0};
    double eigenvalue{0.0};
    int outer_iterations{0};
    bool converged{false};
  };

  // Saddle-point search configuration.

  struct Search {
    fire::Fire fire{
        .dt = 1e-3,
        .dt_max = 1.0,
        .alpha_start = 0.01,
        .f_alpha = 1.0,
        .force_tolerance = 1e-4,
        .max_steps = 100000,
    };
    EigenvalueSolver eigenvalue{
        .tolerance = 1e-4,
        .max_iterations = 300,
        .hessian_eps = 1e-6,
    };
    minimization::Parametrization param = minimization::Unbounded{.rho_min = 1e-18};
    double omega_tolerance{1e-5};
    double tolerance_start{1e-3};
    double tolerance_factor{10.0};
    int max_outer_iterations{20};
    int eigenvalue_clear_count{2};
    int log_interval{100};

    [[nodiscard]] auto find(
        const physics::Model& model,
        const functionals::Weights& weights,
        const arma::vec& initial_density,
        double chemical_potential,
        double target_mass
    ) const -> Result;
  };

  // Find the saddle point (critical cluster) of the grand potential
  // using eigenvector following (after Lutsko).
  //
  // The mass is kept FIXED throughout. The input density (from a
  // fixed-mass minimisation) already sits at the right mass for the
  // critical cluster. The boundary forces are zeroed (fixed boundary).
  //
  // Per outer iteration:
  //   1. Compute smallest eigenvalue/eigenvector of the Hessian
  //   2. Run FIRE2 at fixed mass with direction projection
  //   3. Check |Omega_new - Omega_old| < tol

  [[nodiscard]] inline auto Search::find(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      double target_mass
  ) const -> Result {
    double dv = model.grid.cell_volume();
    arma::uvec bdry = model.grid.boundary_mask();

    auto eig_force_fn = [&](const arma::vec& rho) -> std::pair<double, arma::vec> {
      auto state = init::from_profile(model, rho);
      state.species[0].chemical_potential = chemical_potential;
      auto result = functionals::total(model, state, weights);
      arma::vec grad = result.forces[0];
      grad = fixed_boundary(grad, bdry);
      return {result.grand_potential, grad};
    };

    arma::vec rho = initial_density;
    arma::vec eigvec;
    double omega = 0.0;
    double eigenvalue_val = 0.0;
    bool converged = false;
    int outer = 0;
    double ftol = tolerance_start;

    std::cout << "\n=== Eigenvector following: saddle-point search ===\n";
    std::cout << "  target_mass = " << target_mass << "\n\n";

    for (; outer < max_outer_iterations; ++outer) {
      double omega_old = omega;

      std::cout << std::format("  --- outer iteration {} ---\n", outer);

      // Step 1: compute smallest eigenvalue/eigenvector.

      std::cout << "  Computing smallest eigenvalue...\n";

      auto eig_solver = eigenvalue;
      eig_solver.boundary_mask = bdry;
      auto eig_result = eig_solver.solve(
          eig_force_fn,
          rho,
          (outer >= eigenvalue_clear_count && eigvec.n_elem == rho.n_elem) ? eigvec : arma::vec{}
      );

      eigvec = eig_result.eigenvector;
      eigenvalue_val = eig_result.eigenvalue;

      std::cout << std::format(
          "  eigenvalue = {:.6e}  converged = {}  iters = {}\n",
          eigenvalue_val,
          eig_result.converged,
          eig_result.iterations
      );

      // Step 2: FIRE2 at fixed mass + fixed boundary + direction projection.

      arma::vec x_current = minimization::_internal::from_density(rho, param);
      arma::vec dir_alias = eigvec;
      dir_alias = minimization::_internal::transform_force(dir_alias, x_current, param);
      double dn = arma::norm(dir_alias);
      if (dn > 1e-30) {
        dir_alias /= dn;
      }

      double rho_boundary_avg = 0.0;
      arma::uword n_boundary = 0;
      for (arma::uword i = 0; i < rho.n_elem; ++i) {
        if (bdry(i)) {
          rho_boundary_avg += rho(i);
          n_boundary++;
        }
      }
      if (n_boundary > 0) {
        rho_boundary_avg /= static_cast<double>(n_boundary);
      }
      double m_boundary = rho_boundary_avg * dv * static_cast<double>(n_boundary);

      auto compute = [&](const std::vector<arma::vec>& x_param) -> std::pair<double, std::vector<arma::vec>> {
        arma::vec rho_local = minimization::_internal::to_density(x_param[0], param);

        double m_interior = 0.0;
        for (arma::uword i = 0; i < rho_local.n_elem; ++i) {
          if (!bdry(i)) {
            m_interior += rho_local(i);
          }
        }
        m_interior *= dv;
        double m_target_interior = target_mass - m_boundary;
        if (m_interior > 0.0 && m_target_interior > 0.0) {
          double scale = m_target_interior / m_interior;
          for (arma::uword i = 0; i < rho_local.n_elem; ++i) {
            if (!bdry(i)) {
              rho_local(i) *= scale;
            }
          }
        }

        auto state = init::from_profile(model, rho_local);
        state.species[0].chemical_potential = chemical_potential;
        auto result = functionals::total(model, state, weights);
        arma::vec grad = result.forces[0];

        grad = fixed_boundary(grad, bdry);

        double mu_lagrange = 0.0;
        for (arma::uword i = 0; i < rho_local.n_elem; ++i) {
          if (!bdry(i)) {
            mu_lagrange += grad(i) * rho_local(i);
          }
        }
        if (m_target_interior > 0.0) {
          mu_lagrange /= m_target_interior;
        }
        for (arma::uword i = 0; i < grad.n_elem; ++i) {
          if (!bdry(i)) {
            grad(i) -= mu_lagrange * dv;
          } else {
            grad(i) = 0.0;
          }
        }

        arma::vec f_alias = minimization::_internal::transform_force(grad, x_param[0], param);
        f_alias -= arma::dot(dir_alias, f_alias) * dir_alias;

        return {result.grand_potential, {-f_alias}};
      };

      std::cout << "  Running FIRE2 (fixed mass + projected)...\n";

      auto fire_cfg = fire;
      fire_cfg.force_tolerance = std::max(ftol, fire.force_tolerance);
      ftol /= tolerance_factor;

      auto fire_state = fire_cfg.initialize({minimization::_internal::from_density(rho, param)}, compute);
      auto [e0, forces0] = compute(fire_state.x);

      if (log_interval > 0) {
        std::cout << std::format("    {:>6s}  {:>14s}  {:>14s}  {:>10s}\n", "iter", "Omega", "rms_force", "dt");
        std::cout << "    " << std::string(50, '-') << "\n";
        std::cout << std::format(
            "    {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
            0,
            fire_state.energy,
            fire_state.rms_force,
            fire_state.dt
        );
      }

      for (int i = 0; i < fire_cfg.max_steps && !fire_state.converged; ++i) {
        auto [ns, nf] = fire_cfg.step(std::move(fire_state), forces0, compute);
        fire_state = std::move(ns);
        forces0 = std::move(nf);

        if (log_interval > 0 && ((i + 1) % log_interval == 0 || fire_state.converged)) {
          std::cout << std::format(
              "    {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
              i + 1,
              fire_state.energy,
              fire_state.rms_force,
              fire_state.dt
          );
        }
      }

      // Extract final density.
      rho = minimization::_internal::to_density(fire_state.x[0], param);
      double fm = 0.0;
      for (arma::uword i = 0; i < rho.n_elem; ++i) {
        if (!bdry(i)) {
          fm += rho(i);
        }
      }
      fm *= dv;
      double m_target_interior = target_mass - m_boundary;
      if (fm > 0.0 && m_target_interior > 0.0) {
        double scale = m_target_interior / fm;
        for (arma::uword i = 0; i < rho.n_elem; ++i) {
          if (!bdry(i)) {
            rho(i) *= scale;
          }
        }
      }

      auto s_final = init::from_profile(model, rho);
      s_final.species[0].chemical_potential = chemical_potential;
      auto r_final = functionals::total(model, s_final, weights);
      omega = r_final.grand_potential;

      arma::vec grad_final = r_final.forces[0];
      grad_final = fixed_boundary(grad_final, bdry);
      double gamma = arma::dot(grad_final, eigvec);

      std::cout << std::format(
          "  Omega = {:.8f}  (delta = {:.2e})  gamma = {:.2e}\n\n",
          omega,
          std::abs(omega - omega_old),
          gamma
      );

      if (outer > 0 && std::abs(omega - omega_old) < omega_tolerance) {
        converged = true;
        break;
      }

      x_current = minimization::_internal::from_density(rho, param);
    }

    std::cout << std::format(
        "  Eigenvector following {}  ({} outer iterations)\n",
        converged ? "CONVERGED" : "NOT CONVERGED",
        outer + 1
    );

    return Result{
        .density = rho,
        .eigenvector = eigvec,
        .grand_potential = omega,
        .eigenvalue = eigenvalue_val,
        .outer_iterations = outer + 1,
        .converged = converged,
    };
  }

  // Orient an eigenvector so that the positive direction corresponds to
  // growth (increasing total mass). Generic for any DFT saddle-point.

  [[nodiscard]] inline auto orient_eigenvector(const arma::vec& ev, double cell_volume) -> arma::vec {
    double delta_mass = arma::accu(ev) * cell_volume;
    return (delta_mass < 0.0) ? -ev : arma::vec(ev);
  }

  // Create a density profile perturbed along the eigenvector direction.
  // sign > 0 for growth, sign < 0 for dissolution.

  [[nodiscard]] inline auto
  eigenvector_perturbation(const arma::vec& rho, const arma::vec& ev, double scale, double sign) -> arma::vec {
    return arma::clamp(rho + sign * scale * ev, 1e-18, arma::datum::inf);
  }

  // Result of a constrained saddle-point search: the converged density
  // together with physical analysis of the critical cluster.

  struct ConstrainedResult {
    minimization::Result minimization;
    double background{0.0};
    double mu_background{0.0};
    double omega_cluster{0.0};
    double omega_background{0.0};
    double barrier{0.0};
    double effective_radius{0.0};
    double rho_vapor_meta{0.0};
    double rho_liquid_meta{0.0};
    arma::uvec boundary_mask;
  };

  // Exponential-backoff retry helper for FIRE-based solves. Halves dt
  // on each failure up to max_retries.

  template <typename SolveFn>
  [[nodiscard]] inline auto
  retry_with_backoff(std::string_view label, minimization::Minimizer solver, int max_retries, SolveFn&& solve)
      -> minimization::Result {
    for (int attempt = 0;; ++attempt) {
      try {
        return solve(solver);
      } catch (const std::runtime_error&) {
        if (attempt >= max_retries)
          throw;
        solver.fire.dt = std::max(solver.fire.dt_min, 0.5 * solver.fire.dt);
        solver.fire.dt_max = std::max(solver.fire.dt, 0.5 * solver.fire.dt_max);
        std::println(std::cout, "Retrying {} with dt={:.3e}, dt_max={:.3e}", label, solver.fire.dt, solver.fire.dt_max);
      }
    }
  }

  // Constrained saddle-point search: minimize the Helmholtz free energy
  // at fixed total mass (or fixed excess mass relative to a background)
  // with exponential-backoff retry on FIRE time-step. After convergence,
  // analyse the result to extract the thermodynamic barrier.

  struct ConstrainedSearch {
    minimization::Minimizer minimizer;
    int max_retries{0};

    // Find the saddle point at fixed total mass.

    [[nodiscard]] auto fixed_mass(
        const functionals::Functional& func,
        const arma::vec& rho0,
        double mu,
        double target_mass,
        double rho_v_coex,
        double rho_l_coex,
        const arma::vec& external_field = {}
    ) const -> ConstrainedResult {
      auto result = _retry("constrained fixed-mass solve", [&](const minimization::Minimizer& solver) {
        return solver.fixed_mass(func.model, func.weights, rho0, mu, target_mass, external_field);
      });

      return _analyse_periodic(func, result, rho_v_coex, rho_l_coex);
    }

    // Find the saddle point at fixed excess mass relative to a background.

    [[nodiscard]] auto fixed_excess_mass(
        const functionals::Functional& func,
        const arma::vec& rho0,
        double mu,
        const arma::vec& background,
        double target_excess,
        double rho_v_coex,
        double rho_l_coex,
        const arma::vec& external_field = {},
        const arma::uvec& reservoir_mask = {}
    ) const -> ConstrainedResult {
      auto start = std::chrono::steady_clock::now();
      auto result = _retry("constrained fixed-excess-mass solve", [&](const minimization::Minimizer& solver) {
        return solver.fixed_excess_mass(func.model, func.weights, rho0, mu, background, target_excess, external_field);
      });
      auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

      double final_excess = arma::accu(arma::clamp(result.densities[0] - background, 0.0, arma::datum::inf))
          * func.model.grid.cell_volume();
      std::println(
          std::cout,
          "Constrained solve done: converged={}  iters={}  F={:.6f}  excess_mass={:.6f}  elapsed={:.1f}s",
          result.converged,
          result.iterations,
          result.free_energy,
          final_excess,
          elapsed
      );

      return _analyse_with_background(
          func,
          result,
          background,
          mu,
          reservoir_mask,
          rho_v_coex,
          rho_l_coex,
          external_field
      );
    }

   private:
    template <typename SolveFn>
    [[nodiscard]] auto _retry(std::string_view label, SolveFn&& solve) const -> minimization::Result {
      return retry_with_backoff(label, minimizer, max_retries, std::forward<SolveFn>(solve));
    }

    [[nodiscard]] auto _analyse_periodic(
        const functionals::Functional& func,
        const minimization::Result& result,
        double rho_v_coex,
        double rho_l_coex
    ) const -> ConstrainedResult {
      arma::uvec bdry = func.model.grid.boundary_mask();
      double background = Grid::face_average(result.densities[0], bdry);

      auto eos = func.bulk();
      double mu_bg = eos.chemical_potential(arma::vec{background}, 0);

      double omega_cluster = func.grand_potential(result.densities[0], mu_bg);
      arma::vec rho_uniform(result.densities[0].n_elem, arma::fill::value(background));
      double omega_bg = func.grand_potential(rho_uniform, mu_bg);

      double dv = func.model.grid.cell_volume();
      double delta_rho = rho_l_coex - rho_v_coex;
      double R_eff = dft::effective_radius(result.densities[0], background, delta_rho, dv);

      auto rho_v_opt = functionals::bulk::density_from_chemical_potential(mu_bg, rho_v_coex * 0.9, eos);
      auto rho_l_opt = functionals::bulk::density_from_chemical_potential(mu_bg, rho_l_coex * 1.1, eos);

      return {
          .minimization = result,
          .background = background,
          .mu_background = mu_bg,
          .omega_cluster = omega_cluster,
          .omega_background = omega_bg,
          .barrier = omega_cluster - omega_bg,
          .effective_radius = R_eff,
          .rho_vapor_meta = rho_v_opt.value_or(background),
          .rho_liquid_meta = rho_l_opt.value_or(rho_l_coex),
          .boundary_mask = bdry,
      };
    }

    [[nodiscard]] auto _analyse_with_background(
        const functionals::Functional& func,
        const minimization::Result& result,
        const arma::vec& rho_background,
        double mu_background,
        const arma::uvec& reservoir_mask,
        double rho_v_coex,
        double rho_l_coex,
        const arma::vec& external_field
    ) const -> ConstrainedResult {
      arma::uvec mask = reservoir_mask.n_elem > 0 ? reservoir_mask : func.model.grid.boundary_mask();
      double background = Grid::face_average(rho_background, mask);
      double omega_cluster = func.evaluate(result.densities[0], mu_background, external_field).grand_potential;
      double omega_bg = func.evaluate(rho_background, mu_background, external_field).grand_potential;

      double dv = func.model.grid.cell_volume();
      double delta_rho = rho_l_coex - rho_v_coex;
      double R_eff = dft::effective_radius(result.densities[0], background, delta_rho, dv);

      auto eos = func.bulk();
      auto rho_v_opt = functionals::bulk::density_from_chemical_potential(mu_background, rho_v_coex * 0.9, eos);
      auto rho_l_opt = functionals::bulk::density_from_chemical_potential(mu_background, rho_l_coex * 1.1, eos);

      return {
          .minimization = result,
          .background = background,
          .mu_background = mu_background,
          .omega_cluster = omega_cluster,
          .omega_background = omega_bg,
          .barrier = omega_cluster - omega_bg,
          .effective_radius = R_eff,
          .rho_vapor_meta = rho_v_opt.value_or(background),
          .rho_liquid_meta = rho_l_opt.value_or(rho_l_coex),
          .boundary_mask = mask,
      };
    }
  };

  // Result of a wall-ramp search: the constrained result plus the
  // background density and the initial seed used.

  struct WallRampResult {
    ConstrainedResult cluster;
    arma::vec rho_initial;
    arma::vec rho_background;
  };

  // Callback that produces an initial density given the grid, radial
  // distances from the seed center, coexistence densities, and outer
  // (supersaturated) density.

  using SeedFunction = std::function<arma::vec(const Grid& grid, const arma::vec& r, double rho_l, double rho_out)>;

  // Callback that places a cluster seed onto a non-uniform background.
  // Arguments: grid, background density, seed center, optional predictor
  // spline and its cutoff radius.

  using BackgroundSeedFunction = std::function<arma::vec(
      const Grid& grid,
      const arma::vec& background,
      const std::array<double, 3>& center,
      const math::CubicSpline* predictor_spline,
      double predictor_cutoff
  )>;

  // Wall-ramp search: find a critical cluster in the presence of an
  // external field (wall), using a multi-stage ramp strategy.
  //
  // For periodic (no-wall) systems: rescale seed to target mass and
  // call ConstrainedSearch::fixed_mass().
  //
  // For wall-attached systems:
  //   1. Solve a fully periodic predictor to estimate the excess mass.
  //   2. Equilibrate the background density at the wall.
  //   3. Seed the cluster onto the background.
  //   4. Call ConstrainedSearch::fixed_excess_mass().

  struct WallRampSearch {
    minimization::Minimizer minimizer;
    int max_retries{0};
    double droplet_radius{0.0};
    double rho_l_max_factor{4.0};
    double predictor_dx{0.0}; // Grid spacing for the periodic predictor (0 = same as main grid).

    // Find the critical cluster. The seed functions generate the initial
    // density and are the only experiment-specific input.

    [[nodiscard]] auto find(
        const functionals::Functional& func,
        const physics::walls::WallPotential& wall_potential,
        const arma::vec& external_field,
        const SeedFunction& make_seed,
        const BackgroundSeedFunction& make_background_seed,
        const std::array<double, 3>& seed_center,
        double rho_v_coex,
        double rho_l_coex,
        double rho_out,
        double mu_out
    ) const -> WallRampResult {
      bool has_wall = wall_potential.is_active();
      double dv = func.model.grid.cell_volume();

      auto r = func.model.grid.radial_distances(seed_center);
      arma::vec rho_reference = StepProfile{.radius = droplet_radius, .rho_in = rho_l_coex, .rho_out = rho_out}.apply(r
      );
      double excess_mass = arma::accu(rho_reference - rho_out) * dv;
      double target_mass = arma::accu(rho_reference) * dv;

      auto wall_solver = has_wall ? _wall_adjusted(minimizer) : minimizer;
      auto cluster_solver = has_wall ? _cluster_adjusted(minimizer) : minimizer;
      int retries = has_wall ? max_retries : 0;

      arma::vec rho_background(static_cast<arma::uword>(func.model.grid.total_points()), arma::fill::value(rho_out));
      arma::vec rho0 = make_seed(func.model.grid, r, rho_l_coex, rho_out);

      if (has_wall) {
        auto predictor =
            _periodic_predictor(func, cluster_solver, make_seed, seed_center, rho_l_coex, rho_out, mu_out, retries);
        excess_mass = predictor.excess_mass;

        rho_background = _wall_background(func, wall_solver, rho_l_coex, rho_out, mu_out, external_field, retries);

        rho0 = make_background_seed(
            func.model.grid,
            rho_background,
            seed_center,
            predictor.spline.get(),
            predictor.cutoff
        );
        rho0 = wall_potential.suppress_excess(std::move(rho0), rho_background, func.model.grid);

        if (!predictor.spline) {
          excess_mass = arma::accu(arma::clamp(rho0 - rho_background, 0.0, arma::datum::inf)) * dv;
        }
        rho0 = rescale_excess_mass(rho0, rho_background, excess_mass, dv);
      }

      arma::vec rho_initial = rho0;
      if (!has_wall) {
        rho0 = rescale_mass(rho0, target_mass, dv);
        rho_initial = rho0;
      }

      auto search = ConstrainedSearch{.minimizer = cluster_solver, .max_retries = retries};

      arma::uvec reservoir_mask = wall_potential.reservoir_mask(func.model.grid);
      auto info = has_wall ? search.fixed_excess_mass(
                                 func,
                                 rho0,
                                 mu_out,
                                 rho_background,
                                 excess_mass,
                                 rho_v_coex,
                                 rho_l_coex,
                                 external_field,
                                 reservoir_mask
                             )
                           : search.fixed_mass(func, rho0, mu_out, target_mass, rho_v_coex, rho_l_coex, external_field);

      std::println(std::cout, "\nCritical cluster:");
      std::println(std::cout, "  converged={} ({} iters)", info.minimization.converged, info.minimization.iterations);
      std::println(std::cout, "  R_eff={:.4f}  background={:.6f}", info.effective_radius, info.background);
      std::println(std::cout, "  rho_v(mu)={:.6f}  rho_l(mu)={:.6f}", info.rho_vapor_meta, info.rho_liquid_meta);
      std::println(std::cout, "  Delta_Omega={:.6f}  (barrier height)", info.barrier);

      return {
          .cluster = std::move(info),
          .rho_initial = std::move(rho_initial),
          .rho_background = std::move(rho_background),
      };
    }

   private:
    struct _PeriodicPredictor {
      std::unique_ptr<math::CubicSpline> spline;
      double cutoff{0.0};
      double excess_mass{0.0};
    };

    [[nodiscard]] static auto _wall_adjusted(minimization::Minimizer solver) -> minimization::Minimizer {
      solver.fire.dt = std::min(solver.fire.dt, 0.02);
      solver.fire.dt_max = std::min(solver.fire.dt_max, 0.25);
      solver.fire.max_uphill = std::max(solver.fire.max_uphill, 200);
      return solver;
    }

    [[nodiscard]] static auto _cluster_adjusted(minimization::Minimizer solver) -> minimization::Minimizer {
      solver.fire.max_uphill = std::max(solver.fire.max_uphill, 200);
      return solver;
    }

    [[nodiscard]] auto _periodic_predictor(
        const functionals::Functional& func,
        const minimization::Minimizer& cluster_solver,
        const SeedFunction& make_seed,
        const std::array<double, 3>& seed_origin,
        double rho_l,
        double rho_out,
        double mu_out,
        int retries
    ) const -> _PeriodicPredictor {
      // Snap predictor dx to the nearest commensurate value (≥ requested)
      // so that every box dimension is an exact multiple of dx.
      auto commensurate_dx = [](double target_dx, const std::array<double, 3>& box) {
        double dx = target_dx;
        for (double L : box) {
          long n = std::max(1L, static_cast<long>(std::floor(L / dx)));
          dx = std::min(dx, L / static_cast<double>(n));
        }
        return dx;
      };

      double raw_dx = predictor_dx > 0.0 ? std::max(predictor_dx, func.model.grid.dx) : func.model.grid.dx;
      double pred_dx = commensurate_dx(raw_dx, func.model.grid.box_size);

      auto periodic_model = func.model;
      periodic_model.grid = make_grid(pred_dx, func.model.grid.box_size, {true, true, true});
      auto periodic_func = functionals::make_functional(func.weights.fmt_model, periodic_model);

      auto periodic_center = std::array<double, 3>{
          periodic_model.grid.box_size[0] / 2.0,
          periodic_model.grid.box_size[1] / 2.0,
          periodic_model.grid.box_size[2] / 2.0,
      };
      auto periodic_r = periodic_func.model.grid.radial_distances(periodic_center);

      arma::vec reference = StepProfile{.radius = droplet_radius, .rho_in = rho_l, .rho_out = rho_out}.apply(periodic_r
      );
      arma::vec seed = make_seed(periodic_func.model.grid, periodic_r, rho_l, rho_out);

      double dv = periodic_func.model.grid.cell_volume();
      double target_mass = arma::accu(reference) * dv;
      seed = rescale_mass(seed, target_mass, dv);

      auto solver = cluster_solver;
      solver.use_homogeneous_boundary = true;

      std::println(std::cout, "Periodic predictor: solving homogeneous critical droplet");
      auto start = std::chrono::steady_clock::now();

      auto result =
          retry_with_backoff("periodic critical cluster solve", solver, retries, [&](const minimization::Minimizer& s) {
            return s.fixed_mass(periodic_func.model, periodic_func.weights, seed, mu_out, target_mass);
          });

      auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

      double excess = arma::accu(arma::clamp(result.densities[0] - rho_out, 0.0, arma::datum::inf)) * dv;

      auto profile = periodic_func.model.grid.radial_profile(result.densities[0]);

      std::vector<double> pred_r{0.0};
      std::vector<double> pred_excess{
          std::max(periodic_func.model.grid.center_value(result.densities[0]) - rho_out, 0.0)
      };
      for (std::size_t i = 0; i < profile.r.size(); ++i) {
        if (profile.r[i] > pred_r.back()) {
          pred_r.push_back(profile.r[i]);
          pred_excess.push_back(std::max(profile.values[i] - rho_out, 0.0));
        }
      }
      pred_r.push_back(pred_r.back() + periodic_func.model.grid.dx);
      pred_excess.push_back(0.0);

      std::println(
          std::cout,
          "Periodic predictor done: converged={}  iters={}  F={:.6f}  excess_mass={:.6f}  elapsed={:.1f}s",
          result.converged,
          result.iterations,
          result.free_energy,
          excess,
          elapsed
      );

      return {
          .spline = std::make_unique<math::CubicSpline>(pred_r, pred_excess),
          .cutoff = pred_r.back(),
          .excess_mass = excess,
      };
    }

    [[nodiscard]] auto _wall_background(
        const functionals::Functional& func,
        const minimization::Minimizer& wall_solver,
        double rho_l,
        double rho_out,
        double mu_out,
        const arma::vec& external_field,
        int retries
    ) const -> arma::vec {
      double rho_max = std::max(rho_l_max_factor * rho_l, 2.0);

      auto bg_minimizer = minimizer;
      bg_minimizer.fire = wall_solver.fire;
      bg_minimizer.param = minimization::Bounded{.rho_min = 1e-18, .rho_max = rho_max};
      bg_minimizer.fire.force_tolerance = std::max(minimizer.fire.force_tolerance, 1e-5);
      bg_minimizer.use_homogeneous_boundary = false;
      bg_minimizer.log_interval = 20;

      arma::vec guess = arma::clamp(rho_out * arma::exp(arma::clamp(-external_field, -50.0, 50.0)), 1e-18, rho_max);

      std::println(std::cout, "Wall background: solving grand-potential minimum at final wall field");
      auto start = std::chrono::steady_clock::now();

      auto result =
          retry_with_backoff("wall background solve", wall_solver, retries, [&](const minimization::Minimizer& s) {
            auto solver = bg_minimizer;
            solver.fire.dt = s.fire.dt;
            solver.fire.dt_max = s.fire.dt_max;
            return solver.grand_potential(func.model, func.weights, guess, mu_out, external_field);
          });

      auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::println(
          std::cout,
          "Wall background done: converged={}  iters={}  Omega={:.6f}  elapsed={:.1f}s",
          result.converged,
          result.iterations,
          result.grand_potential,
          elapsed
      );

      return std::move(result.densities[0]);
    }
  };

} // namespace dft::algorithms::saddle_point

#endif // DFT_ALGORITHMS_SADDLE_POINT_HPP
