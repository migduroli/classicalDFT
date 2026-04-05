#ifndef DFT_ALGORITHMS_SADDLE_POINT_HPP
#define DFT_ALGORITHMS_SADDLE_POINT_HPP

#include "dft/algorithms/fire.hpp"
#include "dft/algorithms/minimization.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/grid.hpp"
#include "dft/init.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
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

      arma::vec v = initial_guess.n_elem == n
          ? initial_guess
          : arma::randn(n);
      v /= arma::norm(v);

      if (config.log_interval > 0) {
        std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}\n",
                                 "iter", "eigenvalue", "residual");
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

  }  // namespace _internal

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

    // Find the smallest eigenvalue and its eigenvector of the Hessian
    // d^2 Omega / (drho_i drho_j) using LOBPCG.

    [[nodiscard]] auto solve(
        const ForceFunction& force_fn,
        const arma::vec& rho,
        const arma::vec& initial_guess = {}
    ) const -> EigenvalueResult;
  };

  [[nodiscard]] inline auto EigenvalueSolver::solve(
      const ForceFunction& force_fn,
      const arma::vec& rho,
      const arma::vec& initial_guess
  ) const -> EigenvalueResult {
    _internal::EigenvalueConfig internal_cfg{
        .tolerance = tolerance,
        .max_iterations = max_iterations,
        .hessian_eps = hessian_eps,
        .log_interval = log_interval,
    };

    auto [omega, forces] = force_fn(rho);
    arma::uword n = rho.n_elem;

    arma::vec v = initial_guess.n_elem == n
        ? initial_guess
        : arma::randn(n);
    v /= arma::norm(v);

    if (log_interval > 0) {
      std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}\n",
                               "iter", "eigenvalue", "residual");
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

  using MultiSpeciesForceFunction = std::function<std::pair<double, std::vector<arma::vec>>(
      const std::vector<arma::vec>&
  )>;

  [[nodiscard]] inline auto reversed_forces(
      MultiSpeciesForceFunction compute, std::vector<arma::vec> directions
  ) -> MultiSpeciesForceFunction {
    for (auto& d : directions) {
      double n = arma::norm(d);
      if (n > 0.0) {
        d /= n;
      }
    }

    return [compute = std::move(compute), directions = std::move(directions)](
               const std::vector<arma::vec>& densities
           ) -> std::pair<double, std::vector<arma::vec>> {
      auto [energy, forces] = compute(densities);
      for (std::size_t s = 0; s < forces.size() && s < directions.size(); ++s) {
        double proj = arma::dot(directions[s], forces[s]);
        forces[s] -= 2.0 * proj * directions[s];
      }
      return {energy, std::move(forces)};
    };
  }

  }  // namespace _internal

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

      auto eig_result = eigenvalue.solve(
          eig_force_fn, rho,
          (outer >= eigenvalue_clear_count && eigvec.n_elem == rho.n_elem)
              ? eigvec : arma::vec{});

      eigvec = eig_result.eigenvector;
      eigenvalue_val = eig_result.eigenvalue;

      std::cout << std::format("  eigenvalue = {:.6e}  converged = {}  iters = {}\n",
                               eigenvalue_val, eig_result.converged, eig_result.iterations);

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

      auto compute = [&](const std::vector<arma::vec>& x_param)
          -> std::pair<double, std::vector<arma::vec>> {
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

      auto fire_state = fire_cfg.initialize(
          {minimization::_internal::from_density(rho, param)}, compute);
      auto [e0, forces0] = compute(fire_state.x);

      if (log_interval > 0) {
        std::cout << std::format("    {:>6s}  {:>14s}  {:>14s}  {:>10s}\n",
                                 "iter", "Omega", "rms_force", "dt");
        std::cout << "    " << std::string(50, '-') << "\n";
        std::cout << std::format("    {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
                                 0, fire_state.energy, fire_state.rms_force, fire_state.dt);
      }

      for (int i = 0; i < fire_cfg.max_steps && !fire_state.converged; ++i) {
        auto [ns, nf] = fire_cfg.step(std::move(fire_state), forces0, compute);
        fire_state = std::move(ns);
        forces0 = std::move(nf);

        if (log_interval > 0 &&
            ((i + 1) % log_interval == 0 || fire_state.converged)) {
          std::cout << std::format("    {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
                                   i + 1, fire_state.energy, fire_state.rms_force, fire_state.dt);
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

      std::cout << std::format("  Omega = {:.8f}  (delta = {:.2e})  gamma = {:.2e}\n\n",
                               omega, std::abs(omega - omega_old), gamma);

      if (outer > 0 && std::abs(omega - omega_old) < omega_tolerance) {
        converged = true;
        break;
      }

      x_current = minimization::_internal::from_density(rho, param);
    }

    std::cout << std::format("  Eigenvector following {}  ({} outer iterations)\n",
                             converged ? "CONVERGED" : "NOT CONVERGED", outer + 1);

    return Result{
        .density = rho,
        .eigenvector = eigvec,
        .grand_potential = omega,
        .eigenvalue = eigenvalue_val,
        .outer_iterations = outer + 1,
        .converged = converged,
    };
  }

}  // namespace dft::algorithms::saddle_point

#endif  // DFT_ALGORITHMS_SADDLE_POINT_HPP
