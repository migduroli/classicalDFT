#ifndef DFT_ALGORITHMS_EIGENVECTOR_FOLLOWING_HPP
#define DFT_ALGORITHMS_EIGENVECTOR_FOLLOWING_HPP

#include "dft/algorithms/eigenvalue.hpp"
#include "dft/algorithms/fire.hpp"
#include "dft/algorithms/hessian.hpp"
#include "dft/algorithms/parametrization.hpp"
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

namespace dft::algorithms {

  struct EigenvectorFollowingConfig {
    fire::FireConfig fire{
        .dt = 1e-3,
        .dt_max = 1.0,
        .alpha_start = 0.01,
        .f_alpha = 1.0,
        .force_tolerance = 1e-4,
        .max_steps = 100000,
    };
    EigenvalueConfig eigenvalue{
        .tolerance = 1e-4,
        .max_iterations = 300,
        .hessian_eps = 1e-6,
    };
    parametrization::Parametrization param = parametrization::Unbounded{.rho_min = 1e-18};
    double omega_tolerance{1e-5};
    double tolerance_start{1e-3};
    double tolerance_factor{10.0};
    int max_outer_iterations{20};
    int eigenvalue_clear_count{2};
    int log_interval{100};
  };

  struct SaddlePointResult {
    arma::vec density;
    arma::vec eigenvector;
    double grand_potential{0.0};
    double eigenvalue{0.0};
    int outer_iterations{0};
    bool converged{false};
  };

  // Find the saddle point (critical cluster) of the grand potential
  // Omega[rho] using eigenvector following (after Lutsko).
  //
  // The mass is kept FIXED throughout. The input density (from a
  // fixed-mass minimisation) already sits at the right mass for the
  // critical cluster. The boundary forces are zeroed (fixed boundary).
  //
  // Algorithm per outer iteration:
  //   1. Compute smallest eigenvalue/eigenvector of the Hessian
  //   2. Run FIRE2 at fixed mass with:
  //      - density rescaling to target mass each step
  //      - fixed-boundary (zero forces at face points)
  //      - Lagrange multiplier (interior points only)
  //      - direction projection (remove force component along eigvec)
  //   3. Re-evaluate Omega; repeat until |Omega_new - Omega_old| < tol
  //
  // Force tolerances tighten progressively: start at tolerance_start,
  // divide by tolerance_factor each outer iteration.

  [[nodiscard]] inline auto find_saddle_point(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      double target_mass,
      const EigenvectorFollowingConfig& config = {}
  ) -> SaddlePointResult {
    double dv = model.grid.cell_volume();
    arma::uvec bdry = boundary_mask(model.grid);

    // Force function for the eigenvalue solver: grand-canonical forces
    // with fixed boundary (no mass constraint, no alias).
    // This is used ONLY for Hessian-vector products.

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
    double eigenvalue = 0.0;
    bool converged = false;
    int outer = 0;
    double ftol = config.tolerance_start;

    std::cout << "\n=== Eigenvector following: saddle-point search ===\n";
    std::cout << "  target_mass = " << target_mass << "\n\n";

    for (; outer < config.max_outer_iterations; ++outer) {
      double omega_old = omega;

      std::cout << std::format("  --- outer iteration {} ---\n", outer);

      // Step 1: compute smallest eigenvalue/eigenvector.

      std::cout << "  Computing smallest eigenvalue...\n";

      auto eig_config = config.eigenvalue;
      auto eig_result = smallest_eigenvalue(
          eig_force_fn, rho, eig_config,
          (outer >= config.eigenvalue_clear_count && eigvec.n_elem == rho.n_elem)
              ? eigvec : arma::vec{});

      eigvec = eig_result.eigenvector;
      eigenvalue = eig_result.eigenvalue;

      std::cout << std::format("  eigenvalue = {:.6e}  converged = {}  iters = {}\n",
                               eigenvalue, eig_result.converged, eig_result.iterations);

      // Step 2: FIRE2 at fixed mass + fixed boundary + direction projection.
      //
      // Pipeline per FIRE step (mirrors Lutsko's Species + Minimizer):
      //   a. Convert alias -> density: rho = rho_min + x^2
      //   b. Rescale interior density to target mass (boundary stays fixed)
      //   c. Evaluate grand potential and forces
      //   d. Zero forces at boundary points (fixed boundary)
      //   e. Lagrange multiplier from interior: lambda = sum(f*rho)_interior / M_interior
      //      then f -= lambda * dV for interior points
      //   f. Convert forces to alias space: f_x = 2 * x * f_rho
      //   g. Project out the eigenvector component (in alias space): f -= (d.f) d
      //   h. Negate for FIRE convention

      // Convert eigenvector to alias space (like Jim's add_fixed_direction).
      arma::vec x_current = parametrization::from_density(rho, config.param);
      arma::vec dir_alias = eigvec;
      dir_alias = parametrization::transform_force(dir_alias, x_current, config.param);
      double dn = arma::norm(dir_alias);
      if (dn > 1e-30) dir_alias /= dn;

      // Precompute boundary mass for the mass constraint.
      double rho_boundary_avg = 0.0;
      arma::uword n_boundary = 0;
      for (arma::uword i = 0; i < rho.n_elem; ++i) {
        if (bdry(i)) { rho_boundary_avg += rho(i); n_boundary++; }
      }
      if (n_boundary > 0) rho_boundary_avg /= static_cast<double>(n_boundary);
      double m_boundary = rho_boundary_avg * dv * static_cast<double>(n_boundary);

      auto compute = [&](const std::vector<arma::vec>& x_param)
          -> std::pair<double, std::vector<arma::vec>> {
        arma::vec rho_local = parametrization::to_density(x_param[0], config.param);

        // Rescale interior density to target mass (boundary excluded).
        double m_interior = 0.0;
        for (arma::uword i = 0; i < rho_local.n_elem; ++i) {
          if (!bdry(i)) m_interior += rho_local(i);
        }
        m_interior *= dv;
        double m_target_interior = target_mass - m_boundary;
        if (m_interior > 0.0 && m_target_interior > 0.0) {
          double scale = m_target_interior / m_interior;
          for (arma::uword i = 0; i < rho_local.n_elem; ++i) {
            if (!bdry(i)) rho_local(i) *= scale;
          }
        }

        // Evaluate grand potential and forces.
        auto state = init::from_profile(model, rho_local);
        state.species[0].chemical_potential = chemical_potential;
        auto result = functionals::total(model, state, weights);
        arma::vec grad = result.forces[0];

        // Fixed boundary: zero forces at face points.
        grad = fixed_boundary(grad, bdry);

        // Lagrange multiplier for fixed mass (interior only).
        double mu_lagrange = 0.0;
        for (arma::uword i = 0; i < rho_local.n_elem; ++i) {
          if (!bdry(i)) mu_lagrange += grad(i) * rho_local(i);
        }
        if (m_target_interior > 0.0) mu_lagrange /= m_target_interior;
        for (arma::uword i = 0; i < grad.n_elem; ++i) {
          if (!bdry(i)) grad(i) -= mu_lagrange * dv;
          else grad(i) = 0.0;
        }

        // Convert to alias space.
        arma::vec f_alias = parametrization::transform_force(grad, x_param[0], config.param);

        // Project out the eigenvector component (in alias space).
        f_alias -= arma::dot(dir_alias, f_alias) * dir_alias;

        // Negate for FIRE convention.
        return {result.grand_potential, {-f_alias}};
      };

      std::cout << "  Running FIRE2 (fixed mass + projected)...\n";

      auto fire_config = config.fire;
      fire_config.force_tolerance = std::max(ftol, config.fire.force_tolerance);
      ftol /= config.tolerance_factor;

      auto fire_state = fire::initialize({parametrization::from_density(rho, config.param)},
                                         compute, fire_config);
      auto [e0, forces0] = compute(fire_state.x);

      if (config.log_interval > 0) {
        std::cout << std::format("    {:>6s}  {:>14s}  {:>14s}  {:>10s}\n",
                                 "iter", "Omega", "rms_force", "dt");
        std::cout << "    " << std::string(50, '-') << "\n";
        std::cout << std::format("    {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
                                 0, fire_state.energy, fire_state.rms_force, fire_state.dt);
      }

      for (int i = 0; i < fire_config.max_steps && !fire_state.converged; ++i) {
        auto [ns, nf] = fire::step(std::move(fire_state), forces0, compute, fire_config);
        fire_state = std::move(ns);
        forces0 = std::move(nf);

        if (config.log_interval > 0 &&
            ((i + 1) % config.log_interval == 0 || fire_state.converged)) {
          std::cout << std::format("    {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
                                   i + 1, fire_state.energy, fire_state.rms_force, fire_state.dt);
        }
      }

      // Extract final density.
      rho = parametrization::to_density(fire_state.x[0], config.param);
      double fm = 0.0;
      for (arma::uword i = 0; i < rho.n_elem; ++i) {
        if (!bdry(i)) fm += rho(i);
      }
      fm *= dv;
      double m_target_interior = target_mass - m_boundary;
      if (fm > 0.0 && m_target_interior > 0.0) {
        double scale = m_target_interior / fm;
        for (arma::uword i = 0; i < rho.n_elem; ++i) {
          if (!bdry(i)) rho(i) *= scale;
        }
      }

      // Evaluate Omega at the final density.
      auto s_final = init::from_profile(model, rho);
      s_final.species[0].chemical_potential = chemical_potential;
      auto r_final = functionals::total(model, s_final, weights);
      omega = r_final.grand_potential;

      // Check force along eigenvector (gamma in Jim's code).
      arma::vec grad_final = r_final.forces[0];
      grad_final = fixed_boundary(grad_final, bdry);
      double gamma = arma::dot(grad_final, eigvec);

      std::cout << std::format("  Omega = {:.8f}  (delta = {:.2e})  gamma = {:.2e}\n\n",
                               omega, std::abs(omega - omega_old), gamma);

      if (outer > 0 && std::abs(omega - omega_old) < config.omega_tolerance) {
        converged = true;
        break;
      }

      // Update dir_alias for next iteration (rho changed, so alias changed).
      x_current = parametrization::from_density(rho, config.param);
    }

    std::cout << std::format("  Eigenvector following {}  ({} outer iterations)\n",
                             converged ? "CONVERGED" : "NOT CONVERGED", outer + 1);

    return SaddlePointResult{
        .density = rho,
        .eigenvector = eigvec,
        .grand_potential = omega,
        .eigenvalue = eigenvalue,
        .outer_iterations = outer + 1,
        .converged = converged,
    };
  }

}  // namespace dft::algorithms

#endif  // DFT_ALGORITHMS_EIGENVECTOR_FOLLOWING_HPP
