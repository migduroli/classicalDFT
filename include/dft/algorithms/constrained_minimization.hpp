#ifndef DFT_ALGORITHMS_CONSTRAINED_MINIMIZATION_HPP
#define DFT_ALGORITHMS_CONSTRAINED_MINIMIZATION_HPP

#include "dft/algorithms/fire.hpp"
#include "dft/algorithms/parametrization.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/grid.hpp"
#include "dft/init.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <format>
#include <functional>
#include <iostream>
#include <vector>

namespace dft::algorithms {

  struct ConstrainedMinimizationConfig {
    fire::FireConfig fire{};
    parametrization::Parametrization param = parametrization::Unbounded{};
    bool homogeneous_boundary{true};
    int log_interval{100};
  };

  struct ConstrainedMinimizationResult {
    std::vector<arma::vec> densities;
    double free_energy{0.0};
    double grand_potential{0.0};
    double rms_force{0.0};
    int iterations{0};
    bool converged{false};
  };

  // Minimize the Helmholtz free energy F[rho] at fixed total mass N,
  // using FIRE2 in a positivity-preserving parameter space.
  //
  // The critical (unstable) cluster of the open system Omega[rho]
  // at chemical potential mu is equivalent to the stable minimum
  // of F[rho] at fixed N, where the Lagrange multiplier lambda = mu.
  //
  // The pipeline per FIRE step (mirroring Lutsko's Species class):
  //   1. Convert parameters to density: rho = rho_min + x^2
  //   2. Rescale density to enforce target mass
  //   3. Evaluate functional derivatives dOmega/drho * dV
  //   4. Apply boundary conditions (homogeneous or fixed)
  //   5. Subtract Lagrange multiplier: f -= lambda * dV
  //   6. Convert forces to parameter space: f_x = 2x * f_rho
  //   7. Negate for FIRE convention (f = -grad)

  [[nodiscard]] inline auto minimize_at_fixed_mass(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      double target_mass,
      const ConstrainedMinimizationConfig& config = {}
  ) -> ConstrainedMinimizationResult {
    double dv = model.grid.cell_volume();
    arma::uvec bdry = boundary_mask(model.grid);

    auto compute = [&](const std::vector<arma::vec>& x_param)
        -> std::pair<double, std::vector<arma::vec>> {
      arma::vec rho = parametrization::to_density(x_param[0], config.param);

      // Rescale to target mass.
      double mass = arma::accu(rho) * dv;
      if (mass > 0.0) rho *= target_mass / mass;

      // Evaluate Helmholtz free energy and gradient.
      // Chemical potential is set to zero so that forces are dF/drho * dV.
      // The Lagrange multiplier for fixed mass acts as the chemical
      // potential and is determined self-consistently.
      auto state = init::from_profile(model, rho);
      state.species[0].chemical_potential = 0.0;
      auto result = functionals::total(model, state, weights);
      arma::vec grad = result.forces[0];

      // Boundary conditions.
      if (config.homogeneous_boundary) {
        grad = homogeneous_boundary(grad, bdry);
      }

      // Lagrange multiplier for fixed mass.
      double lambda = arma::dot(grad, rho) / target_mass;
      grad -= lambda * dv;

      // Convert to parameter space and negate for FIRE.
      arma::vec f = parametrization::transform_force(grad, x_param[0], config.param);
      return {result.grand_potential, {-f}};
    };

    arma::vec x0 = parametrization::from_density(initial_density, config.param);

    if (config.log_interval > 0) {
      std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}  {:>10s}\n",
                               "iter", "F", "monitor", "dt");
      std::cout << "  " << std::string(50, '-') << "\n";
    }

    double volume = static_cast<double>(initial_density.n_elem) * dv;
    auto state = fire::initialize({x0}, compute, config.fire);
    auto [e0, forces] = compute(state.x);

    if (config.log_interval > 0) {
      std::cout << std::format("  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
                               0, state.energy, state.rms_force, state.dt);
    }

    // Jim's convergence: |vv/Volume| where vv = 0.9*vv + 0.1*|dF/dt|.
    double vv = 1.0;
    bool converged = false;

    for (int i = 0; i < config.fire.max_steps && !converged; ++i) {
      double old_energy = state.energy;
      auto [ns, nf] = fire::step(std::move(state), forces, compute, config.fire);
      state = std::move(ns);
      forces = std::move(nf);

      vv = vv * 0.9 + 0.1 * (std::abs(state.energy - old_energy) / state.dt);
      double monitor = std::abs(vv / volume);
      converged = monitor < config.fire.force_tolerance;
      state.converged = converged;

      if (config.log_interval > 0 &&
          ((i + 1) % config.log_interval == 0 || converged)) {
        std::cout << std::format("  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
                                 i + 1, state.energy, monitor, state.dt);
      }
    }

    // Extract final density.
    arma::vec rho_final = parametrization::to_density(state.x[0], config.param);
    double fm = arma::accu(rho_final) * dv;
    if (fm > 0.0) rho_final *= target_mass / fm;

    auto s_final = init::from_profile(model, rho_final);
    s_final.species[0].chemical_potential = chemical_potential;
    auto r_final = functionals::total(model, s_final, weights);

    return ConstrainedMinimizationResult{
        .densities = {rho_final},
        .free_energy = r_final.free_energy,
        .grand_potential = r_final.grand_potential,
        .rms_force = state.rms_force,
        .iterations = state.iteration,
        .converged = state.converged,
    };
  }

  // Minimize the grand potential Omega[rho] in the open (grand-canonical) system.
  //
  // This mirrors Lutsko's droplet.cpp: the chemical potential is fixed
  // (set from the background density), homogeneous boundary conditions
  // keep the faces at the average boundary force, and mass is free to
  // change.  FIRE converges to the nearest stationary point of Omega
  // under these boundary conditions — for a droplet initial condition
  // this gives the critical cluster.

  [[nodiscard]] inline auto minimize_grand_potential(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      const ConstrainedMinimizationConfig& config = {}
  ) -> ConstrainedMinimizationResult {
    arma::uvec bdry = boundary_mask(model.grid);

    auto compute = [&](const std::vector<arma::vec>& x_param)
        -> std::pair<double, std::vector<arma::vec>> {
      arma::vec rho = parametrization::to_density(x_param[0], config.param);

      auto state = init::from_profile(model, rho);
      state.species[0].chemical_potential = chemical_potential;
      auto result = functionals::total(model, state, weights);
      arma::vec grad = result.forces[0];

      if (config.homogeneous_boundary) {
        grad = homogeneous_boundary(grad, bdry);
      }

      arma::vec f = parametrization::transform_force(grad, x_param[0], config.param);
      return {result.grand_potential, {-f}};
    };

    arma::vec x0 = parametrization::from_density(initial_density, config.param);

    if (config.log_interval > 0) {
      std::cout << std::format("  {:>6s}  {:>14s}  {:>14s}  {:>10s}\n",
                               "iter", "Omega", "monitor", "dt");
      std::cout << "  " << std::string(50, '-') << "\n";
    }

    double volume = static_cast<double>(initial_density.n_elem) * model.grid.cell_volume();
    auto state = fire::initialize({x0}, compute, config.fire);
    auto [e0, forces] = compute(state.x);

    if (config.log_interval > 0) {
      std::cout << std::format("  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
                               0, state.energy, state.rms_force, state.dt);
    }

    double vv = 1.0;
    bool converged = false;

    for (int i = 0; i < config.fire.max_steps && !converged; ++i) {
      double old_energy = state.energy;
      auto [ns, nf] = fire::step(std::move(state), forces, compute, config.fire);
      state = std::move(ns);
      forces = std::move(nf);

      vv = vv * 0.9 + 0.1 * (std::abs(state.energy - old_energy) / state.dt);
      double monitor = std::abs(vv / volume);
      converged = monitor < config.fire.force_tolerance;
      state.converged = converged;

      if (config.log_interval > 0 &&
          ((i + 1) % config.log_interval == 0 || converged)) {
        std::cout << std::format("  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}\n",
                                 i + 1, state.energy, monitor, state.dt);
      }
    }

    arma::vec rho_final = parametrization::to_density(state.x[0], config.param);

    auto s_final = init::from_profile(model, rho_final);
    s_final.species[0].chemical_potential = chemical_potential;
    auto r_final = functionals::total(model, s_final, weights);

    return ConstrainedMinimizationResult{
        .densities = {rho_final},
        .free_energy = r_final.free_energy,
        .grand_potential = r_final.grand_potential,
        .rms_force = state.rms_force,
        .iterations = state.iteration,
        .converged = state.converged,
    };
  }

}  // namespace dft::algorithms

#endif  // DFT_ALGORITHMS_CONSTRAINED_MINIMIZATION_HPP
