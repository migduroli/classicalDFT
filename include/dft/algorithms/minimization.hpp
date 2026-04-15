#ifndef DFT_ALGORITHMS_MINIMIZATION_HPP
#define DFT_ALGORITHMS_MINIMIZATION_HPP

#include "dft/algorithms/fire.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/grid.hpp"
#include "dft/init.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <cmath>
#include <format>
#include <functional>
#include <iostream>
#include <print>
#include <stdexcept>
#include <variant>
#include <vector>

namespace dft::algorithms::minimization {

  // Density parametrizations map unconstrained variables x in R
  // to physical densities rho > 0, guaranteeing positivity.
  // The chain rule transforms forces accordingly: f_x = f_rho * drho/dx.

  // Unbounded: rho = rho_min + x^2.
  // Maps R -> (rho_min, inf).

  struct Unbounded {
    double rho_min{1e-18};
  };

  // Bounded: rho = rho_min + range * x^2 / (1 + x^2).
  // Maps R -> (rho_min, rho_max).

  struct Bounded {
    double rho_min{1e-18};
    double rho_max{1.0};
  };

  using Parametrization = std::variant<Unbounded, Bounded>;

  namespace detail {

    // Convert parameters to density: rho(x).

    [[nodiscard]] inline auto to_density(const arma::vec& x, const Parametrization& p) -> arma::vec {
      return std::visit(
          [&](const auto& t) -> arma::vec {
            using T = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<T, Unbounded>) {
              return t.rho_min + x % x;
            } else {
              double range = t.rho_max - t.rho_min;
              arma::vec x2 = x % x;
              return t.rho_min + range * x2 / (1.0 + x2);
            }
          },
          p
      );
    }

    // Convert density to parameters: x(rho).

    [[nodiscard]] inline auto from_density(const arma::vec& rho, const Parametrization& p) -> arma::vec {
      return std::visit(
          [&](const auto& t) -> arma::vec {
            using T = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<T, Unbounded>) {
              return arma::sqrt(arma::clamp(rho - t.rho_min, 0.0, arma::datum::inf));
            } else {
              double range = t.rho_max - t.rho_min;
              arma::vec delta = arma::clamp(rho - t.rho_min, 0.0, range * (1.0 - 1e-14));
              return arma::sqrt(delta / (range - delta));
            }
          },
          p
      );
    }

    // Transform forces from density space to parameter space:
    //   f_x = f_rho * drho/dx.

    [[nodiscard]] inline auto transform_force(const arma::vec& f_rho, const arma::vec& x, const Parametrization& p)
        -> arma::vec {
      return std::visit(
          [&](const auto& t) -> arma::vec {
            using T = std::decay_t<decltype(t)>;
            if constexpr (std::is_same_v<T, Unbounded>) {
              return f_rho % (2.0 * x);
            } else {
              double range = t.rho_max - t.rho_min;
              arma::vec x2 = x % x;
              arma::vec denom = (1.0 + x2) % (1.0 + x2);
              return f_rho % (2.0 * range * x / denom);
            }
          },
          p
      );
    }

    // Convergence monitor: exponential moving average of |dE/dt|
    // normalized by the system volume.
    //   vv = 0.9 * vv + 0.1 * |dE/dt|
    //   monitor = |vv / volume|

    [[nodiscard]] inline auto
    convergence_monitor(double vv, double old_energy, double new_energy, double dt, double volume)
        -> std::pair<double, double> {
      double new_vv = vv * 0.9 + 0.1 * (std::abs(new_energy - old_energy) / dt);
      return {new_vv, std::abs(new_vv / volume)};
    }

    [[nodiscard]] inline auto
    rescale_to_target_mass(const arma::vec& density, double target_mass, double cell_volume, double min_density = 0.0)
        -> arma::vec {
      double mass = arma::accu(density) * cell_volume;
      if (mass <= 0.0) {
        return arma::clamp(density, min_density, arma::datum::inf);
      }
      return arma::clamp(density * (target_mass / mass), min_density, arma::datum::inf);
    }

  } // namespace detail

  struct Result {
    std::vector<arma::vec> densities;
    double free_energy{0.0};
    double grand_potential{0.0};
    double rms_force{0.0};
    int iterations{0};
    bool converged{false};
  };

  struct Minimizer {
    fire::Fire fire{};
    Parametrization param = Unbounded{};
    bool use_homogeneous_boundary{true};
    int log_interval{100};

    // Minimize the Helmholtz free energy F[rho] at fixed total mass N.

    [[nodiscard]] auto fixed_mass(
        const physics::Model& model,
        const functionals::Weights& weights,
        const arma::vec& initial_density,
        double chemical_potential,
        double target_mass
    ) const -> Result;

    [[nodiscard]] auto fixed_mass(
        const physics::Model& model,
        const functionals::Weights& weights,
        const arma::vec& initial_density,
        double chemical_potential,
        double target_mass,
        const arma::vec& external_field
    ) const -> Result;

    // Minimize the Helmholtz free energy at fixed excess mass relative
    // to a background density profile rho_bg. This preserves wall-induced
    // layering while constraining only the droplet excess:
    //   N_ex = integral (rho - rho_bg) dV.

    [[nodiscard]] auto fixed_excess_mass(
        const physics::Model& model,
        const functionals::Weights& weights,
        const arma::vec& initial_density,
        double chemical_potential,
        const arma::vec& background_density,
        double target_excess_mass
    ) const -> Result;

    [[nodiscard]] auto fixed_excess_mass(
        const physics::Model& model,
        const functionals::Weights& weights,
        const arma::vec& initial_density,
        double chemical_potential,
        const arma::vec& background_density,
        double target_excess_mass,
        const arma::vec& external_field
    ) const -> Result;

    // Minimize the grand potential Omega[rho] in the grand-canonical ensemble.

    [[nodiscard]] auto grand_potential(
        const physics::Model& model,
        const functionals::Weights& weights,
        const arma::vec& initial_density,
        double chemical_potential
    ) const -> Result;

    [[nodiscard]] auto grand_potential(
        const physics::Model& model,
        const functionals::Weights& weights,
        const arma::vec& initial_density,
        double chemical_potential,
        const arma::vec& external_field
    ) const -> Result;
  };

  // Build a Picard constraint that fixes the total mass of each species
  // relative to a background density:
  //
  //   N_ex = integral (rho - rho_bg) dV = target_mass
  //
  // After each iteration, the excess density (rho - rho_bg) is
  // rescaled so the integral matches the target.

  using Constraint = std::function<std::vector<arma::vec>(const std::vector<arma::vec>&)>;

  [[nodiscard]] inline auto fixed_mass_constraint(
      std::vector<double> target_masses,
      std::vector<arma::vec> backgrounds,
      double cell_volume,
      double min_density = 1e-30
  ) -> Constraint {
    return [target_masses = std::move(target_masses), backgrounds = std::move(backgrounds), cell_volume, min_density](
               const std::vector<arma::vec>& densities
           ) -> std::vector<arma::vec> {
      auto result = densities;
      for (std::size_t s = 0; s < result.size() && s < target_masses.size(); ++s) {
        arma::vec excess = result[s] - backgrounds[s];
        double current_mass = arma::accu(excess) * cell_volume;
        if (std::abs(current_mass) > 1e-30) {
          double scale = target_masses[s] / current_mass;
          result[s] = backgrounds[s] + scale * excess;
          result[s] = arma::clamp(result[s], min_density, arma::datum::inf);
        }
      }
      return result;
    };
  }

  [[nodiscard]] inline auto Minimizer::fixed_mass(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      double target_mass
  ) const -> Result {
    return fixed_mass(model, weights, initial_density, chemical_potential, target_mass, {});
  }

  [[nodiscard]] inline auto Minimizer::fixed_mass(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      double target_mass,
      const arma::vec& external_field
  ) const -> Result {
    double dv = model.grid.cell_volume();
    arma::uvec bdry = model.grid.boundary_mask();

    if (!external_field.is_empty() && external_field.n_elem != initial_density.n_elem) {
      throw std::invalid_argument("External field size must match density size");
    }

    auto make_state = [&](const arma::vec& rho) {
      auto state = init::from_profile(model, rho);
      if (!external_field.is_empty()) {
        state.species[0].density.external_field = external_field;
      }
      return state;
    };

    auto compute = [&](const std::vector<arma::vec>& x_param) -> std::pair<double, std::vector<arma::vec>> {
      arma::vec rho = detail::to_density(x_param[0], param);
      if (!rho.is_finite()) {
        throw std::runtime_error("Minimizer::fixed_mass: non-finite density encountered");
      }

      // Rescale to target mass.
      rho = detail::rescale_to_target_mass(rho, target_mass, dv);

      // Evaluate Helmholtz free energy and gradient.
      // Chemical potential is set to zero so that forces are dF/drho * dV.
      // The Lagrange multiplier for fixed mass acts as the chemical
      // potential and is determined self-consistently.
      auto state = make_state(rho);
      state.species[0].chemical_potential = 0.0;
      auto result = functionals::total(model, state, weights);
      if (!std::isfinite(result.free_energy) || !std::isfinite(result.grand_potential)) {
        throw std::runtime_error("Minimizer::fixed_mass: non-finite functional value encountered");
      }
      arma::vec grad = result.forces[0];
      if (!grad.is_finite()) {
        throw std::runtime_error("Minimizer::fixed_mass: non-finite gradient encountered");
      }

      // Boundary conditions.
      if (use_homogeneous_boundary) {
        grad = homogeneous_boundary(grad, bdry);
      }

      // Lagrange multiplier for fixed mass.
      double lambda = arma::dot(grad, rho) / target_mass;
      grad -= lambda * dv;

      // Convert to parameter space and negate for FIRE.
      arma::vec f = detail::transform_force(grad, x_param[0], param);
      if (!f.is_finite()) {
        throw std::runtime_error("Minimizer::fixed_mass: non-finite transformed force encountered");
      }
      return {result.grand_potential, {-f}};
    };

    arma::vec x0 = detail::from_density(initial_density, param);

    if (log_interval > 0) {
      std::println(std::cout, "  {:>6s}  {:>14s}  {:>14s}  {:>10s}", "iter", "F", "monitor", "dt");
      std::println(std::cout, "  {}", std::string(50, '-'));
    }

    double volume = static_cast<double>(initial_density.n_elem) * dv;
    auto state = fire.initialize({x0}, compute);
    auto [e0, forces] = compute(state.x);

    if (log_interval > 0) {
      std::println(std::cout, "  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}", 0, state.energy, state.rms_force, state.dt);
    }

    double vv = 1.0;
    bool converged = false;

    for (int i = 0; i < fire.max_steps && !converged; ++i) {
      double old_energy = state.energy;
      auto [ns, nf] = fire.step(std::move(state), forces, compute);
      state = std::move(ns);
      forces = std::move(nf);

      if (!std::isfinite(state.energy) || !std::isfinite(state.rms_force)) {
        throw std::runtime_error("Minimizer::fixed_mass: non-finite FIRE state encountered");
      }

      auto [new_vv, monitor] = detail::convergence_monitor(vv, old_energy, state.energy, state.dt, volume);
      if (!std::isfinite(new_vv) || !std::isfinite(monitor)) {
        throw std::runtime_error("Minimizer::fixed_mass: non-finite convergence monitor encountered");
      }
      vv = new_vv;
      converged = monitor < fire.force_tolerance;
      state.converged = converged;

      if (log_interval > 0 && ((i + 1) % log_interval == 0 || converged)) {
        std::println(std::cout, "  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}", i + 1, state.energy, monitor, state.dt);
      }
    }

    // Extract final density.
    arma::vec rho_final = detail::to_density(state.x[0], param);
    rho_final = detail::rescale_to_target_mass(rho_final, target_mass, dv);

    auto s_final = make_state(rho_final);
    s_final.species[0].chemical_potential = chemical_potential;
    auto r_final = functionals::total(model, s_final, weights);

    return Result{
        .densities = {rho_final},
        .free_energy = r_final.free_energy,
        .grand_potential = r_final.grand_potential,
        .rms_force = state.rms_force,
        .iterations = state.iteration,
        .converged = state.converged,
    };
  }

  [[nodiscard]] inline auto Minimizer::fixed_excess_mass(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      const arma::vec& background_density,
      double target_excess_mass
  ) const -> Result {
    return fixed_excess_mass(
        model,
        weights,
        initial_density,
        chemical_potential,
        background_density,
        target_excess_mass,
        {}
    );
  }

  [[nodiscard]] inline auto Minimizer::fixed_excess_mass(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      const arma::vec& background_density,
      double target_excess_mass,
      const arma::vec& external_field
  ) const -> Result {
    double dv = model.grid.cell_volume();
    arma::uvec bdry = model.grid.boundary_mask();

    if (!(target_excess_mass > 0.0)) {
      throw std::invalid_argument("Target excess mass must be positive");
    }

    if (background_density.n_elem != initial_density.n_elem) {
      throw std::invalid_argument("Background density size must match initial density size");
    }
    if (!external_field.is_empty() && external_field.n_elem != initial_density.n_elem) {
      throw std::invalid_argument("External field size must match density size");
    }

    auto make_state = [&](const arma::vec& rho) {
      auto state = init::from_profile(model, rho);
      if (!external_field.is_empty()) {
        state.species[0].density.external_field = external_field;
      }
      return state;
    };

    auto compute = [&](const std::vector<arma::vec>& x_param) -> std::pair<double, std::vector<arma::vec>> {
      arma::vec excess = detail::to_density(x_param[0], param);
      if (!excess.is_finite()) {
        throw std::runtime_error("Minimizer::fixed_excess_mass: non-finite excess density encountered");
      }

      excess = detail::rescale_to_target_mass(excess, target_excess_mass, dv);
      double excess_mass = arma::accu(excess) * dv;
      if (!(excess_mass > 0.0)) {
        throw std::runtime_error("Minimizer::fixed_excess_mass: non-positive excess mass encountered");
      }

      arma::vec rho = background_density + excess;
      if (!rho.is_finite()) {
        throw std::runtime_error("Minimizer::fixed_excess_mass: non-finite density encountered");
      }

      auto state = make_state(rho);
      state.species[0].chemical_potential = 0.0;
      auto result = functionals::total(model, state, weights);
      if (!std::isfinite(result.free_energy) || !std::isfinite(result.grand_potential)) {
        throw std::runtime_error("Minimizer::fixed_excess_mass: non-finite functional value encountered");
      }
      arma::vec grad = result.forces[0];
      if (!grad.is_finite()) {
        throw std::runtime_error("Minimizer::fixed_excess_mass: non-finite gradient encountered");
      }

      if (use_homogeneous_boundary) {
        grad = homogeneous_boundary(grad, bdry);
      }

      double lambda = arma::dot(grad, excess) / target_excess_mass;
      grad -= lambda * dv;

      arma::vec f = detail::transform_force(grad, x_param[0], param);
      if (!f.is_finite()) {
        throw std::runtime_error("Minimizer::fixed_excess_mass: non-finite transformed force encountered");
      }
      return {result.free_energy, {-f}};
    };

    arma::vec excess0 = arma::clamp(initial_density - background_density, 0.0, arma::datum::inf);
    excess0 = detail::rescale_to_target_mass(excess0, target_excess_mass, dv);
    arma::vec x0 = detail::from_density(excess0, param);

    if (log_interval > 0) {
      std::println(std::cout, "  {:>6s}  {:>14s}  {:>14s}  {:>10s}", "iter", "F", "monitor", "dt");
      std::println(std::cout, "  {}", std::string(50, '-'));
    }

    double volume = static_cast<double>(initial_density.n_elem) * dv;
    auto state = fire.initialize({x0}, compute);
    auto [e0, forces] = compute(state.x);

    if (log_interval > 0) {
      std::println(std::cout, "  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}", 0, state.energy, state.rms_force, state.dt);
    }

    double vv = 1.0;
    bool converged = false;

    for (int i = 0; i < fire.max_steps && !converged; ++i) {
      double old_energy = state.energy;
      auto [ns, nf] = fire.step(std::move(state), forces, compute);
      state = std::move(ns);
      forces = std::move(nf);

      if (!std::isfinite(state.energy) || !std::isfinite(state.rms_force)) {
        throw std::runtime_error("Minimizer::fixed_excess_mass: non-finite FIRE state encountered");
      }

      auto [new_vv, monitor] = detail::convergence_monitor(vv, old_energy, state.energy, state.dt, volume);
      if (!std::isfinite(new_vv) || !std::isfinite(monitor)) {
        throw std::runtime_error("Minimizer::fixed_excess_mass: non-finite convergence monitor encountered");
      }
      vv = new_vv;
      converged = monitor < fire.force_tolerance;
      state.converged = converged;

      if (log_interval > 0 && ((i + 1) % log_interval == 0 || converged)) {
        std::println(std::cout, "  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}", i + 1, state.energy, monitor, state.dt);
      }
    }

    arma::vec excess_final = detail::to_density(state.x[0], param);
    excess_final = detail::rescale_to_target_mass(excess_final, target_excess_mass, dv);
    arma::vec rho_final = background_density + excess_final;

    auto s_final = make_state(rho_final);
    s_final.species[0].chemical_potential = chemical_potential;
    auto r_final = functionals::total(model, s_final, weights);

    return Result{
        .densities = {rho_final},
        .free_energy = r_final.free_energy,
        .grand_potential = r_final.grand_potential,
        .rms_force = state.rms_force,
        .iterations = state.iteration,
        .converged = state.converged,
    };
  }

  [[nodiscard]] inline auto Minimizer::grand_potential(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential
  ) const -> Result {
    return grand_potential(model, weights, initial_density, chemical_potential, {});
  }

  [[nodiscard]] inline auto Minimizer::grand_potential(
      const physics::Model& model,
      const functionals::Weights& weights,
      const arma::vec& initial_density,
      double chemical_potential,
      const arma::vec& external_field
  ) const -> Result {
    arma::uvec bdry = model.grid.boundary_mask();

    if (!external_field.is_empty() && external_field.n_elem != initial_density.n_elem) {
      throw std::invalid_argument("External field size must match density size");
    }

    auto make_state = [&](const arma::vec& rho) {
      auto state = init::from_profile(model, rho);
      if (!external_field.is_empty()) {
        state.species[0].density.external_field = external_field;
      }
      return state;
    };

    auto compute = [&](const std::vector<arma::vec>& x_param) -> std::pair<double, std::vector<arma::vec>> {
      arma::vec rho = detail::to_density(x_param[0], param);
      if (!rho.is_finite()) {
        throw std::runtime_error("Minimizer::grand_potential: non-finite density encountered");
      }

      auto state = make_state(rho);
      state.species[0].chemical_potential = chemical_potential;
      auto result = functionals::total(model, state, weights);
      if (!std::isfinite(result.free_energy) || !std::isfinite(result.grand_potential)) {
        throw std::runtime_error("Minimizer::grand_potential: non-finite functional value encountered");
      }
      arma::vec grad = result.forces[0];
      if (!grad.is_finite()) {
        throw std::runtime_error("Minimizer::grand_potential: non-finite gradient encountered");
      }

      if (use_homogeneous_boundary) {
        grad = homogeneous_boundary(grad, bdry);
      }

      arma::vec f = detail::transform_force(grad, x_param[0], param);
      if (!f.is_finite()) {
        throw std::runtime_error("Minimizer::grand_potential: non-finite transformed force encountered");
      }
      return {result.grand_potential, {-f}};
    };

    arma::vec x0 = detail::from_density(initial_density, param);

    if (log_interval > 0) {
      std::println(std::cout, "  {:>6s}  {:>14s}  {:>14s}  {:>10s}", "iter", "Omega", "monitor", "dt");
      std::println(std::cout, "  {}", std::string(50, '-'));
    }

    double volume = static_cast<double>(initial_density.n_elem) * model.grid.cell_volume();
    auto state = fire.initialize({x0}, compute);
    auto [e0, forces] = compute(state.x);

    if (log_interval > 0) {
      std::println(std::cout, "  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}", 0, state.energy, state.rms_force, state.dt);
    }

    double vv = 1.0;
    bool converged = false;

    for (int i = 0; i < fire.max_steps && !converged; ++i) {
      double old_energy = state.energy;
      auto [ns, nf] = fire.step(std::move(state), forces, compute);
      state = std::move(ns);
      forces = std::move(nf);

      if (!std::isfinite(state.energy) || !std::isfinite(state.rms_force)) {
        throw std::runtime_error("Minimizer::grand_potential: non-finite FIRE state encountered");
      }

      auto [new_vv, monitor] = detail::convergence_monitor(vv, old_energy, state.energy, state.dt, volume);
      if (!std::isfinite(new_vv) || !std::isfinite(monitor)) {
        throw std::runtime_error("Minimizer::grand_potential: non-finite convergence monitor encountered");
      }
      vv = new_vv;
      converged = monitor < fire.force_tolerance;
      state.converged = converged;

      if (log_interval > 0 && ((i + 1) % log_interval == 0 || converged)) {
        std::println(std::cout, "  {:>6d}  {:>14.6f}  {:>14.6e}  {:>10.2e}", i + 1, state.energy, monitor, state.dt);
      }
    }

    arma::vec rho_final = detail::to_density(state.x[0], param);

    auto s_final = make_state(rho_final);
    s_final.species[0].chemical_potential = chemical_potential;
    auto r_final = functionals::total(model, s_final, weights);

    return Result{
        .densities = {rho_final},
        .free_energy = r_final.free_energy,
        .grand_potential = r_final.grand_potential,
        .rms_force = state.rms_force,
        .iterations = state.iteration,
        .converged = state.converged,
    };
  }

} // namespace dft::algorithms::minimization

#endif // DFT_ALGORITHMS_MINIMIZATION_HPP
