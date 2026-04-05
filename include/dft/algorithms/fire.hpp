#ifndef DFT_ALGORITHMS_FIRE_HPP
#define DFT_ALGORITHMS_FIRE_HPP

#include <algorithm>
#include <armadillo>
#include <functional>
#include <stdexcept>
#include <vector>

namespace dft::algorithms::fire {

  // Complete state of a FIRE2 minimizer at one instant.
  // Returned after each step and passed to the next.

  struct FireState {
    std::vector<arma::vec> x;
    std::vector<arma::vec> v;
    double dt;
    double alpha;
    int n_positive{ 0 };
    int n_negative{ 0 };
    double energy{ 0.0 };
    double rms_force{ 0.0 };
    int iteration{ 0 };
    bool converged{ false };
  };

  // Force function: given positions x, returns (energy, forces).
  // Forces are negative gradients: f = -grad(E).

  using ForceFunction = std::function<std::pair<double, std::vector<arma::vec>>(const std::vector<arma::vec>&)>;

  struct Fire {
    double dt{ 1e-3 };
    double dt_max{ 1e-2 };
    double dt_min{ 1e-8 };
    double alpha_start{ 0.1 };
    double f_inc{ 1.1 };
    double f_dec{ 0.5 };
    double f_alpha{ 0.99 };
    int n_delay{ 5 };
    int max_uphill{ 20 };
    double force_tolerance{ 0.1 };
    int max_steps{ 10000 };

    // Initialize a FIRE state from starting positions.

    [[nodiscard]] auto initialize(std::vector<arma::vec> x0, const ForceFunction& compute) const -> FireState;

    // Perform one FIRE2 step. Returns the updated state and forces.

    [[nodiscard]] auto step(FireState state, const std::vector<arma::vec>& forces, const ForceFunction& compute) const
        -> std::pair<FireState, std::vector<arma::vec>>;

    // Run FIRE2 to convergence or max_steps.

    [[nodiscard]] auto minimize(std::vector<arma::vec> x0, const ForceFunction& compute) const -> FireState;
  };

  [[nodiscard]] inline auto Fire::initialize(std::vector<arma::vec> x0, const ForceFunction& compute) const
      -> FireState {
    auto [energy, forces] = compute(x0);

    double total_dof = 0.0;
    double sum_f2 = 0.0;
    std::vector<arma::vec> v(x0.size());
    for (std::size_t s = 0; s < x0.size(); ++s) {
      v[s] = arma::zeros(x0[s].n_elem);
      sum_f2 += arma::dot(forces[s], forces[s]);
      total_dof += static_cast<double>(forces[s].n_elem);
    }

    double rms = std::sqrt(sum_f2 / total_dof);

    return FireState{
      .x = std::move(x0),
      .v = std::move(v),
      .dt = dt,
      .alpha = alpha_start,
      .energy = energy,
      .rms_force = rms,
      .converged = rms < force_tolerance,
    };
  }

  [[nodiscard]] inline auto
  Fire::step(FireState state, const std::vector<arma::vec>& forces, const ForceFunction& compute) const
      -> std::pair<FireState, std::vector<arma::vec>> {
    // Power: P = v . f  (f is negative gradient, so P > 0 means downhill)
    double power = 0.0;
    for (std::size_t s = 0; s < state.x.size(); ++s) {
      power += arma::dot(state.v[s], forces[s]);
    }

    if (power > 0.0 || state.iteration == 0) {
      state.n_positive++;
      state.n_negative = 0;
      if (state.n_positive > n_delay) {
        state.dt = std::min(state.dt * f_inc, dt_max);
        state.alpha *= f_alpha;
      }
    } else {
      state.n_positive = 0;
      state.n_negative++;

      if (state.n_negative > max_uphill) {
        throw std::runtime_error("fire::step: exceeded max uphill steps");
      }

      // Backtrack position and kill velocity FIRST (FIRE2 convention:
      // undo half-step with current dt, then reduce dt).
      for (std::size_t s = 0; s < state.x.size(); ++s) {
        state.x[s] -= 0.5 * state.dt * state.v[s];
        state.v[s].zeros();
      }

      if (state.iteration > n_delay) {
        state.dt = std::max(state.dt * f_dec, dt_min);
        state.alpha = alpha_start;
      }
    }

    // Semi-implicit Euler: update velocity, then mix, then position.
    for (std::size_t s = 0; s < state.x.size(); ++s) {
      state.v[s] += state.dt * forces[s];
    }

    double v_norm = 0.0;
    double f_norm = 0.0;
    for (std::size_t s = 0; s < state.x.size(); ++s) {
      v_norm += arma::dot(state.v[s], state.v[s]);
      f_norm += arma::dot(forces[s], forces[s]);
    }
    v_norm = std::sqrt(v_norm);
    f_norm = std::sqrt(f_norm);

    // FIRE mixing: v = (1 - alpha) * v + alpha * |v|/|f| * f
    if (f_norm > 0.0) {
      double ratio = v_norm / f_norm;
      for (std::size_t s = 0; s < state.x.size(); ++s) {
        state.v[s] = (1.0 - state.alpha) * state.v[s] + state.alpha * ratio * forces[s];
      }
    }

    // Update positions
    for (std::size_t s = 0; s < state.x.size(); ++s) {
      state.x[s] += state.dt * state.v[s];
    }

    // Recompute forces at new position
    auto [energy, new_forces] = compute(state.x);
    state.energy = energy;

    double total_dof = 0.0;
    double sum_f2 = 0.0;
    for (std::size_t s = 0; s < state.x.size(); ++s) {
      sum_f2 += arma::dot(new_forces[s], new_forces[s]);
      total_dof += static_cast<double>(new_forces[s].n_elem);
    }
    state.rms_force = std::sqrt(sum_f2 / total_dof);
    state.iteration++;
    state.converged = state.rms_force < force_tolerance;

    return { std::move(state), std::move(new_forces) };
  }

  [[nodiscard]] inline auto Fire::minimize(std::vector<arma::vec> x0, const ForceFunction& compute) const -> FireState {
    auto state = initialize(std::move(x0), compute);
    auto [energy, forces] = compute(state.x);

    for (int i = 0; i < max_steps && !state.converged; ++i) {
      auto [new_state, new_forces] = step(std::move(state), forces, compute);
      state = std::move(new_state);
      forces = std::move(new_forces);
    }

    return state;
  }

}  // namespace dft::algorithms::fire

#endif  // DFT_ALGORITHMS_FIRE_HPP
