#include "dft/dynamics/fire2.h"

#include "dft/math/arithmetic.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dft::dynamics {

  Fire2Minimizer::Fire2Minimizer(Solver& solver, Fire2Config config)
      : Minimizer(solver, config.force_limit, config.min_density),
        config_(config),
        dt_(config.dt),
        alpha_(config.alpha_start),
        dt_best_(config.dt) {
    // Initialize velocities to zero
    int n = solver.num_species();
    velocities_.resize(static_cast<size_t>(n));
    for (int s = 0; s < n; ++s) {
      velocities_[static_cast<size_t>(s)] = arma::zeros(solver.density(s).size());
    }

    // Compute initial forces
    (void)compute_energy_and_forces();
  }

  void Fire2Minimizer::do_reset() {
    dt_ = config_.dt;
    alpha_ = config_.alpha_start;
    n_positive_ = 0;
    n_negative_ = 0;
    backtracks_ = 0;
    dt_best_ = config_.dt;
    rms_force_ = 0.0;
    f_max_ = 0.0;
    iteration_ = 0;

    int n = mutable_solver().num_species();
    velocities_.resize(static_cast<size_t>(n));
    for (int s = 0; s < n; ++s) {
      velocities_[static_cast<size_t>(s)] = arma::zeros(mutable_solver().density(s).size());
    }

    (void)compute_energy_and_forces();
  }

  double Fire2Minimizer::do_step() {
    ++iteration_;

    int begin = 0;
    int end = mutable_solver().num_species();

    // Compute power: P = v · (-dF/dx) = -v · dF/dx
    // The solver stores dF/dx (gradient, not force), so P = -v · dF/dx
    math::arithmetic::CompensatedSum power;
    for (int s = begin; s < end; ++s) {
      auto si = static_cast<size_t>(s);
      const arma::vec& df = mutable_solver().species(s).force();
      power += -arma::dot(velocities_[si], df);
    }
    double p = power.sum();

    // Save state for potential backtracking
    std::vector<arma::vec> x_saved(static_cast<size_t>(end - begin));
    std::vector<arma::vec> v_saved(static_cast<size_t>(end - begin));
    std::vector<arma::vec> df_saved(static_cast<size_t>(end - begin));
    for (int s = begin; s < end; ++s) {
      auto si = static_cast<size_t>(s);
      x_saved[si] = aliases()[si];
      v_saved[si] = velocities_[si];
      df_saved[si] = mutable_solver().species(s).force();
    }

    if (p > 0) {
      // Moving downhill
      ++n_positive_;
      n_negative_ = 0;
      if (n_positive_ > config_.n_delay) {
        dt_ = std::min(dt_ * config_.f_inc, config_.dt_max);
        alpha_ *= config_.f_alf;
      }
    } else {
      // Moving uphill
      n_positive_ = 0;
      ++n_negative_;

      if (n_negative_ > config_.max_uphill_steps) {
        throw std::runtime_error("Fire2Minimizer: too many consecutive uphill steps");
      }

      // Decrease timestep and reset damping (unless in initial delay period)
      if (!(iteration_ <= config_.n_delay)) {
        if (dt_ * config_.f_dec >= config_.dt_min) {
          dt_ *= config_.f_dec;
        }
        alpha_ = config_.alpha_start;
      }

      // Backtrack half a step and damp velocities
      for (int s = begin; s < end; ++s) {
        auto si = static_cast<size_t>(s);
        aliases()[si] -= 0.5 * dt_ * velocities_[si];
        velocities_[si] *= 0.1;
      }
      ++backtracks_;
    }

    double step_energy = 0.0;
    try {
      step_energy = semi_implicit_euler(begin, end);
      dt_best_ = std::max(dt_, dt_best_);
    } catch (const std::exception&) {
      // Restore state and reduce timestep
      for (int s = begin; s < end; ++s) {
        auto si = static_cast<size_t>(s);
        aliases()[si] = x_saved[si];
        velocities_[si] = v_saved[si] * 0.5;
      }
      dt_ /= 2.0;
      f_max_ = 1000.0;
      ++backtracks_;
      step_energy = compute_energy_and_forces();
    }

    // Adapt dt_max after repeated backtracks
    if (backtracks_ >= 10) {
      config_.dt_max = std::min(dt_best_, 0.9 * config_.dt_max);
      dt_best_ = 0.0;
      backtracks_ = 0;
    }

    return step_energy;
  }

  double Fire2Minimizer::semi_implicit_euler(int begin, int end) {
    // Update velocities and compute norms for FIRE mixing
    double vnorm_sq = 0.0;
    double fnorm_sq = 0.0;
    arma::uword total_dof = 0;

    for (int s = begin; s < end; ++s) {
      auto si = static_cast<size_t>(s);
      const arma::vec& df = mutable_solver().species(s).force();

      // v += -dt * dF/dx  (gradient descent: force = -gradient)
      velocities_[si] -= dt_ * df;

      double v = arma::norm(velocities_[si]);
      double f = arma::norm(df);
      vnorm_sq += v * v;
      fnorm_sq += f * f;
      total_dof += df.n_elem;
    }

    rms_force_ = std::sqrt(fnorm_sq / static_cast<double>(total_dof));

    // FIRE mixing: v = (1 - alpha) * v - alpha * |v|/|F| * dF/dx
    double ratio = (fnorm_sq > 0.0) ? std::sqrt(vnorm_sq / fnorm_sq) : 0.0;
    for (int s = begin; s < end; ++s) {
      auto si = static_cast<size_t>(s);
      velocities_[si] *= (1.0 - alpha_);
      velocities_[si] -= alpha_ * ratio * mutable_solver().species(s).force();

      // Update positions: x += dt * v
      aliases()[si] += dt_ * velocities_[si];
    }

    // Recompute forces at new position
    double e = compute_energy_and_forces();

    // Update force statistics
    f_max_ = 0.0;
    fnorm_sq = 0.0;
    total_dof = 0;
    for (int s = begin; s < end; ++s) {
      const arma::vec& df = mutable_solver().species(s).force();
      double d_v = mutable_solver().density(s).cell_volume();
      double inf = arma::max(arma::abs(df)) / d_v;
      f_max_ = std::max(f_max_, inf);
      double f = arma::norm(df);
      fnorm_sq += f * f;
      total_dof += df.n_elem;
    }
    rms_force_ = std::sqrt(fnorm_sq / static_cast<double>(total_dof));

    return e;
  }

}  // namespace dft::dynamics
