#include "dft/dynamics/minimizer.h"

#include <cmath>
#include <stdexcept>

namespace dft::dynamics {

  Minimizer::Minimizer(Solver& solver, double force_limit, double min_density)
      : solver_(solver), force_limit_(force_limit), min_density_(min_density) {
    initialize_aliases();
  }

  // ── Public interface ──────────────────────────────────────────────────────

  bool Minimizer::run(long max_steps) {
    step_count_ = 0;
    initialize_aliases();
    do_reset();
    return resume(max_steps);
  }

  bool Minimizer::resume(long max_steps) {
    bool converged = false;
    long steps_taken = 0;

    while (true) {
      energy_ = do_step();
      ++step_count_;
      ++steps_taken;

      max_force_ = convergence_monitor();

      // Invoke user callback if set
      if (step_callback_ && !step_callback_(step_count_, energy_, max_force_)) {
        break;
      }

      if (max_force_ < force_limit_) {
        converged = true;
        break;
      }

      if (max_steps > 0 && steps_taken >= max_steps) {
        break;
      }

      // Check minimum density
      double volume =
          solver_.density(0).box_size()(0) * solver_.density(0).box_size()(1) * solver_.density(0).box_size()(2);
      double n_total = solver_.density(0).number_of_atoms();
      if (n_total / volume < min_density_) {
        break;
      }
    }

    return converged;
  }

  void Minimizer::reset() {
    initialize_aliases();
    do_reset();
  }

  // ── Fixed direction ───────────────────────────────────────────────────────

  void Minimizer::set_fixed_direction(const arma::vec& direction) {
    if (solver_.num_species() > 0 && direction.n_elem != solver_.density(0).size()) {
      throw std::invalid_argument("Minimizer: fixed direction size does not match density size");
    }
    // Normalize
    double norm = arma::norm(direction);
    if (norm < 1e-30) {
      throw std::invalid_argument("Minimizer: fixed direction has zero norm");
    }
    fixed_direction_ = direction / norm;
  }

  // ── Protected helpers ─────────────────────────────────────────────────────

  double Minimizer::compute_density_forces() {
    solver_.set_densities_from_aliases(aliases_);
    double f = solver_.compute_free_energy_and_forces();

    // Project forces orthogonal to the fixed direction, if set
    if (fixed_direction_.has_value()) {
      const auto& d = fixed_direction_.value();
      for (int s = 0; s < solver_.num_species(); ++s) {
        arma::vec& force = solver_.species(s).force();
        double proj = 2.0 * arma::dot(d, force);
        force -= proj * d;
      }
    }

    return f;
  }

  double Minimizer::compute_energy_and_forces() {
    double f = compute_density_forces();

    // Convert forces from density-space dΩ/dρ to alias-space dΩ/dx.
    // Chain rule: dΩ/dx_i = 2*x_i * dΩ/dρ_i  (since ρ = ρ_min + x²).
    // This modifies species' force vectors in-place, leaving aliases_ untouched.
    for (int s = 0; s < solver_.num_species(); ++s) {
      arma::vec& force = solver_.species(s).force();
      force = 2.0 * aliases_[static_cast<size_t>(s)] % force;
    }

    return f;
  }

  // ── Private helpers ───────────────────────────────────────────────────────

  void Minimizer::initialize_aliases() {
    int n = solver_.num_species();
    aliases_.resize(static_cast<size_t>(n));
    for (int s = 0; s < n; ++s) {
      aliases_[static_cast<size_t>(s)] = solver_.species(s).density_alias();
    }
  }

  double Minimizer::convergence_monitor() const {
    return solver_.convergence_monitor();
  }

}  // namespace dft::dynamics
