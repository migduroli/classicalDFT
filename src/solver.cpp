#include "dft/solver.h"

#include "dft/math/arithmetic.h"

#include <cmath>
#include <stdexcept>

namespace dft {

  // ── Components ──────────────────────────────────────────────────────────

  void Solver::add_species(std::unique_ptr<species::Species> s) {
    species_.push_back(std::move(s));
  }

  void Solver::add_interaction(std::unique_ptr<functional::interaction::Interaction> i) {
    interactions_.push_back(std::move(i));
  }

  void Solver::set_fmt(std::unique_ptr<functional::fmt::FMT> fmt) {
    fmt_ = std::move(fmt);
  }

  // ── Inspectors ──────────────────────────────────────────────────────────

  int Solver::num_species() const noexcept {
    return static_cast<int>(species_.size());
  }

  const species::Species& Solver::species(int i) const {
    return *species_.at(static_cast<size_t>(i));
  }

  species::Species& Solver::species(int i) {
    return *species_.at(static_cast<size_t>(i));
  }

  const density::Density& Solver::density(int i) const {
    return species_.at(static_cast<size_t>(i))->density();
  }

  double Solver::convergence_monitor() const {
    double d = 0.0;
    for (const auto& s : species_) {
      d += s->convergence_monitor();
    }
    return d;
  }

  std::string Solver::fmt_name() const {
    if (!fmt_) {
      return "none";
    }
    return fmt_->name();
  }

  // ── Free energy decomposition ───────────────────────────────────────────

  double Solver::ideal_free_energy() const noexcept {
    return f_ideal_;
  }
  double Solver::hard_sphere_free_energy() const noexcept {
    return f_hard_sphere_;
  }
  double Solver::mean_field_free_energy() const noexcept {
    return f_mean_field_;
  }
  double Solver::external_free_energy() const noexcept {
    return f_external_;
  }

  // ── Core computation ────────────────────────────────────────────────────

  double Solver::compute_free_energy_and_forces(bool excess_only) {
    if (species_.empty()) {
      throw std::runtime_error("Solver: no species added");
    }

    for (auto& s : species_) {
      s->zero_force();
    }
    for (auto& s : species_) {
      s->begin_force_calculation();
    }

    math::arithmetic::CompensatedSum f_total;

    // Ideal gas: F_id = sum_i (rho_i * ln(rho_i) - rho_i) * dV
    f_ideal_ = 0.0;
    if (!excess_only) {
      math::arithmetic::CompensatedSum f_id;
      for (auto& s : species_) {
        const auto& rho = s->density().values();
        double d_v = s->density().cell_volume();
        arma::uword n = s->density().size();
        for (arma::uword pos = 0; pos < n; ++pos) {
          double d0 = rho(pos);
          f_id += (d0 * std::log(d0) - d0) * d_v;
          s->add_to_force(pos, std::log(d0) * d_v);
        }
      }
      f_ideal_ = f_id.sum();
      f_total += f_ideal_;
    }

    // Hard-sphere (FMT) contribution
    f_hard_sphere_ = 0.0;
    if (fmt_) {
      for (auto& s : species_) {
        auto* fmt_s = dynamic_cast<functional::fmt::Species*>(s.get());
        if (fmt_s) {
          f_hard_sphere_ += fmt_s->compute_forces(*fmt_);
        }
      }
      f_total += f_hard_sphere_;
    } else {
      // Without FMT, species still need their FFT computed for interactions
      for (auto& s : species_) {
        s->density().forward_fft();
      }
    }

    // Mean-field interaction contribution
    f_mean_field_ = 0.0;
    for (auto& i : interactions_) {
      f_mean_field_ += i->compute_forces();
    }
    f_total += f_mean_field_;

    // External field + chemical potential
    f_external_ = 0.0;
    if (!excess_only) {
      for (auto& s : species_) {
        f_external_ += s->external_field_energy(true);
      }
      f_total += f_external_;
    }

    for (auto& s : species_) {
      s->end_force_calculation();
    }

    return f_total.sum();
  }

  // ── Bulk thermodynamics ─────────────────────────────────────────────────

  double Solver::chemical_potential(double dens) const {
    return chemical_potential(std::vector<double>(1, dens), 0);
  }

  double Solver::chemical_potential(const std::vector<double>& densities, int species_index) const {
    math::arithmetic::CompensatedSum mu;
    mu += std::log(densities.at(static_cast<size_t>(species_index)));

    if (fmt_) {
      auto* fmt_s =
          dynamic_cast<const functional::fmt::Species*>(species_.at(static_cast<size_t>(species_index)).get());
      if (fmt_s) {
        mu += fmt_->bulk_excess_chemical_potential(densities.at(static_cast<size_t>(species_index)), fmt_s->diameter());
      }
    }

    for (const auto& i : interactions_) {
      mu += i->bulk_chemical_potential(densities.at(static_cast<size_t>(species_index)));
    }

    return mu.sum();
  }

  double Solver::grand_potential_density(double dens) const {
    return grand_potential_density(std::vector<double>(1, dens));
  }

  double Solver::grand_potential_density(const std::vector<double>& densities) const {
    double f = helmholtz_free_energy_density(densities);
    for (int i = 0; i < static_cast<int>(densities.size()); ++i) {
      f -= densities.at(static_cast<size_t>(i)) * chemical_potential(densities, i);
    }
    return f;
  }

  double Solver::helmholtz_free_energy_density(double dens) const {
    return helmholtz_free_energy_density(std::vector<double>(1, dens));
  }

  double Solver::helmholtz_free_energy_density(const std::vector<double>& densities) const {
    double f = 0.0;

    // Ideal gas: rho * ln(rho) - rho
    for (double rho : densities) {
      f += rho * std::log(rho) - rho;
    }

    // Hard-sphere excess
    if (fmt_) {
      for (size_t s = 0; s < densities.size() && s < species_.size(); ++s) {
        auto* fmt_s = dynamic_cast<const functional::fmt::Species*>(species_[s].get());
        if (fmt_s) {
          f += fmt_->bulk_free_energy_density(densities[s], fmt_s->diameter());
        }
      }
    }

    // Mean-field
    for (const auto& i : interactions_) {
      // For single-species self-interaction: 0.5 * a_vdw * rho^2
      f += i->bulk_free_energy_density(densities.at(0), densities.size() > 1 ? densities.at(1) : densities.at(0));
    }

    return f;
  }

  // ── Coexistence utilities ───────────────────────────────────────────────

  static double pressure(const Solver& solver, double rho) {
    return -solver.grand_potential_density(rho);
  }

  static double chempot(const Solver& solver, double rho) {
    return solver.chemical_potential(rho);
  }

  void Solver::find_spinodal(double max_density, double step, double& rho_s1, double& rho_s2, double tol) const {
    // Bracket the pressure maximum (first spinodal)
    double x = 2 * step;
    double p0 = pressure(*this, step);
    double p = pressure(*this, 2 * step);
    double dp = p - p0;

    if (dp < 0) {
      throw std::runtime_error("Solver::find_spinodal: could not bracket pressure maximum");
    }

    while (dp > 0 && x < max_density - step) {
      dp = -p;
      x += step;
      p = pressure(*this, x);
      dp += p;
    }
    if (x >= max_density - step) {
      throw std::runtime_error("Solver::find_spinodal: max_density exceeded while searching for pressure maximum");
    }

    // Golden-section refinement for the maximum
    double a = x - 2 * step;
    double b = x;
    double r = (3.0 - std::sqrt(5.0)) / 2.0;
    double u = a + r * (b - a);
    double v = b - r * (b - a);
    double fu = pressure(*this, u);
    double fv = pressure(*this, v);

    while (b - a > tol) {
      if (fu > fv) {
        b = v;
        v = u;
        fv = fu;
        u = a + r * (b - a);
        fu = pressure(*this, u);
      } else {
        a = u;
        u = v;
        fu = fv;
        v = b - r * (b - a);
        fv = pressure(*this, v);
      }
    }
    rho_s1 = (a + b) / 2.0;

    // Bracket the pressure minimum (second spinodal)
    x = rho_s1;
    p = pressure(*this, x);
    dp = 0.0;
    do {
      dp = -p;
      x += step;
      p = pressure(*this, x);
      dp += p;
    } while (dp < 0 && x < max_density - step);

    if (x >= max_density - step) {
      throw std::runtime_error("Solver::find_spinodal: max_density exceeded while searching for pressure minimum");
    }

    // Golden-section refinement for the minimum
    a = x - 2 * step;
    b = x;
    r = (3.0 - std::sqrt(5.0)) / 2.0;
    u = a + r * (b - a);
    v = b - r * (b - a);
    fu = pressure(*this, u);
    fv = pressure(*this, v);

    while (b - a > tol) {
      if (fu < fv) {
        b = v;
        v = u;
        fv = fu;
        u = a + r * (b - a);
        fu = pressure(*this, u);
      } else {
        a = u;
        u = v;
        fu = fv;
        v = b - r * (b - a);
        fv = pressure(*this, v);
      }
    }
    rho_s2 = (a + b) / 2.0;
  }

  double Solver::find_density_from_chemical_potential(double mu, double rho_min, double rho_max, double tol) const {
    double mu1 = chempot(*this, rho_min);
    double mu2 = chempot(*this, rho_max);

    if (mu1 > mu2) {
      std::swap(rho_min, rho_max);
      std::swap(mu1, mu2);
    }

    if (mu2 < mu || mu1 > mu) {
      throw std::runtime_error("Solver::find_density_from_chemical_potential: mu not bracketed");
    }

    while (mu2 - mu1 > tol) {
      double x = (rho_min + rho_max) / 2.0;
      double mu_x = chempot(*this, x);
      if (mu_x > mu) {
        rho_max = x;
        mu2 = mu_x;
      } else {
        rho_min = x;
        mu1 = mu_x;
      }
    }
    return (rho_min + rho_max) / 2.0;
  }

  void Solver::find_coexistence(double max_density, double step, double& rho_v, double& rho_l, double tol) const {
    double rho_s1 = 0.0;
    double rho_s2 = 0.0;
    find_spinodal(max_density, step, rho_s1, rho_s2, tol);

    // Bracket: start at spinodal vapor density and reduce
    double x_vap = rho_s1;
    double x_liq = find_density_from_chemical_potential(chempot(*this, x_vap), rho_s2, max_density - step, tol);
    double dp1 = pressure(*this, x_vap) - pressure(*this, x_liq);
    double dp2 = dp1;

    do {
      dp2 = dp1;
      x_vap /= 1.1;
      x_liq = find_density_from_chemical_potential(chempot(*this, x_vap), rho_s2, max_density - step, tol);
      dp1 = pressure(*this, x_vap) - pressure(*this, x_liq);
    } while (x_vap > 1e-16 && ((dp1 < 0) == (dp2 < 0)));

    if (x_vap < 1e-16) {
      throw std::runtime_error("Solver::find_coexistence: failed to bracket");
    }

    // Bisection refinement
    double y1 = x_vap;
    double y2 = x_vap * 1.1;

    while (std::abs(y2 - y1) > tol) {
      double y = (y1 + y2) / 2.0;
      x_liq = find_density_from_chemical_potential(chempot(*this, y), rho_s2, max_density - step, tol);
      double dp = pressure(*this, y) - pressure(*this, x_liq);
      if ((dp < 0) == (dp1 < 0)) {
        y1 = y;
        dp1 = dp;
      } else {
        y2 = y;
      }
    }

    rho_v = (y1 + y2) / 2.0;
    rho_l = x_liq;
  }

  // ── Structural properties ───────────────────────────────────────────────

  double Solver::real_space_dcf(double r, double dens) const {
    if (species_.size() != 1) {
      throw std::runtime_error("Solver::real_space_dcf: only implemented for single species");
    }
    (void)r;
    (void)dens;
    // TODO: implement once FMT provides real-space DCF
    return 0.0;
  }

  double Solver::fourier_space_dcf(double k, double dens) const {
    if (species_.size() != 1) {
      throw std::runtime_error("Solver::fourier_space_dcf: only implemented for single species");
    }
    (void)k;
    (void)dens;
    // TODO: implement once FMT provides Fourier-space DCF
    return 0.0;
  }

  // ── Alias support ───────────────────────────────────────────────────────

  void Solver::set_densities_from_aliases(std::vector<arma::vec>& aliases) {
    for (size_t s = 0; s < species_.size(); ++s) {
      species_[s]->set_density_from_alias(aliases.at(s));
    }
  }

  void Solver::convert_forces_to_alias_derivatives(std::vector<arma::vec>& aliases) {
    for (size_t s = 0; s < species_.size(); ++s) {
      aliases[s] = species_[s]->alias_force(aliases.at(s));
    }
  }

  // ── HessianOperator interface ───────────────────────────────────────────

  arma::uword Solver::dimension() const noexcept {
    if (species_.empty()) {
      return 0;
    }
    arma::uword n = 0;
    for (const auto& s : species_) {
      n += s->density().size();
    }
    return n;
  }

  void Solver::hessian_dot_v(const arma::vec& v, arma::vec& result) const {
    if (species_.empty()) {
      throw std::runtime_error("Solver::hessian_dot_v: no species");
    }

    // For single species: H*v = (1/rho) * v * dV  (ideal gas contribution)
    const auto& rho = species_[0]->density().values();
    double d_v = species_[0]->density().cell_volume();
    result = d_v * v / rho;

    // FMT and interaction second derivatives would add here
    // TODO: implement when FMTSpecies::add_second_derivative is available
  }

}  // namespace dft
