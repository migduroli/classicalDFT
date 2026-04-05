#ifndef DFT_FUNCTIONALS_BULK_COEXISTENCE_HPP
#define DFT_FUNCTIONALS_BULK_COEXISTENCE_HPP

#include "dft/algorithms/solvers/newton.hpp"
#include "dft/functionals/bulk/thermodynamics.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <cmath>
#include <optional>

namespace dft::functionals::bulk {

  struct Spinodal {
    double rho_low;
    double rho_high;
  };

  struct Coexistence {
    double rho_vapor;
    double rho_liquid;
  };

  struct PhaseSearch {
    double rho_max{ 1.0 };
    double rho_scan_step{ 0.01 };
    algorithms::solvers::Newton newton{ .max_iterations = 200, .tolerance = 1e-10 };

    [[nodiscard]] auto find_spinodal(const BulkThermodynamics& eos) const -> std::optional<Spinodal>;

    [[nodiscard]] auto find_coexistence(const BulkThermodynamics& eos) const -> std::optional<Coexistence>;
  };

  // Invert mu(rho) to find rho given a target chemical potential.

  [[nodiscard]] inline auto density_from_chemical_potential(
      double target_mu,
      double initial_guess,
      const BulkThermodynamics& eos,
      const algorithms::solvers::Newton& solver = { .max_iterations = 200, .tolerance = 1e-10 }
  ) -> std::optional<double> {
    auto residual = [&](const arma::vec& x) -> arma::vec {
      return arma::vec{ eos.chemical_potential(arma::vec{ x(0) }, 0) - target_mu };
    };
    auto result = solver.solve(arma::vec{ initial_guess }, residual);
    if (!result.converged || result.solution(0) <= 0.0) {
      return std::nullopt;
    }
    return result.solution(0);
  }

  namespace _internal {

    [[nodiscard]] inline auto dp_drho(double rho, const BulkThermodynamics& eos, double h = 1e-7) -> double {
      double p_plus = eos.pressure(arma::vec{ rho + h });
      double p_minus = eos.pressure(arma::vec{ rho - h });
      return (p_plus - p_minus) / (2.0 * h);
    }

    [[nodiscard]] inline auto bisect_dp_drho(double lo, double hi, const BulkThermodynamics& eos) -> double {
      double f_lo = dp_drho(lo, eos);
      for (int i = 0; i < 60; ++i) {
        double mid = 0.5 * (lo + hi);
        double f_mid = dp_drho(mid, eos);
        if ((f_lo > 0.0) == (f_mid > 0.0)) {
          lo = mid;
          f_lo = f_mid;
        } else {
          hi = mid;
        }
      }
      return 0.5 * (lo + hi);
    }

  }  // namespace _internal

  [[nodiscard]] inline auto PhaseSearch::find_spinodal(const BulkThermodynamics& eos) const -> std::optional<Spinodal> {
    double step_size = rho_scan_step;
    double max_rho = rho_max;

    double prev_dp = _internal::dp_drho(step_size, eos);
    double low_lo = 0.0, low_hi = 0.0;
    double high_lo = 0.0, high_hi = 0.0;
    bool found_low = false;
    bool found_high = false;

    double prev_rho = step_size;
    for (double rho = 2.0 * step_size; rho < max_rho; rho += step_size) {
      double curr_dp = _internal::dp_drho(rho, eos);
      if (prev_dp > 0.0 && curr_dp <= 0.0 && !found_low) {
        low_lo = prev_rho;
        low_hi = rho;
        found_low = true;
      } else if (prev_dp < 0.0 && curr_dp >= 0.0 && found_low) {
        high_lo = prev_rho;
        high_hi = rho;
        found_high = true;
        break;
      }
      prev_dp = curr_dp;
      prev_rho = rho;
    }

    if (!found_low || !found_high) {
      return std::nullopt;
    }

    return Spinodal{
      .rho_low = _internal::bisect_dp_drho(low_lo, low_hi, eos),
      .rho_high = _internal::bisect_dp_drho(high_lo, high_hi, eos),
    };
  }

  [[nodiscard]] inline auto PhaseSearch::find_coexistence(const BulkThermodynamics& eos) const
      -> std::optional<Coexistence> {
    auto spinodal = find_spinodal(eos);
    if (!spinodal) {
      return std::nullopt;
    }

    double rho_s1 = spinodal->rho_low;
    double rho_s2 = spinodal->rho_high;

    auto delta_p = [&](double rho_v) -> std::optional<double> {
      double mu_v = eos.chemical_potential(arma::vec{ rho_v }, 0);
      auto rho_l = density_from_chemical_potential(mu_v, rho_s2 * 1.2, eos, newton);
      if (!rho_l || *rho_l <= rho_s2) {
        return std::nullopt;
      }
      return eos.pressure(arma::vec{ rho_v }) - eos.pressure(arma::vec{ *rho_l });
    };

    double rho_v = rho_s1;
    auto dp_prev = delta_p(rho_v);
    if (!dp_prev) {
      return std::nullopt;
    }

    double rho_hi = rho_v;
    double rho_lo = rho_v;
    bool bracketed = false;

    while (rho_v > 1e-12) {
      rho_v /= 1.1;
      auto dp = delta_p(rho_v);
      if (!dp) {
        continue;
      }
      if ((*dp < 0.0) != (*dp_prev < 0.0)) {
        rho_lo = rho_v;
        rho_hi = rho_v * 1.1;
        bracketed = true;
        break;
      }
      dp_prev = dp;
    }

    if (!bracketed) {
      return std::nullopt;
    }

    auto dp_lo_val = delta_p(rho_lo);
    if (!dp_lo_val) {
      return std::nullopt;
    }

    for (int i = 0; i < 60; ++i) {
      double mid = 0.5 * (rho_lo + rho_hi);
      auto dp_mid = delta_p(mid);
      if (!dp_mid) {
        rho_hi = mid;
        continue;
      }
      if ((*dp_lo_val > 0.0) == (*dp_mid > 0.0)) {
        rho_lo = mid;
        dp_lo_val = dp_mid;
      } else {
        rho_hi = mid;
      }
    }

    double rv = 0.5 * (rho_lo + rho_hi);
    double mu_v = eos.chemical_potential(arma::vec{ rv }, 0);
    auto rl = density_from_chemical_potential(mu_v, rho_s2 * 1.2, eos, newton);
    if (!rl) {
      return std::nullopt;
    }

    return Coexistence{ .rho_vapor = rv, .rho_liquid = *rl };
  }

}  // namespace dft::functionals::bulk

#endif  // DFT_FUNCTIONALS_BULK_COEXISTENCE_HPP
