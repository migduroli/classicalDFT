#ifndef DFT_FUNCTIONALS_BULK_PHASE_DIAGRAM_HPP
#define DFT_FUNCTIONALS_BULK_PHASE_DIAGRAM_HPP

#include "dft/algorithms/solvers/continuation.hpp"
#include "dft/algorithms/solvers/newton.hpp"
#include "dft/functionals/bulk/thermodynamics.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <functional>
#include <optional>
#include <vector>

namespace dft::functionals::bulk {

  // Spinodal densities mark the boundaries of mechanical instability
  // where dP/drho = 0.

  struct Spinodal {
    double rho_low;
    double rho_high;
  };

  // Coexistence densities satisfy equal pressure and chemical potential
  // across the vapor-liquid boundary (Maxwell construction).

  struct Coexistence {
    double rho_vapor;
    double rho_liquid;
  };

  // Configuration for phase diagram searches.

  struct PhaseSearchConfig {
    double rho_max{1.0};
    double rho_scan_step{0.01};
    algorithms::solvers::NewtonConfig newton{.max_iterations = 200, .tolerance = 1e-10};
  };

  // Weight factory: produces bulk Weights at a given temperature.
  // Used by all temperature-scanning functions.

  using WeightFactory = std::function<Weights(double kT)>;

  // Result of tracing the full coexistence (binodal) curve from a starting
  // temperature up to the critical point. All arrays share the same length.

  struct CoexistenceCurve {
    arma::vec temperature;
    arma::vec rho_vapor;
    arma::vec rho_liquid;
    double critical_temperature{0.0};
    double critical_density{0.0};
  };

  // Result of tracing the spinodal curve from a starting temperature
  // up to the critical point. Both branches are aligned on the same
  // temperature grid.

  struct SpinodalCurve {
    arma::vec temperature;
    arma::vec rho_low;
    arma::vec rho_high;
    double critical_temperature{0.0};
    double critical_density{0.0};
  };

  // Full phase diagram containing both binodal and spinodal curves.

  struct PhaseDiagram {
    CoexistenceCurve binodal;
    SpinodalCurve spinodal;
  };

  // Configuration for full phase diagram computation.

  struct PhaseDiagramConfig {
    double start_temperature{0.6};
    double density_gap_tolerance{0.005};
    double turning_point_margin{0.02};
    double spinodal_temperature_step{0.01};
    PhaseSearchConfig search{};
    algorithms::continuation::ContinuationConfig continuation{
        .initial_step = 0.005,
        .max_step = 0.05,
        .min_step = 1e-6,
        .growth_factor = 1.5,
        .shrink_factor = 0.5,
        .newton = {.max_iterations = 300, .tolerance = 1e-10},
    };
  };

  namespace detail {

    // Numerical derivative of pressure w.r.t. density (central difference).

    [[nodiscard]] inline auto dp_drho(
        double rho, const std::vector<Species>& species,
        const Weights& weights, double h = 1e-7
    ) -> double {
      double p_plus = pressure(arma::vec{rho + h}, species, weights);
      double p_minus = pressure(arma::vec{rho - h}, species, weights);
      return (p_plus - p_minus) / (2.0 * h);
    }

  }  // namespace detail

  // Find the density at which the chemical potential equals a target value.
  // Uses Newton's method from the given initial guess.

  [[nodiscard]] inline auto density_from_chemical_potential(
      double target_mu, double initial_guess,
      const std::vector<Species>& species, const Weights& weights,
      const algorithms::solvers::NewtonConfig& config = {.max_iterations = 200, .tolerance = 1e-10}
  ) -> std::optional<double> {
    auto residual = [&](const arma::vec& x) -> arma::vec {
      return arma::vec{chemical_potential(arma::vec{x(0)}, species, weights, 0) - target_mu};
    };

    auto result = algorithms::solvers::newton(arma::vec{initial_guess}, residual, config);

    if (!result.converged || result.solution(0) <= 0.0) {
      return std::nullopt;
    }
    return result.solution(0);
  }

  // Find the spinodal densities where dP/drho = 0 for a single-component
  // system. A coarse scan locates sign changes in the pressure derivative,
  // then Newton refines each root.
  // Returns nullopt if no van der Waals loop is found.

  [[nodiscard]] inline auto find_spinodal(
      const std::vector<Species>& species, const Weights& weights,
      const PhaseSearchConfig& config = {}
  ) -> std::optional<Spinodal> {
    double step = config.rho_scan_step;
    double max_rho = config.rho_max;

    // Coarse scan for sign changes in dP/drho.
    double prev_dp = detail::dp_drho(step, species, weights);
    double rho_low_guess = 0.0;
    double rho_high_guess = 0.0;

    for (double rho = 2.0 * step; rho < max_rho; rho += step) {
      double curr_dp = detail::dp_drho(rho, species, weights);
      if (prev_dp > 0.0 && curr_dp <= 0.0 && rho_low_guess == 0.0) {
        rho_low_guess = rho - 0.5 * step;
      } else if (prev_dp < 0.0 && curr_dp >= 0.0 && rho_low_guess > 0.0) {
        rho_high_guess = rho - 0.5 * step;
        break;
      }
      prev_dp = curr_dp;
    }

    if (rho_low_guess == 0.0 || rho_high_guess == 0.0) {
      return std::nullopt;
    }

    // Refine each root with Newton on dP/drho = 0.
    auto residual = [&](const arma::vec& x) -> arma::vec {
      return arma::vec{detail::dp_drho(x(0), species, weights)};
    };

    auto r_low = algorithms::solvers::newton(arma::vec{rho_low_guess}, residual, config.newton);
    auto r_high = algorithms::solvers::newton(arma::vec{rho_high_guess}, residual, config.newton);

    if (!r_low.converged || !r_high.converged) {
      return std::nullopt;
    }

    return Spinodal{.rho_low = r_low.solution(0), .rho_high = r_high.solution(0)};
  }

  // Find the vapor-liquid coexistence densities (Maxwell construction).
  //
  // The equal-mu condition uniquely determines rho_l given rho_v,
  // reducing the problem to a 1D root: delta_P(rho_v) = 0.
  // A coarse scan from the spinodal brackets this root, then Newton
  // refines it.
  // Returns nullopt if no coexistence is found.

  [[nodiscard]] inline auto find_coexistence(
      const std::vector<Species>& species, const Weights& weights,
      const PhaseSearchConfig& config = {}
  ) -> std::optional<Coexistence> {
    auto spinodal = find_spinodal(species, weights, config);
    if (!spinodal) {
      return std::nullopt;
    }

    double rho_s1 = spinodal->rho_low;
    double rho_s2 = spinodal->rho_high;

    // Given rho_v, find rho_l with equal mu, then return P_v - P_l.
    auto delta_p = [&](double rho_v) -> std::optional<double> {
      double mu_v = chemical_potential(arma::vec{rho_v}, species, weights, 0);
      auto rho_l = density_from_chemical_potential(
          mu_v, rho_s2 * 1.2, species, weights, config.newton
      );
      if (!rho_l || *rho_l <= rho_s2) {
        return std::nullopt;
      }
      return pressure(arma::vec{rho_v}, species, weights)
           - pressure(arma::vec{*rho_l}, species, weights);
    };

    // Bracket: scan downward from spinodal vapor density.
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

    // Refine the 1D root delta_P(rho_v) = 0 with Newton.
    auto residual = [&](const arma::vec& x) -> arma::vec {
      return arma::vec{delta_p(x(0)).value_or(1e10)};
    };

    double mid = 0.5 * (rho_lo + rho_hi);
    auto result = algorithms::solvers::newton(arma::vec{mid}, residual, config.newton);

    if (!result.converged || result.solution(0) <= 0.0) {
      return std::nullopt;
    }

    double rv = result.solution(0);
    double mu_v = chemical_potential(arma::vec{rv}, species, weights, 0);
    auto rl = density_from_chemical_potential(mu_v, rho_s2 * 1.2, species, weights, config.newton);
    if (!rl) {
      return std::nullopt;
    }

    return Coexistence{.rho_vapor = rv, .rho_liquid = *rl};
  }

  // Trace the coexistence curve as a function of temperature using
  // pseudo-arclength continuation. The weight factory produces the
  // precomputed Weights at each temperature.
  //
  // x = (rho_vapor, rho_liquid), lambda = kT
  // R = (P_v - P_l, mu_v - mu_l) = 0

  [[nodiscard]] inline auto trace_coexistence(
      const Coexistence& start, double start_kT,
      const std::vector<Species>& species, const WeightFactory& weight_factory,
      const algorithms::continuation::ContinuationConfig& config = {},
      std::function<bool(const algorithms::continuation::CurvePoint&)> stop = {}
  ) -> std::vector<algorithms::continuation::CurvePoint> {
    auto residual = [&](const arma::vec& x, double kT) -> arma::vec {
      auto w = weight_factory(kT);
      arma::vec rv{x(0)};
      arma::vec rl{x(1)};
      return arma::vec{
          pressure(rv, species, w) - pressure(rl, species, w),
          chemical_potential(rv, species, w, 0) - chemical_potential(rl, species, w, 0),
      };
    };

    algorithms::continuation::CurvePoint start_point{
        .x = arma::vec{start.rho_vapor, start.rho_liquid},
        .lambda = start_kT,
        .dx_ds = arma::vec{0.0, 0.0},
        .dlambda_ds = 1.0,
    };

    return algorithms::continuation::trace(start_point, residual, config, std::move(stop));
  }

  // Trace the full binodal (coexistence) curve from a starting temperature
  // up to the critical point using pseudo-arclength continuation.
  // The curve is truncated at the maximum temperature (critical point);
  // beyond Tc the continuation follows an unphysical branch.

  [[nodiscard]] inline auto binodal(
      const std::vector<Species>& species, const WeightFactory& weight_factory,
      const PhaseDiagramConfig& config = {}
  ) -> std::optional<CoexistenceCurve> {
    auto w_start = weight_factory(config.start_temperature);
    auto seed = find_coexistence(species, w_start, config.search);
    if (!seed) {
      return std::nullopt;
    }

    double T_max_seen = config.start_temperature;
    auto curve = trace_coexistence(
        *seed, config.start_temperature, species, weight_factory, config.continuation,
        [&](const algorithms::continuation::CurvePoint& pt) {
          T_max_seen = std::max(T_max_seen, pt.lambda);
          double gap = std::abs(pt.x(1) - pt.x(0));
          return gap < config.density_gap_tolerance
              || pt.lambda < T_max_seen - config.turning_point_margin;
        }
    );

    if (curve.empty()) {
      return std::nullopt;
    }

    arma::vec T(curve.size()), rv(curve.size()), rl(curve.size());
    for (std::size_t i = 0; i < curve.size(); ++i) {
      T(i) = curve[i].lambda;
      rv(i) = curve[i].x(0);
      rl(i) = curve[i].x(1);
    }

    arma::uword i_max = T.index_max();
    return CoexistenceCurve{
        .temperature = T.head(i_max + 1),
        .rho_vapor = rv.head(i_max + 1),
        .rho_liquid = rl.head(i_max + 1),
        .critical_temperature = T(i_max),
        .critical_density = 0.5 * (rv(i_max) + rl(i_max)),
    };
  }

  // Trace the spinodal curve from a starting temperature up to the
  // critical point. At each temperature, the spinodal densities
  // (where dP/drho = 0) are found by coarse scan + Newton refinement.
  // The temperature step is refined automatically near the critical
  // point where the density gap closes.

  [[nodiscard]] inline auto spinodal(
      const std::vector<Species>& species, const WeightFactory& weight_factory,
      const PhaseDiagramConfig& config = {}
  ) -> std::optional<SpinodalCurve> {
    std::vector<double> T_vals, lo_vals, hi_vals;

    // Bisection on dp_drho: guaranteed convergence within a bracket.
    auto bisect_root = [&](double a, double b, const Weights& w) -> std::optional<double> {
      double fa = detail::dp_drho(a, species, w);
      double fb = detail::dp_drho(b, species, w);
      if (fa * fb > 0.0) {
        return std::nullopt;
      }
      for (int i = 0; i < 60; ++i) {
        double mid = 0.5 * (a + b);
        double fm = detail::dp_drho(mid, species, w);
        if (fa * fm <= 0.0) {
          b = mid;
        } else {
          a = mid;
          fa = fm;
        }
      }
      return 0.5 * (a + b);
    };

    // Bracket a root around a guess by expanding outward.
    auto bracket_root = [&](double guess, double sign_left, double step,
                            const Weights& w) -> std::optional<std::pair<double, double>> {
      for (double delta = step; delta < 0.3; delta += step) {
        double a = std::max(guess - delta, 1e-6);
        double b = guess + delta;
        double fa = detail::dp_drho(a, species, w);
        double fb = detail::dp_drho(b, species, w);
        if (fa * fb < 0.0) {
          return std::pair{a, b};
        }
      }
      return std::nullopt;
    };

    std::optional<Spinodal> prev;
    int consecutive_failures = 0;

    double dT = config.spinodal_temperature_step;
    for (double T = config.start_temperature; ; T += dT) {
      auto w = weight_factory(T);

      std::optional<Spinodal> sp;

      if (prev) {
        // Bisection from previous values: search for brackets nearby.
        double scan_step = config.search.rho_scan_step;
        auto lo_bracket = bracket_root(prev->rho_low, 1.0, scan_step, w);
        auto hi_bracket = bracket_root(prev->rho_high, -1.0, scan_step, w);

        if (lo_bracket && hi_bracket) {
          auto rlo = bisect_root(lo_bracket->first, lo_bracket->second, w);
          auto rhi = bisect_root(hi_bracket->first, hi_bracket->second, w);
          if (rlo && rhi && *rhi > *rlo) {
            sp = Spinodal{.rho_low = *rlo, .rho_high = *rhi};
          }
        }
      }

      // Fall back to full coarse scan.
      if (!sp) {
        sp = find_spinodal(species, w, config.search);
      }

      if (!sp) {
        if (++consecutive_failures >= 3) {
          break;
        }
        continue;
      }

      consecutive_failures = 0;
      prev = sp;
      T_vals.push_back(T);
      lo_vals.push_back(sp->rho_low);
      hi_vals.push_back(sp->rho_high);

      double gap = sp->rho_high - sp->rho_low;
      if (gap < 0.1) {
        dT = std::min(dT, 0.005);
      }
      if (gap < 0.05) {
        dT = std::min(dT, 0.002);
      }
    }

    if (T_vals.empty()) {
      return std::nullopt;
    }

    double Tc = T_vals.back();
    double rho_c = 0.5 * (lo_vals.back() + hi_vals.back());

    return SpinodalCurve{
        .temperature = arma::vec(T_vals),
        .rho_low = arma::vec(lo_vals),
        .rho_high = arma::vec(hi_vals),
        .critical_temperature = Tc,
        .critical_density = rho_c,
    };
  }

  // Full phase diagram (binodal + spinodal) for a single-component system.
  // Returns nullopt only if both curves fail to compute.

  [[nodiscard]] inline auto phase_diagram(
      const std::vector<Species>& species, const WeightFactory& weight_factory,
      const PhaseDiagramConfig& config = {}
  ) -> std::optional<PhaseDiagram> {
    auto b = binodal(species, weight_factory, config);
    auto s = spinodal(species, weight_factory, config);

    if (!b && !s) {
      return std::nullopt;
    }

    return PhaseDiagram{
        .binodal = b.value_or(CoexistenceCurve{}),
        .spinodal = s.value_or(SpinodalCurve{}),
    };
  }

}  // namespace dft::functionals::bulk

#endif  // DFT_FUNCTIONALS_BULK_PHASE_DIAGRAM_HPP
