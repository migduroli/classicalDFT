#ifndef DFT_FUNCTIONALS_BULK_PHASE_DIAGRAM_HPP
#define DFT_FUNCTIONALS_BULK_PHASE_DIAGRAM_HPP

#include "dft/algorithms/solvers/continuation.hpp"
#include "dft/functionals/bulk/coexistence.hpp"
#include "dft/math/spline.hpp"

#include <armadillo>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace dft::functionals::bulk {

  // Result of tracing the full coexistence (binodal) curve from a starting
  // temperature up to the critical point. All arrays share the same length.

  struct CoexistenceCurve {
    arma::vec temperature;
    arma::vec rho_vapor;
    arma::vec rho_liquid;
    double critical_temperature{ 0.0 };
    double critical_density{ 0.0 };
  };

  // Result of tracing the spinodal curve from a starting temperature
  // up to the critical point. Both branches are aligned on the same
  // temperature grid.

  struct SpinodalCurve {
    arma::vec temperature;
    arma::vec rho_low;
    arma::vec rho_high;
    double critical_temperature{ 0.0 };
    double critical_density{ 0.0 };
  };

  // Phase boundaries at a single temperature: densities on both sides
  // of the binodal and spinodal. NaN indicates the boundary is not
  // available (e.g. above Tc).

  struct PhaseBoundaries {
    double temperature;
    double binodal_vapor{ std::numeric_limits<double>::quiet_NaN() };
    double binodal_liquid{ std::numeric_limits<double>::quiet_NaN() };
    double spinodal_low{ std::numeric_limits<double>::quiet_NaN() };
    double spinodal_high{ std::numeric_limits<double>::quiet_NaN() };
  };

  // Full phase diagram containing both binodal and spinodal curves,
  // with a unified critical point.

  struct PhaseDiagram {
    CoexistenceCurve binodal;
    SpinodalCurve spinodal;
    double critical_temperature{ 0.0 };
    double critical_density{ 0.0 };

    [[nodiscard]] auto interpolate(double temperature) const -> PhaseBoundaries {
      PhaseBoundaries result{ .temperature = temperature };

      auto try_spline = [](const arma::vec& T, const arma::vec& y, double t) -> double {
        if (T.n_elem < 4 || t < T.front() || t > T.back()) {
          return std::numeric_limits<double>::quiet_NaN();
        }
        math::CubicSpline
            spline(std::span<const double>(T.memptr(), T.n_elem), std::span<const double>(y.memptr(), y.n_elem));
        return spline(t);
      };

      if (!binodal.temperature.is_empty()) {
        result.binodal_vapor = try_spline(binodal.temperature, binodal.rho_vapor, temperature);
        result.binodal_liquid = try_spline(binodal.temperature, binodal.rho_liquid, temperature);
      }

      if (!spinodal.temperature.is_empty()) {
        result.spinodal_low = try_spline(spinodal.temperature, spinodal.rho_low, temperature);
        result.spinodal_high = try_spline(spinodal.temperature, spinodal.rho_high, temperature);
      }

      return result;
    }
  };

  // EoS factory: produces a BulkThermodynamics at a given temperature.

  using EoSFactory = std::function<BulkThermodynamics(double kT)>;

  // Build an EoS factory from an FMT model, species list, and interactions.
  // At each temperature the factory creates a BulkThermodynamics with
  // analytical a_vdw (no grid). Suitable for PhaseDiagramBuilder.
  //
  //   auto eos_at = make_eos_factory(RSLT{}, species, interactions);
  //   auto eos = eos_at(0.8);  // BulkThermodynamics at kT = 0.8

  [[nodiscard]] inline auto make_eos_factory(
      fmt::FMTModel fmt_model,
      std::vector<Species> species,
      std::vector<physics::Interaction> interactions
  ) -> EoSFactory {
    return [fmt_model = std::move(fmt_model),
            species = std::move(species),
            interactions = std::move(interactions)](double kT) {
      return make_bulk_thermodynamics(species, make_bulk_weights(fmt_model, interactions, kT));
    };
  }

  // Configuration for full phase diagram computation.

  struct PhaseDiagramBuilder {
    double start_temperature{ 0.6 };
    double density_gap_tolerance{ 0.005 };
    double turning_point_margin{ 0.02 };
    double spinodal_temperature_step{ 0.01 };
    PhaseSearch search{};
    algorithms::continuation::Continuation continuation{
      .initial_step = 0.005,
      .max_step = 0.05,
      .min_step = 1e-6,
      .growth_factor = 1.5,
      .shrink_factor = 0.5,
      .newton = { .max_iterations = 300, .tolerance = 1e-10 },
    };

    [[nodiscard]] auto binodal(const EoSFactory& eos_factory) const -> std::optional<CoexistenceCurve>;

    [[nodiscard]] auto spinodal_curve(const EoSFactory& eos_factory) const -> std::optional<SpinodalCurve>;
  };

  namespace _internal {

    // Trace the coexistence curve as a function of temperature using
    // pseudo-arclength continuation. The weight factory produces the
    // precomputed Weights at each temperature.
    //
    // x = (rho_vapor, rho_liquid), lambda = kT
    // R = (P_v - P_l, mu_v - mu_l) = 0

    [[nodiscard]] inline auto trace_coexistence(
        const Coexistence& start,
        double start_kT,
        const EoSFactory& eos_factory,
        const algorithms::continuation::Continuation& continuation = {},
        std::function<bool(const algorithms::continuation::CurvePoint&)> stop = {}
    ) -> std::vector<algorithms::continuation::CurvePoint> {
      auto residual = [&](const arma::vec& x, double kT) -> arma::vec {
        auto eos = eos_factory(kT);
        arma::vec rv{ x(0) };
        arma::vec rl{ x(1) };
        return arma::vec{
          eos.pressure(rv) - eos.pressure(rl),
          eos.chemical_potential(rv, 0) - eos.chemical_potential(rl, 0),
        };
      };

      algorithms::continuation::CurvePoint start_point{
        .x = arma::vec{ start.rho_vapor, start.rho_liquid },
        .lambda = start_kT,
        .dx_ds = arma::vec{ 0.0, 0.0 },
        .dlambda_ds = 1.0,
      };

      return continuation.trace(start_point, residual, std::move(stop));
    }

  }  // namespace _internal

  // Trace the full binodal (coexistence) curve from a starting temperature
  // up to the critical point using pseudo-arclength continuation.
  // The curve is truncated at the maximum temperature (critical point);
  // beyond Tc the continuation follows an unphysical branch.

  [[nodiscard]] inline auto PhaseDiagramBuilder::binodal(const EoSFactory& eos_factory) const
      -> std::optional<CoexistenceCurve> {
    auto eos_start = eos_factory(start_temperature);
    auto seed = search.find_coexistence(eos_start);
    if (!seed) {
      return std::nullopt;
    }

    double T_max_seen = start_temperature;
    auto curve = _internal::trace_coexistence(
        *seed,
        start_temperature,
        eos_factory,
        continuation,
        [&](const algorithms::continuation::CurvePoint& pt) {
          T_max_seen = std::max(T_max_seen, pt.lambda);
          double gap = std::abs(pt.x(1) - pt.x(0));
          return gap < density_gap_tolerance || pt.lambda < T_max_seen - turning_point_margin;
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

    // Near Tc the continuation can swap the vapor/liquid labels.
    // Enforce rv < rl at every point by swapping when crossed.
    for (arma::uword i = 0; i <= i_max; ++i) {
      if (rv(i) > rl(i)) {
        std::swap(rv(i), rl(i));
      }
    }

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

  [[nodiscard]] inline auto PhaseDiagramBuilder::spinodal_curve(const EoSFactory& eos_factory) const
      -> std::optional<SpinodalCurve> {
    std::vector<double> T_vals, lo_vals, hi_vals;

    // Bisection on dp_drho: guaranteed convergence within a bracket.
    auto bisect_root = [&](double a, double b, const BulkThermodynamics& eos) -> std::optional<double> {
      double fa = _internal::dp_drho(a, eos);
      double fb = _internal::dp_drho(b, eos);
      if (fa * fb > 0.0) {
        return std::nullopt;
      }
      for (int i = 0; i < 60; ++i) {
        double mid = 0.5 * (a + b);
        double fm = _internal::dp_drho(mid, eos);
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
    auto bracket_root = [&](double guess, double sign_left, double step, const BulkThermodynamics& eos
                        ) -> std::optional<std::pair<double, double>> {
      for (double delta = step; delta < 0.3; delta += step) {
        double a = std::max(guess - delta, 1e-6);
        double b = guess + delta;
        double fa = _internal::dp_drho(a, eos);
        double fb = _internal::dp_drho(b, eos);
        if (fa * fb < 0.0) {
          return std::pair{ a, b };
        }
      }
      return std::nullopt;
    };

    std::optional<Spinodal> prev;
    int consecutive_failures = 0;

    double dT = spinodal_temperature_step;
    for (double T = start_temperature;; T += dT) {
      auto eos = eos_factory(T);

      std::optional<Spinodal> sp;

      if (prev) {
        // Bisection from previous values: search for brackets nearby.
        double scan_step = search.rho_scan_step;
        auto lo_bracket = bracket_root(prev->rho_low, 1.0, scan_step, eos);
        auto hi_bracket = bracket_root(prev->rho_high, -1.0, scan_step, eos);

        if (lo_bracket && hi_bracket) {
          auto rlo = bisect_root(lo_bracket->first, lo_bracket->second, eos);
          auto rhi = bisect_root(hi_bracket->first, hi_bracket->second, eos);
          if (rlo && rhi && *rhi > *rlo) {
            sp = Spinodal{ .rho_low = *rlo, .rho_high = *rhi };
          }
        }
      }

      // Fall back to full coarse scan.
      if (!sp) {
        sp = search.find_spinodal(eos);
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

  [[nodiscard]] inline auto phase_diagram(const EoSFactory& eos_factory, const PhaseDiagramBuilder& builder = {})
      -> std::optional<PhaseDiagram> {
    auto b = builder.binodal(eos_factory);
    auto s = builder.spinodal_curve(eos_factory);

    if (!b && !s) {
      return std::nullopt;
    }

    double Tc = 0.0;
    double rho_c = 0.0;
    if (b) {
      Tc = b->critical_temperature;
      rho_c = b->critical_density;
    } else if (s) {
      Tc = s->critical_temperature;
      rho_c = s->critical_density;
    }

    return PhaseDiagram{
      .binodal = b.value_or(CoexistenceCurve{}),
      .spinodal = s.value_or(SpinodalCurve{}),
      .critical_temperature = Tc,
      .critical_density = rho_c,
    };
  }

}  // namespace dft::functionals::bulk

#endif  // DFT_FUNCTIONALS_BULK_PHASE_DIAGRAM_HPP
