#ifndef DFT_THERMODYNAMICS_ENSKOG_H
#define DFT_THERMODYNAMICS_ENSKOG_H

#include "dft/math/autodiff.h"

#include <cmath>
#include <numbers>
#include <string>
#include <variant>

namespace dft::thermodynamics {

  // ── Packing fraction conversion ─────────────────────────────────────────────

  template <typename T = double>
  [[nodiscard]] inline constexpr T packing_fraction(T density) noexcept {
    return (std::numbers::pi / 6.0) * density;
  }

  [[nodiscard]] inline constexpr double density_from_eta(double eta) noexcept {
    return 6.0 * eta / std::numbers::pi;
  }

  // ── Contact value (universal, model-independent) ──────────────────────────

  [[nodiscard]] inline double contact_value(double eta) {
    double e = 1.0 - eta;
    return (1.0 - 0.5 * eta) / (e * e * e);
  }

  // ── Hard-sphere fluid models (flat structs, no inheritance) ───────────────

  struct CarnahanStarling {
    template <typename T = double>
    [[nodiscard]] static T excess_free_energy(T eta) {
      T e = T(1.0) - eta;
      return eta * (T(4.0) - T(3.0) * eta) / (e * e);
    }

    [[nodiscard]] static std::string name() { return "CarnahanStarling"; }
  };

  struct PercusYevickVirial {
    template <typename T = double>
    [[nodiscard]] static T excess_free_energy(T eta) {
      using std::log;
      T e = T(1.0) - eta;
      return T(2.0) * log(e) + T(6.0) * eta / e;
    }

    [[nodiscard]] static std::string name() { return "PercusYevickVirial"; }
  };

  struct PercusYevickCompressibility {
    template <typename T = double>
    [[nodiscard]] static T excess_free_energy(T eta) {
      using std::log;
      T e = T(1.0) - eta;
      return -log(e) + T(1.5) * eta * (T(2.0) - eta) / (e * e);
    }

    [[nodiscard]] static std::string name() { return "PercusYevickCompressibility"; }
  };

  // ── Sum type for hard-sphere models ───────────────────────────────────────

  using HardSphereModel = std::variant<CarnahanStarling, PercusYevickVirial, PercusYevickCompressibility>;

  // ── Free functions: operate on any model via std::visit + autodiff ────────

  [[nodiscard]] inline double hs_excess_free_energy(const HardSphereModel& model, double eta, int order = 0) {
    return std::visit(
        [eta, order](const auto& m) {
          if (order == 0) {
            return static_cast<double>(m.excess_free_energy(eta));
          }
          auto [f, df, d2f, d3f] =
              math::derivatives_up_to_3([&](math::dual3rd x) { return m.excess_free_energy(x); }, eta);
          switch (order) {
            case 1:
              return df;
            case 2:
              return d2f;
            case 3:
              return d3f;
            default:
              return f;
          }
        },
        model
    );
  }

  [[nodiscard]] inline double hs_pressure(const HardSphereModel& model, double eta) {
    return std::visit(
        [eta](const auto& m) {
          auto [f, df] = math::derivatives_up_to_1([&](math::dual x) { return m.excess_free_energy(x); }, eta);
          return 1.0 + eta * df;
        },
        model
    );
  }

  [[nodiscard]] inline double hs_free_energy(const HardSphereModel& model, double density) {
    double eta = packing_fraction(density);
    return std::visit(
        [&](const auto& m) { return std::log(density) - 1.0 + static_cast<double>(m.excess_free_energy(eta)); }, model
    );
  }

  [[nodiscard]] inline double hs_chemical_potential(const HardSphereModel& model, double density) {
    double eta = packing_fraction(density);
    return std::visit(
        [&](const auto& m) {
          auto [f, df] = math::derivatives_up_to_1([&](math::dual x) { return m.excess_free_energy(x); }, eta);
          return std::log(density) + f + eta * df;
        },
        model
    );
  }

  [[nodiscard]] inline std::string hs_name(const HardSphereModel& model) {
    return std::visit([](const auto& m) { return m.name(); }, model);
  }

  // ── Transport coefficients (Enskog theory, d = kT = 1) ───────────────────

  namespace transport {

    [[nodiscard]] inline double bulk_viscosity(double density, double chi) {
      return (5.0 / (16.0 * std::sqrt(std::numbers::pi))) * (64.0 / 45.0) * std::numbers::pi * density * density * chi;
    }

    [[nodiscard]] inline double shear_viscosity(double density, double chi) {
      double term = 1.0 + 4.0 * std::numbers::pi * density * chi / 15.0;
      return (5.0 / (16.0 * std::sqrt(std::numbers::pi) * chi)) * term * term + 0.6 * bulk_viscosity(density, chi);
    }

    [[nodiscard]] inline double thermal_conductivity(double density, double chi) {
      double term = 1.0 + 0.4 * std::numbers::pi * density * chi;
      return (75.0 / (64.0 * std::sqrt(std::numbers::pi) * chi)) * term * term +
          (128.0 / 225.0) * std::numbers::pi * density * density * chi;
    }

    [[nodiscard]] inline double sound_damping(double density, double chi) {
      constexpr double CV = 1.5;
      constexpr double CP = 2.5;
      constexpr double GAMMA = CP / CV;
      return (((GAMMA - 1.0) / CP) * thermal_conductivity(density, chi) + (4.0 / 3.0) * shear_viscosity(density, chi) +
              bulk_viscosity(density, chi)) /
          density;
    }

  }  // namespace transport

}  // namespace dft::thermodynamics

#endif  // DFT_THERMODYNAMICS_ENSKOG_H
