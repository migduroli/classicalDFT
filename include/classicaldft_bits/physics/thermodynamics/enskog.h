#ifndef CLASSICALDFT_PHYSICS_THERMODYNAMICS_ENSKOG_H
#define CLASSICALDFT_PHYSICS_THERMODYNAMICS_ENSKOG_H

#include <cmath>
#include <numbers>

namespace dft_core::physics::thermodynamics {

  // ── Packing fraction conversion ─────────────────────────────────────────────

  [[nodiscard]] inline constexpr double packing_fraction(double density) noexcept {
    return std::numbers::pi * density / 6.0;
  }

  [[nodiscard]] inline constexpr double density_from_eta(double eta) noexcept {
    return 6.0 * eta / std::numbers::pi;
  }

  // ── Hard-sphere fluid thermodynamics ──────────────────────────────────────

  /**
   * @brief Abstract hard-sphere fluid model.
   *
   * Subclasses define the excess free energy per particle as a function of
   * packing fraction $\eta$. All derivatives are with respect to $\eta$.
   * The base class provides derived thermodynamic quantities.
   */
  class HardSphereFluid {
   public:
    virtual ~HardSphereFluid() = default;

    [[nodiscard]] virtual double excess_free_energy(double eta) const = 0;
    [[nodiscard]] virtual double d_excess_free_energy(double eta) const = 0;
    [[nodiscard]] virtual double d2_excess_free_energy(double eta) const = 0;
    [[nodiscard]] virtual double d3_excess_free_energy(double eta) const = 0;

    /**
     * @brief Carnahan-Starling contact value $\chi(\eta) = (1 - \eta/2)/(1-\eta)^3$.
     */
    [[nodiscard]] double contact_value(double eta) const {
      double e1 = 1.0 - eta;
      return (1.0 - 0.5 * eta) / (e1 * e1 * e1);
    }

    /**
     * @brief Pressure $P/(\rho\,k_BT) = 1 + \eta\,f'_{ex}(\eta)$.
     */
    [[nodiscard]] double pressure(double eta) const { return 1.0 + eta * d_excess_free_energy(eta); }

    /**
     * @brief Total free energy per particle: $\ln\rho - 1 + f_{ex}(\eta(\rho))$.
     */
    [[nodiscard]] double free_energy(double density) const {
      return std::log(density) - 1.0 + excess_free_energy(packing_fraction(density));
    }

    /**
     * @brief Chemical potential $\mu/k_BT = \ln\rho + f_{ex}(\eta) + \eta\,f'_{ex}(\eta)$.
     */
    [[nodiscard]] double chemical_potential(double density) const {
      double eta = packing_fraction(density);
      return std::log(density) + excess_free_energy(eta) + eta * d_excess_free_energy(eta);
    }
  };

  // ── Carnahan-Starling ─────────────────────────────────────────────────────

  class CarnahanStarling final : public HardSphereFluid {
   public:
    [[nodiscard]] double excess_free_energy(double eta) const override {
      double e2 = (1.0 - eta) * (1.0 - eta);
      return eta * (4.0 - 3.0 * eta) / e2;
    }

    [[nodiscard]] double d_excess_free_energy(double eta) const override {
      double e3 = (1.0 - eta) * (1.0 - eta) * (1.0 - eta);
      return (4.0 - 2.0 * eta) / e3;
    }

    [[nodiscard]] double d2_excess_free_energy(double eta) const override {
      return (10.0 - 4.0 * eta) / std::pow(1.0 - eta, 4);
    }

    [[nodiscard]] double d3_excess_free_energy(double eta) const override {
      return 12.0 * (3.0 - eta) / std::pow(1.0 - eta, 5);
    }
  };

  // ── Percus-Yevick ─────────────────────────────────────────────────────────

  class PercusYevick final : public HardSphereFluid {
   public:
    enum class Route { Virial, Compressibility };

    explicit PercusYevick(Route route = Route::Compressibility) : route_(route) {}

    [[nodiscard]] double excess_free_energy(double eta) const override {
      if (route_ == Route::Virial) {
        return 2.0 * std::log(1.0 - eta) + 6.0 * eta / (1.0 - eta);
      }
      double e2 = (1.0 - eta) * (1.0 - eta);
      return -std::log(1.0 - eta) + 1.5 * eta * (2.0 - eta) / e2;
    }

    [[nodiscard]] double d_excess_free_energy(double eta) const override {
      if (route_ == Route::Virial) {
        return (4.0 + 2.0 * eta) / ((1.0 - eta) * (1.0 - eta));
      }
      return (4.0 - 2.0 * eta + eta * eta) / std::pow(1.0 - eta, 3);
    }

    [[nodiscard]] double d2_excess_free_energy(double eta) const override {
      if (route_ == Route::Virial) {
        return (10.0 + 2.0 * eta) / std::pow(1.0 - eta, 3);
      }
      return (10.0 - 2.0 * eta + eta * eta) / std::pow(1.0 - eta, 4);
    }

    [[nodiscard]] double d3_excess_free_energy(double eta) const override {
      if (route_ == Route::Virial) {
        return (32.0 + 4.0 * eta) / std::pow(1.0 - eta, 4);
      }
      return (38.0 - 4.0 * eta + 2.0 * eta * eta) / std::pow(1.0 - eta, 5);
    }

   private:
    Route route_;
  };

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
      constexpr double cv = 1.5;
      constexpr double cp = 2.5;
      constexpr double gamma = cp / cv;
      return (((gamma - 1.0) / cp) * thermal_conductivity(density, chi) + (4.0 / 3.0) * shear_viscosity(density, chi) +
              bulk_viscosity(density, chi)) /
          density;
    }

  }  // namespace transport

}  // namespace dft_core::physics::thermodynamics

#endif  // CLASSICALDFT_PHYSICS_THERMODYNAMICS_ENSKOG_H
