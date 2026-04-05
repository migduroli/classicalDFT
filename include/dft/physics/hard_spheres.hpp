#ifndef DFT_PHYSICS_HARD_SPHERES_HPP
#define DFT_PHYSICS_HARD_SPHERES_HPP

#include "dft/math/autodiff.hpp"

#include <numbers>
#include <string_view>
#include <variant>

namespace dft::physics::hard_spheres {

  // Packing fraction conversion (unit diameter d = 1)

  template <typename T = double>
  [[nodiscard]] inline constexpr auto packing_fraction(T density) noexcept -> T {
    return (std::numbers::pi / 6.0) * density;
  }

  [[nodiscard]] inline constexpr auto density_from_eta(double eta) noexcept -> double {
    return 6.0 * eta / std::numbers::pi;
  }

  // Contact value of the pair correlation function at contact (model-independent)

  [[nodiscard]] inline auto contact_value(double eta) -> double {
    double e = 1.0 - eta;
    return (1.0 - 0.5 * eta) / (e * e * e);
  }

  // Hard-sphere fluid models: flat structs with a templated
  // excess_free_energy(eta) to support autodiff forward types.

  struct CarnahanStarling {
    static constexpr std::string_view NAME = "CarnahanStarling";

    template <typename T = double>
    [[nodiscard]] static auto excess_free_energy(T eta) -> T {
      T e = T(1.0) - eta;
      return eta * (T(4.0) - T(3.0) * eta) / (e * e);
    }

    [[nodiscard]] static auto d_excess_free_energy(double eta) -> double {
      auto [f, df] =
          math::derivatives_up_to_1([](math::dual x) -> math::dual { return excess_free_energy(x); }, eta);
      return df;
    }

    [[nodiscard]] static auto d2_excess_free_energy(double eta) -> double {
      auto [f, df, d2f] =
          math::derivatives_up_to_2([](math::dual2nd x) -> math::dual2nd { return excess_free_energy(x); }, eta);
      return d2f;
    }

    [[nodiscard]] static auto d3_excess_free_energy(double eta) -> double {
      auto [f, df, d2f, d3f] =
          math::derivatives_up_to_3([](math::dual3rd x) -> math::dual3rd { return excess_free_energy(x); }, eta);
      return d3f;
    }

    [[nodiscard]] static auto pressure(double eta) -> double { return 1.0 + eta * d_excess_free_energy(eta); }

    [[nodiscard]] static auto free_energy(double density) -> double {
      double eta = packing_fraction(density);
      return std::log(density) - 1.0 + excess_free_energy(eta);
    }

    [[nodiscard]] static auto chemical_potential(double density) -> double {
      double eta = packing_fraction(density);
      return std::log(density) + excess_free_energy(eta) + eta * d_excess_free_energy(eta);
    }
  };

  struct PercusYevickVirial {
    static constexpr std::string_view NAME = "PercusYevickVirial";

    template <typename T = double>
    [[nodiscard]] static auto excess_free_energy(T eta) -> T {
      using std::log;
      T e = T(1.0) - eta;
      return T(2.0) * log(e) + T(6.0) * eta / e;
    }

    [[nodiscard]] static auto d_excess_free_energy(double eta) -> double {
      auto [f, df] =
          math::derivatives_up_to_1([](math::dual x) -> math::dual { return excess_free_energy(x); }, eta);
      return df;
    }

    [[nodiscard]] static auto d2_excess_free_energy(double eta) -> double {
      auto [f, df, d2f] =
          math::derivatives_up_to_2([](math::dual2nd x) -> math::dual2nd { return excess_free_energy(x); }, eta);
      return d2f;
    }

    [[nodiscard]] static auto d3_excess_free_energy(double eta) -> double {
      auto [f, df, d2f, d3f] =
          math::derivatives_up_to_3([](math::dual3rd x) -> math::dual3rd { return excess_free_energy(x); }, eta);
      return d3f;
    }

    [[nodiscard]] static auto pressure(double eta) -> double { return 1.0 + eta * d_excess_free_energy(eta); }

    [[nodiscard]] static auto free_energy(double density) -> double {
      double eta = packing_fraction(density);
      return std::log(density) - 1.0 + excess_free_energy(eta);
    }

    [[nodiscard]] static auto chemical_potential(double density) -> double {
      double eta = packing_fraction(density);
      return std::log(density) + excess_free_energy(eta) + eta * d_excess_free_energy(eta);
    }
  };

  struct PercusYevickCompressibility {
    static constexpr std::string_view NAME = "PercusYevickCompressibility";

    template <typename T = double>
    [[nodiscard]] static auto excess_free_energy(T eta) -> T {
      using std::log;
      T e = T(1.0) - eta;
      return -log(e) + T(1.5) * eta * (T(2.0) - eta) / (e * e);
    }

    [[nodiscard]] static auto d_excess_free_energy(double eta) -> double {
      auto [f, df] =
          math::derivatives_up_to_1([](math::dual x) -> math::dual { return excess_free_energy(x); }, eta);
      return df;
    }

    [[nodiscard]] static auto d2_excess_free_energy(double eta) -> double {
      auto [f, df, d2f] =
          math::derivatives_up_to_2([](math::dual2nd x) -> math::dual2nd { return excess_free_energy(x); }, eta);
      return d2f;
    }

    [[nodiscard]] static auto d3_excess_free_energy(double eta) -> double {
      auto [f, df, d2f, d3f] =
          math::derivatives_up_to_3([](math::dual3rd x) -> math::dual3rd { return excess_free_energy(x); }, eta);
      return d3f;
    }

    [[nodiscard]] static auto pressure(double eta) -> double { return 1.0 + eta * d_excess_free_energy(eta); }

    [[nodiscard]] static auto free_energy(double density) -> double {
      double eta = packing_fraction(density);
      return std::log(density) - 1.0 + excess_free_energy(eta);
    }

    [[nodiscard]] static auto chemical_potential(double density) -> double {
      double eta = packing_fraction(density);
      return std::log(density) + excess_free_energy(eta) + eta * d_excess_free_energy(eta);
    }
  };

  using HardSphereModel = std::variant<CarnahanStarling, PercusYevickVirial, PercusYevickCompressibility>;

  // Enskog transport coefficients (hard spheres with d = kT = 1)

  namespace transport {

    [[nodiscard]] inline auto bulk_viscosity(double density, double chi) -> double {
      return (5.0 / (16.0 * std::sqrt(std::numbers::pi))) * (64.0 / 45.0) * std::numbers::pi * density * density *
          chi;
    }

    [[nodiscard]] inline auto shear_viscosity(double density, double chi) -> double {
      double term = 1.0 + 4.0 * std::numbers::pi * density * chi / 15.0;
      return (5.0 / (16.0 * std::sqrt(std::numbers::pi) * chi)) * term * term +
          0.6 * bulk_viscosity(density, chi);
    }

    [[nodiscard]] inline auto thermal_conductivity(double density, double chi) -> double {
      double term = 1.0 + 0.4 * std::numbers::pi * density * chi;
      return (75.0 / (64.0 * std::sqrt(std::numbers::pi) * chi)) * term * term +
          (128.0 / 225.0) * std::numbers::pi * density * density * chi;
    }

    [[nodiscard]] inline auto sound_damping(double density, double chi) -> double {
      constexpr double CV = 1.5;
      constexpr double CP = 2.5;
      constexpr double GAMMA = CP / CV;
      return (((GAMMA - 1.0) / CP) * thermal_conductivity(density, chi) +
              (4.0 / 3.0) * shear_viscosity(density, chi) + bulk_viscosity(density, chi)) /
          density;
    }

  }  // namespace transport

}  // namespace dft::physics::hard_spheres

#endif  // DFT_PHYSICS_HARD_SPHERES_HPP
