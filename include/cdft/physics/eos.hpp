#pragma once

#include "cdft/numerics/autodiff.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <variant>

namespace cdft::physics {

  // ── Utility ───────────────────────────────────────────────────────────────

  template <typename T = double>
  [[nodiscard]] inline constexpr T packing_fraction(T density) noexcept {
    return (std::numbers::pi / 6.0) * density;
  }

  [[nodiscard]] inline constexpr double density_from_eta(double eta) noexcept {
    return 6.0 * eta / std::numbers::pi;
  }

  // ── Hard-sphere contact value (universal) ─────────────────────────────────

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

  // ── Derived thermodynamic quantities (free functions, work with any model) ─

  [[nodiscard]] inline double hs_excess_free_energy(const HardSphereModel& model, double eta) {
    return std::visit([eta](const auto& m) { return static_cast<double>(m.excess_free_energy(eta)); }, model);
  }

  [[nodiscard]] inline double hs_pressure(const HardSphereModel& model, double eta) {
    auto [f, df] = std::visit([eta](const auto& m) {
      return derivatives_up_to_1([&](dual x) { return m.excess_free_energy(x); }, eta);
    }, model);
    return 1.0 + eta * df;
  }

  [[nodiscard]] inline double hs_free_energy(const HardSphereModel& model, double density) {
    double eta = packing_fraction(density);
    return std::log(density) - 1.0 + hs_excess_free_energy(model, eta);
  }

  [[nodiscard]] inline double hs_chemical_potential(const HardSphereModel& model, double density) {
    double eta = packing_fraction(density);
    auto [f, df] = std::visit([eta](const auto& m) {
      return derivatives_up_to_1([&](dual x) { return m.excess_free_energy(x); }, eta);
    }, model);
    return std::log(density) + f + eta * df;
  }

  // ── Transport (Enskog theory, d = kT = 1) ────────────────────────────────

  namespace transport {

    [[nodiscard]] inline double bulk_viscosity(double density, double chi) {
      constexpr double prefactor = 5.0 / (16.0 * std::numbers::inv_sqrtpi * 0.5);  // 5/(16√π)
      return prefactor * (64.0 / 45.0) * std::numbers::pi * density * density * chi;
    }

    [[nodiscard]] inline double shear_viscosity(double density, double chi) {
      constexpr double prefactor = 5.0 / (16.0 * std::numbers::inv_sqrtpi * 0.5);
      double b = 1.0 + (4.0 / 15.0) * std::numbers::pi * density * chi;
      return prefactor * (b * b / chi) + 0.6 * bulk_viscosity(density, chi);
    }

    [[nodiscard]] inline double thermal_conductivity(double density, double chi) {
      double b = 1.0 + 0.4 * std::numbers::pi * density * chi;
      constexpr double prefactor = 75.0 / (64.0 * std::numbers::inv_sqrtpi * 0.5);
      return prefactor * (b * b / chi)
             + (128.0 / 225.0) * std::numbers::pi * density * density * chi;
    }

    [[nodiscard]] inline double sound_damping(double density, double chi) {
      constexpr double gamma = 5.0 / 3.0;
      constexpr double Cp = 2.5;
      double kappa = thermal_conductivity(density, chi);
      double eta_s = shear_viscosity(density, chi);
      double eta_b = bulk_viscosity(density, chi);
      return (1.0 / density) * ((gamma - 1.0) / Cp * kappa + (4.0 / 3.0) * eta_s + eta_b);
    }

  }  // namespace transport

  // ── Full equation of state (flat structs, temperature-dependent) ──────────

  struct IdealGas {
    double kT;

    explicit IdealGas(double kT) : kT(kT) {
      if (kT <= 0.0) throw std::invalid_argument("Temperature must be positive");
    }

    template <typename T = double>
    [[nodiscard]] static T excess_free_energy_per_particle(T /*density*/) { return T(0.0); }
    [[nodiscard]] static std::string name() { return "IdealGas"; }
  };

  struct PercusYevickEOS {
    double kT;

    explicit PercusYevickEOS(double kT) : kT(kT) {
      if (kT <= 0.0) throw std::invalid_argument("Temperature must be positive");
    }

    template <typename T = double>
    [[nodiscard]] T excess_free_energy_per_particle(T density) const {
      return PercusYevickCompressibility::excess_free_energy(packing_fraction(density));
    }
    [[nodiscard]] static std::string name() { return "PercusYevickEOS"; }
  };

  // Lennard-Jones EOS models need source files (complex parametrizations)
  struct LennardJonesJZG {
    double kT;
    double tail_correction = 0.0;

    LennardJonesJZG(double kT, double cutoff_radius = -1.0, bool shifted = false);

    template <typename T = double>
    [[nodiscard]] T excess_free_energy_per_particle(T density) const;

    [[nodiscard]] static std::string name() { return "LennardJonesJZG"; }

   private:
    [[nodiscard]] double a_coeff(int i) const;
    [[nodiscard]] double b_coeff(int i) const;

    template <typename T>
    [[nodiscard]] T G_integral(T rho, int i) const;

    static constexpr double GAMMA = 3.0;
    static constexpr std::array<double, 32> X = {
        0.8623085097507421,   2.976218765822098,    -8.402230115796038,   0.1054136629203555,   -0.8564583828174598,
        1.582759470107601,    0.7639421948305453,   1.753173414312048,    2.798291772190376e3,  -4.8394220260857657e-2,
        0.9963265197721935,   -3.698000291272493e1, 2.084012299434647e1,  8.305402124717285e1,  -9.574799715203068e2,
        -1.477746229234994e2, 6.398607852471505e1,  1.603993673294834e1,  6.805916615864377e1,  -2.791293578795945e3,
        -6.245128304568454,   -8.116836104958410e3, 1.488735559561229e1,  -1.059346754655084e4, -1.131607632802822e2,
        -8.867771540418822e3, -3.986982844450543e1, -4.689270299917261e3, 2.593535277438717e2,  -2.694523589434903e3,
        -7.218487631550215e2, 1.721802063863269e2,
    };
  };

  struct LennardJonesMecke {
    double kT;
    double tail_correction = 0.0;

    LennardJonesMecke(double kT, double cutoff_radius = -1.0, bool shifted = false);

    template <typename T = double>
    [[nodiscard]] T excess_free_energy_per_particle(T density) const;

    [[nodiscard]] static std::string name() { return "LennardJonesMecke"; }

   private:
    static constexpr double RHO_C = 0.3107;
    static constexpr double KT_C = 1.328;

    struct Term {
      double c;
      double m;
      int n;
      int p;
      int q;
    };

    static constexpr std::array<Term, 32> TERMS = {{
        {0.33619760720e-05, -2, 9, 0, 0},    {-0.14707220591e+01, -1, 1, 0, 0},   {-0.11972121043e+00, -1, 2, 0, 0},
        {-0.11350363539e-04, -1, 9, 0, 0},   {-0.26778688896e-04, -0.5, 8, 0, 0}, {0.12755936511e-05, -0.5, 10, 0, 0},
        {0.40088615477e-02, 0.5, 1, 0, 0},   {0.52305580273e-05, 0.5, 7, 0, 0},   {-0.10214454556e-07, 1, 10, 0, 0},
        {-0.14526799362e-01, -5, 1, -1, 1},  {0.64975356409e-01, -4, 1, -1, 1},   {-0.60304755494e-01, -2, 1, -1, 1},
        {-0.14925537332e+00, -2, 2, -1, 1},  {-0.31664355868e-03, -2, 8, -1, 1},  {0.28312781935e-01, -1, 1, -1, 1},
        {0.13039603845e-03, -1, 10, -1, 1},  {0.10121435381e-01, 0, 4, -1, 1},    {-0.15425936014e-04, 0, 9, -1, 1},
        {-0.61568007279e-01, -5, 2, -1, 2},  {0.76001994423e-02, -4, 5, -1, 2},   {-0.18906040708e+00, -3, 1, -1, 2},
        {0.33141311846e+00, -2, 2, -1, 2},   {-0.25229604842e+00, -2, 3, -1, 2},  {0.13145401812e+00, -2, 4, -1, 2},
        {-0.48672350917e-01, -1, 2, -1, 2},  {0.14756043863e-02, -10, 3, -1, 3},  {-0.85996667747e-02, -6, 4, -1, 3},
        {0.33880247915e-01, -4, 2, -1, 3},   {0.69427495094e-02, 0, 2, -1, 3},    {-0.22271531045e-07, -24, 5, -1, 4},
        {-0.22656880018e-03, -10, 2, -1, 4}, {0.24056013779e-02, -2, 10, -1, 4},
    }};
  };

  // ── EOS sum type ──────────────────────────────────────────────────────────

  using EquationOfState = std::variant<IdealGas, PercusYevickEOS, LennardJonesJZG, LennardJonesMecke>;

  // ── Derived quantities (free functions, work with any EOS) ────────────────

  [[nodiscard]] inline double eos_excess_free_energy_per_particle(const EquationOfState& eos, double density) {
    return std::visit([density](const auto& m) {
      return static_cast<double>(m.excess_free_energy_per_particle(density));
    }, eos);
  }

  [[nodiscard]] inline double eos_free_energy_per_particle(const EquationOfState& eos, double density) {
    return std::log(density) - 1.0 + eos_excess_free_energy_per_particle(eos, density);
  }

  [[nodiscard]] inline double eos_excess_free_energy_density(const EquationOfState& eos, double density) {
    return density * eos_excess_free_energy_per_particle(eos, density);
  }

  [[nodiscard]] inline double eos_pressure(const EquationOfState& eos, double density) {
    double d_phi = std::visit([density](const auto& m) {
      auto [f, df] = derivatives_up_to_1(
          [&](dual x) { return m.excess_free_energy_per_particle(x); }, density);
      return df;
    }, eos);
    return 1.0 + density * d_phi;
  }

  [[nodiscard]] inline double eos_temperature(const EquationOfState& eos) {
    return std::visit([](const auto& m) { return m.kT; }, eos);
  }

  [[nodiscard]] inline std::string eos_name(const EquationOfState& eos) {
    return std::visit([](const auto& m) { return m.name(); }, eos);
  }

}  // namespace cdft::physics
