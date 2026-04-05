#ifndef DFT_PHYSICS_EOS_HPP
#define DFT_PHYSICS_EOS_HPP

#include "dft/math/autodiff.hpp"
#include "dft/physics/hard_spheres.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace dft::physics::eos {

  // Ideal gas (no excess free energy)

  struct IdealGas {
    static constexpr std::string_view NAME = "IdealGas";

    double kT;

    template <typename T = double>
    [[nodiscard]] auto excess_free_energy(T /*density*/) const -> T {
      return T(0.0);
    }

    [[nodiscard]] auto d_excess_free_energy(double /*density*/) const -> double { return 0.0; }

    [[nodiscard]] auto d2_excess_free_energy(double /*density*/) const -> double { return 0.0; }

    [[nodiscard]] auto free_energy(double density) const -> double { return std::log(density) - 1.0; }

    [[nodiscard]] auto excess_chemical_potential(double /*density*/) const -> double { return 0.0; }

    [[nodiscard]] auto chemical_potential(double density) const -> double { return std::log(density); }

    [[nodiscard]] auto pressure(double /*density*/) const -> double { return 1.0; }
  };

  // Percus-Yevick (compressibility route, wraps hard-sphere PY)

  struct PercusYevick {
    static constexpr std::string_view NAME = "PercusYevick";

    double kT;

    template <typename T = double>
    [[nodiscard]] auto excess_free_energy(T density) const -> T {
      T eta = hard_spheres::packing_fraction(density);
      return hard_spheres::PercusYevickCompressibility::excess_free_energy(eta);
    }

    [[nodiscard]] auto d_excess_free_energy(double density) const -> double {
      auto [f, df] =
          math::derivatives_up_to_1([this](math::dual x) -> math::dual { return excess_free_energy(x); }, density);
      return df;
    }

    [[nodiscard]] auto d2_excess_free_energy(double density) const -> double {
      auto [f, df, d2f] = math::derivatives_up_to_2(
          [this](math::dual2nd x) -> math::dual2nd { return excess_free_energy(x); },
          density
      );
      return d2f;
    }

    [[nodiscard]] auto free_energy(double density) const -> double {
      return std::log(density) - 1.0 + static_cast<double>(excess_free_energy(density));
    }

    [[nodiscard]] auto excess_chemical_potential(double density) const -> double {
      return static_cast<double>(excess_free_energy(density)) + density * d_excess_free_energy(density);
    }

    [[nodiscard]] auto chemical_potential(double density) const -> double {
      return std::log(density) + excess_chemical_potential(density);
    }

    [[nodiscard]] auto pressure(double density) const -> double {
      return 1.0 + density * d_excess_free_energy(density);
    }
  };

  // Lennard-Jones Johnson-Zollweg-Gubbins (JZG) 32-parameter EOS

  struct LennardJonesJZG {
    static constexpr std::string_view NAME = "LennardJonesJZG";

    double kT;
    double tail_correction{ 0.0 };

    template <typename T = double>
    [[nodiscard]] auto excess_free_energy(T density) const -> T;

    [[nodiscard]] auto a_coeff(int i) const -> double;
    [[nodiscard]] auto b_coeff(int i) const -> double;

    template <typename T>
    [[nodiscard]] auto g_integral(T rho, int i) const -> T;

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

    [[nodiscard]] auto d_excess_free_energy(double density) const -> double {
      auto [f, df] =
          math::derivatives_up_to_1([this](math::dual x) -> math::dual { return excess_free_energy(x); }, density);
      return df;
    }

    [[nodiscard]] auto d2_excess_free_energy(double density) const -> double {
      auto [f, df, d2f] = math::derivatives_up_to_2(
          [this](math::dual2nd x) -> math::dual2nd { return excess_free_energy(x); },
          density
      );
      return d2f;
    }

    [[nodiscard]] auto free_energy(double density) const -> double {
      return std::log(density) - 1.0 + static_cast<double>(excess_free_energy(density));
    }

    [[nodiscard]] auto excess_chemical_potential(double density) const -> double {
      return static_cast<double>(excess_free_energy(density)) + density * d_excess_free_energy(density);
    }

    [[nodiscard]] auto chemical_potential(double density) const -> double {
      return std::log(density) + excess_chemical_potential(density);
    }

    [[nodiscard]] auto pressure(double density) const -> double {
      return 1.0 + density * d_excess_free_energy(density);
    }
  };

  // Lennard-Jones Mecke et al. EOS

  struct LennardJonesMecke {
    static constexpr std::string_view NAME = "LennardJonesMecke";

    double kT;
    double tail_correction{ 0.0 };

    template <typename T = double>
    [[nodiscard]] auto excess_free_energy(T density) const -> T;

    static constexpr double RHO_C = 0.3107;
    static constexpr double KT_C = 1.328;

    struct Term {
      double c;
      double m;
      int n;
      int p;
      int q;
    };

    static constexpr std::array<Term, 32> TERMS = { {
        { 0.33619760720e-05, -2, 9, 0, 0 },    { -0.14707220591e+01, -1, 1, 0, 0 },
        { -0.11972121043e+00, -1, 2, 0, 0 },   { -0.11350363539e-04, -1, 9, 0, 0 },
        { -0.26778688896e-04, -0.5, 8, 0, 0 }, { 0.12755936511e-05, -0.5, 10, 0, 0 },
        { 0.40088615477e-02, 0.5, 1, 0, 0 },   { 0.52305580273e-05, 0.5, 7, 0, 0 },
        { -0.10214454556e-07, 1, 10, 0, 0 },   { -0.14526799362e-01, -5, 1, -1, 1 },
        { 0.64975356409e-01, -4, 1, -1, 1 },   { -0.60304755494e-01, -2, 1, -1, 1 },
        { -0.14925537332e+00, -2, 2, -1, 1 },  { -0.31664355868e-03, -2, 8, -1, 1 },
        { 0.28312781935e-01, -1, 1, -1, 1 },   { 0.13039603845e-03, -1, 10, -1, 1 },
        { 0.10121435381e-01, 0, 4, -1, 1 },    { -0.15425936014e-04, 0, 9, -1, 1 },
        { -0.61568007279e-01, -5, 2, -1, 2 },  { 0.76001994423e-02, -4, 5, -1, 2 },
        { -0.18906040708e+00, -3, 1, -1, 2 },  { 0.33141311846e+00, -2, 2, -1, 2 },
        { -0.25229604842e+00, -2, 3, -1, 2 },  { 0.13145401812e+00, -2, 4, -1, 2 },
        { -0.48672350917e-01, -1, 2, -1, 2 },  { 0.14756043863e-02, -10, 3, -1, 3 },
        { -0.85996667747e-02, -6, 4, -1, 3 },  { 0.33880247915e-01, -4, 2, -1, 3 },
        { 0.69427495094e-02, 0, 2, -1, 3 },    { -0.22271531045e-07, -24, 5, -1, 4 },
        { -0.22656880018e-03, -10, 2, -1, 4 }, { 0.24056013779e-02, -2, 10, -1, 4 },
    } };

    [[nodiscard]] auto d_excess_free_energy(double density) const -> double {
      auto [f, df] =
          math::derivatives_up_to_1([this](math::dual x) -> math::dual { return excess_free_energy(x); }, density);
      return df;
    }

    [[nodiscard]] auto d2_excess_free_energy(double density) const -> double {
      auto [f, df, d2f] = math::derivatives_up_to_2(
          [this](math::dual2nd x) -> math::dual2nd { return excess_free_energy(x); },
          density
      );
      return d2f;
    }

    [[nodiscard]] auto free_energy(double density) const -> double {
      return std::log(density) - 1.0 + static_cast<double>(excess_free_energy(density));
    }

    [[nodiscard]] auto excess_chemical_potential(double density) const -> double {
      return static_cast<double>(excess_free_energy(density)) + density * d_excess_free_energy(density);
    }

    [[nodiscard]] auto chemical_potential(double density) const -> double {
      return std::log(density) + excess_chemical_potential(density);
    }

    [[nodiscard]] auto pressure(double density) const -> double {
      return 1.0 + density * d_excess_free_energy(density);
    }
  };

  using EosModel = std::variant<IdealGas, PercusYevick, LennardJonesJZG, LennardJonesMecke>;

  // Validated factories

  [[nodiscard]] inline auto make_ideal_gas(double kT) -> IdealGas {
    if (kT <= 0.0) {
      throw std::invalid_argument("make_ideal_gas: temperature must be positive");
    }
    return IdealGas{ .kT = kT };
  }

  [[nodiscard]] inline auto make_percus_yevick(double kT) -> PercusYevick {
    if (kT <= 0.0) {
      throw std::invalid_argument("make_percus_yevick: temperature must be positive");
    }
    return PercusYevick{ .kT = kT };
  }

  [[nodiscard]] auto make_lennard_jones_jzg(double kT, double cutoff_radius = -1.0, bool shifted = false)
      -> LennardJonesJZG;
  [[nodiscard]] auto make_lennard_jones_mecke(double kT, double cutoff_radius = -1.0, bool shifted = false)
      -> LennardJonesMecke;

}  // namespace dft::physics::eos

#endif  // DFT_PHYSICS_EOS_HPP
