#ifndef DFT_THERMODYNAMICS_EOS_H
#define DFT_THERMODYNAMICS_EOS_H

#include "dft/math/autodiff.h"
#include "dft/thermodynamics/enskog.h"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <variant>

namespace dft::thermodynamics::eos {

  // ── Ideal gas ─────────────────────────────────────────────────────────────

  struct IdealGas {
    double kT;

    explicit IdealGas(double kT) : kT(kT) {
      if (kT <= 0.0)
        throw std::invalid_argument("Temperature must be positive");
    }

    template <typename T = double>
    [[nodiscard]] T excess_free_energy(T /*density*/) const {
      return T(0.0);
    }

    [[nodiscard]] static std::string name() { return "IdealGas"; }
  };

  // ── Percus-Yevick (compressibility route) ─────────────────────────────────

  struct PercusYevick {
    double kT;

    explicit PercusYevick(double kT) : kT(kT) {
      if (kT <= 0.0)
        throw std::invalid_argument("Temperature must be positive");
    }

    template <typename T = double>
    [[nodiscard]] T excess_free_energy(T density) const {
      T eta = packing_fraction(density);
      return PercusYevickCompressibility::excess_free_energy(eta);
    }

    [[nodiscard]] static std::string name() { return "PercusYevick"; }
  };

  // ── Lennard-Jones: Johnson-Zollweg-Gubbins (JZG) ─────────────────────────

  struct LennardJonesJZG {
    double kT;
    double tail_correction = 0.0;

    LennardJonesJZG(double kT, double cutoff_radius = -1.0, bool shifted = false);

    template <typename T = double>
    [[nodiscard]] T excess_free_energy(T density) const;

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

  // ── Lennard-Jones: Mecke et al. ───────────────────────────────────────────

  struct LennardJonesMecke {
    double kT;
    double tail_correction = 0.0;

    LennardJonesMecke(double kT, double cutoff_radius = -1.0, bool shifted = false);

    template <typename T = double>
    [[nodiscard]] T excess_free_energy(T density) const;

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

  // ── JZG template members ──────────────────────────────────────────────────

  template <typename T>
  T LennardJonesJZG::G_integral(T rho, int i) const {
    using std::exp;
    T f_val = exp(-GAMMA * rho * rho);
    T g_prev = (T(1.0) - f_val) / (T(2.0) * GAMMA);
    if (i == 1)
      return g_prev;
    for (int k = 2; k <= i; ++k) {
      using autodiff::detail::pow;
      using std::pow;
      T rho_pow = pow(rho, 2 * (k - 1));
      g_prev = -(f_val * rho_pow - T(2.0 * (k - 1)) * g_prev) / (T(2.0) * GAMMA);
    }
    return g_prev;
  }

  template <typename T>
  T LennardJonesJZG::excess_free_energy(T density) const {
    using autodiff::detail::pow;
    using std::pow;
    T f = T(0.0);
    for (int i = 1; i <= 8; ++i) {
      f = f + T(a_coeff(i)) * pow(density, i) / T(static_cast<double>(i));
    }
    for (int i = 1; i <= 6; ++i) {
      f = f + T(b_coeff(i)) * G_integral(density, i);
    }
    f = f + T(tail_correction) * density;
    return f / T(kT);
  }

  // ── Mecke template members ────────────────────────────────────────────────

  template <typename T>
  T LennardJonesMecke::excess_free_energy(T density) const {
    using autodiff::detail::pow;
    using std::exp;
    using std::pow;
    T rho_r = density / T(RHO_C);
    double t_r = kT / KT_C;

    T s = T(0.1617) * rho_r / T(0.689 + 0.311 * std::pow(t_r, 0.3674));
    T fhs = s * (T(4.0) - T(3.0) * s) / ((T(1.0) - s) * (T(1.0) - s));

    T fa = T(0.0);
    for (const auto& t : TERMS) {
      T term_val = T(t.c * std::pow(t_r, t.m)) * pow(rho_r, t.n);
      if (t.p != 0) {
        term_val = term_val * exp(T(static_cast<double>(t.p)) * pow(rho_r, t.q));
      }
      fa = fa + term_val;
    }

    fa = fa + T(tail_correction) * density / T(kT);
    return fhs + fa;
  }

  // ── Sum type for equations of state ───────────────────────────────────────

  using EosModel = std::variant<IdealGas, PercusYevick, LennardJonesJZG, LennardJonesMecke>;

  // ── Free functions: operate on any model via std::visit + autodiff ────────

  [[nodiscard]] inline double eos_excess_free_energy(const EosModel& model, double density, int order = 0) {
    return std::visit(
        [density, order](const auto& m) {
          if (order == 0) {
            return static_cast<double>(m.excess_free_energy(density));
          }
          auto [f, df, d2f, d3f] =
              math::derivatives_up_to_3([&](math::dual3rd x) { return m.excess_free_energy(x); }, density);
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

  [[nodiscard]] inline double eos_temperature(const EosModel& model) {
    return std::visit([](const auto& m) { return m.kT; }, model);
  }

  [[nodiscard]] inline std::string eos_name(const EosModel& model) {
    return std::visit([](const auto& m) { return m.name(); }, model);
  }

  [[nodiscard]] inline double eos_free_energy_per_particle(const EosModel& model, double density) {
    return std::log(density) - 1.0 + eos_excess_free_energy(model, density);
  }

  [[nodiscard]] inline double eos_excess_free_energy_density(const EosModel& model, double density) {
    return density * eos_excess_free_energy(model, density);
  }

  [[nodiscard]] inline double eos_d_excess_free_energy_density(const EosModel& model, double density) {
    return eos_excess_free_energy(model, density) + density * eos_excess_free_energy(model, density, 1);
  }

  [[nodiscard]] inline double eos_d2_excess_free_energy_density(const EosModel& model, double density) {
    return 2.0 * eos_excess_free_energy(model, density, 1) + density * eos_excess_free_energy(model, density, 2);
  }

  [[nodiscard]] inline double eos_pressure(const EosModel& model, double density) {
    return 1.0 + density * eos_excess_free_energy(model, density, 1);
  }

}  // namespace dft::thermodynamics::eos

#endif  // DFT_THERMODYNAMICS_EOS_H
