#include "dft/physics/eos.hpp"

#include <numbers>

namespace dft::physics::eos {

  // JZG factory

  auto make_lennard_jones_jzg(double kT, double cutoff_radius, bool shifted) -> LennardJonesJZG {
    if (kT <= 0.0) {
      throw std::invalid_argument("make_lennard_jones_jzg: temperature must be positive");
    }
    double tc = 0.0;
    if (cutoff_radius > 0.0) {
      tc = -(32.0 / 9.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - 1.5 * std::pow(cutoff_radius, -3.0));
      if (!shifted) {
        tc += (8.0 / 3.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - std::pow(cutoff_radius, -3.0));
      }
    }
    return LennardJonesJZG{.kT = kT, .tail_correction = tc};
  }

  // Mecke factory

  auto make_lennard_jones_mecke(double kT, double cutoff_radius, bool shifted) -> LennardJonesMecke {
    if (kT <= 0.0) {
      throw std::invalid_argument("make_lennard_jones_mecke: temperature must be positive");
    }
    double tc = 0.0;
    if (cutoff_radius > 0.0) {
      tc = -(32.0 / 9.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - 1.5 * std::pow(cutoff_radius, -3.0));
      if (!shifted) {
        tc += (8.0 / 3.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - std::pow(cutoff_radius, -3.0));
      }
    }
    return LennardJonesMecke{.kT = kT, .tail_correction = tc};
  }

  // JZG a_i and b_i coefficient functions

  auto LennardJonesJZG::a_coeff(int i) const -> double {
    double t = kT;
    double t_inv = 1.0 / t;
    double t_inv2 = t_inv * t_inv;
    switch (i) {
      case 1:
        return X[0] * t + X[1] * std::sqrt(t) + X[2] + X[3] * t_inv + X[4] * t_inv2;
      case 2:
        return X[5] * t + X[6] + X[7] * t_inv + X[8] * t_inv2;
      case 3:
        return X[9] * t + X[10] + X[11] * t_inv;
      case 4:
        return X[12];
      case 5:
        return X[13] * t_inv + X[14] * t_inv2;
      case 6:
        return X[15] * t_inv;
      case 7:
        return X[16] * t_inv + X[17] * t_inv2;
      case 8:
        return X[18] * t_inv2;
      default:
        throw std::runtime_error("LennardJonesJZG: invalid a_coeff index");
    }
  }

  auto LennardJonesJZG::b_coeff(int i) const -> double {
    double t_inv = 1.0 / kT;
    double t_inv2 = t_inv * t_inv;
    double t_inv3 = t_inv * t_inv2;
    double t_inv4 = t_inv2 * t_inv2;
    switch (i) {
      case 1:
        return X[19] * t_inv2 + X[20] * t_inv3;
      case 2:
        return X[21] * t_inv2 + X[22] * t_inv4;
      case 3:
        return X[23] * t_inv2 + X[24] * t_inv3;
      case 4:
        return X[25] * t_inv2 + X[26] * t_inv4;
      case 5:
        return X[27] * t_inv2 + X[28] * t_inv3;
      case 6:
        return X[29] * t_inv2 + X[30] * t_inv3 + X[31] * t_inv4;
      default:
        throw std::runtime_error("LennardJonesJZG: invalid b_coeff index");
    }
  }

  // JZG G-integral (recursive formula)

  template <typename T>
  auto LennardJonesJZG::g_integral(T rho, int i) const -> T {
    using std::exp;
    T f_val = exp(-GAMMA * rho * rho);
    T g_prev = (T(1.0) - f_val) / (T(2.0) * GAMMA);
    if (i == 1) {
      return g_prev;
    }
    for (int k = 2; k <= i; ++k) {
      using autodiff::detail::pow;
      using std::pow;
      T rho_pow = pow(rho, 2 * (k - 1));
      g_prev = -(f_val * rho_pow - T(2.0 * (k - 1)) * g_prev) / (T(2.0) * GAMMA);
    }
    return g_prev;
  }

  // JZG excess free energy (template, explicit instantiations below)

  template <typename T>
  auto LennardJonesJZG::excess_free_energy(T density) const -> T {
    using autodiff::detail::pow;
    using std::pow;
    T f = T(0.0);
    for (int i = 1; i <= 8; ++i) {
      f = f + T(a_coeff(i)) * pow(density, i) / T(static_cast<double>(i));
    }
    for (int i = 1; i <= 6; ++i) {
      f = f + T(b_coeff(i)) * g_integral(density, i);
    }
    f = f + T(tail_correction) * density;
    return f / T(kT);
  }

  template auto LennardJonesJZG::g_integral<double>(double, int) const -> double;
  template auto LennardJonesJZG::g_integral<math::dual>(math::dual, int) const -> math::dual;
  template auto LennardJonesJZG::g_integral<math::dual2nd>(math::dual2nd, int) const -> math::dual2nd;
  template auto LennardJonesJZG::g_integral<math::dual3rd>(math::dual3rd, int) const -> math::dual3rd;

  template auto LennardJonesJZG::excess_free_energy<double>(double) const -> double;
  template auto LennardJonesJZG::excess_free_energy<math::dual>(math::dual) const -> math::dual;
  template auto LennardJonesJZG::excess_free_energy<math::dual2nd>(math::dual2nd) const -> math::dual2nd;
  template auto LennardJonesJZG::excess_free_energy<math::dual3rd>(math::dual3rd) const -> math::dual3rd;

  // Mecke excess free energy (template, explicit instantiations below)

  template <typename T>
  auto LennardJonesMecke::excess_free_energy(T density) const -> T {
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

  template auto LennardJonesMecke::excess_free_energy<double>(double) const -> double;
  template auto LennardJonesMecke::excess_free_energy<math::dual>(math::dual) const -> math::dual;
  template auto LennardJonesMecke::excess_free_energy<math::dual2nd>(math::dual2nd) const -> math::dual2nd;
  template auto LennardJonesMecke::excess_free_energy<math::dual3rd>(math::dual3rd) const -> math::dual3rd;

}  // namespace dft::physics::eos
