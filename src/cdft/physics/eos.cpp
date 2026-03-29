#include "cdft/physics/eos.hpp"

#include <cmath>
#include <numbers>

namespace cdft::physics {

  // ── LennardJonesJZG ──────────────────────────────────────────────────────

  LennardJonesJZG::LennardJonesJZG(double kT, double cutoff_radius, bool shifted) : kT(kT) {
    if (kT <= 0.0) throw std::invalid_argument("Temperature must be positive");
    if (cutoff_radius > 0.0) {
      tail_correction =
          -(32.0 / 9.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - 1.5 * std::pow(cutoff_radius, -3.0));
      if (!shifted) {
        tail_correction +=
            (8.0 / 3.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - std::pow(cutoff_radius, -3.0));
      }
    }
  }

  double LennardJonesJZG::a_coeff(int i) const {
    double t = kT;
    double t_inv = 1.0 / t;
    double t_inv2 = t_inv * t_inv;
    switch (i) {
      case 1: return X[0] * t + X[1] * std::sqrt(t) + X[2] + X[3] * t_inv + X[4] * t_inv2;
      case 2: return X[5] * t + X[6] + X[7] * t_inv + X[8] * t_inv2;
      case 3: return X[9] * t + X[10] + X[11] * t_inv;
      case 4: return X[12];
      case 5: return X[13] * t_inv + X[14] * t_inv2;
      case 6: return X[15] * t_inv;
      case 7: return X[16] * t_inv + X[17] * t_inv2;
      case 8: return X[18] * t_inv2;
      default: throw std::invalid_argument("LennardJonesJZG: invalid a_coeff index");
    }
  }

  double LennardJonesJZG::b_coeff(int i) const {
    double t_inv = 1.0 / kT;
    double t_inv2 = t_inv * t_inv;
    double t_inv3 = t_inv * t_inv2;
    double t_inv4 = t_inv2 * t_inv2;
    switch (i) {
      case 1: return X[19] * t_inv2 + X[20] * t_inv3;
      case 2: return X[21] * t_inv2 + X[22] * t_inv4;
      case 3: return X[23] * t_inv2 + X[24] * t_inv3;
      case 4: return X[25] * t_inv2 + X[26] * t_inv4;
      case 5: return X[27] * t_inv2 + X[28] * t_inv3;
      case 6: return X[29] * t_inv2 + X[30] * t_inv3 + X[31] * t_inv4;
      default: throw std::invalid_argument("LennardJonesJZG: invalid b_coeff index");
    }
  }

  template <typename T>
  T LennardJonesJZG::G_integral(T rho, int i) const {
    using std::exp;
    using std::pow;
    T f_val = exp(-GAMMA * rho * rho);
    T g_prev = (T(1.0) - f_val) / (T(2.0) * GAMMA);
    if (i == 1) return g_prev;
    for (int k = 2; k <= i; ++k) {
      T rho_pow = pow(rho, 2 * (k - 1));
      g_prev = -(f_val * rho_pow - T(2.0 * (k - 1)) * g_prev) / (T(2.0) * GAMMA);
    }
    return g_prev;
  }

  template <typename T>
  T LennardJonesJZG::excess_free_energy_per_particle(T density) const {
    using std::pow;
    T f = T(0.0);
    for (int i = 1; i <= 8; ++i) {
      f += a_coeff(i) * pow(density, i) / static_cast<double>(i);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * G_integral(density, i);
    }
    f += tail_correction * density;
    return f / kT;
  }

  // Explicit instantiations for double and dual types
  template double LennardJonesJZG::excess_free_energy_per_particle(double) const;
  template cdft::dual LennardJonesJZG::excess_free_energy_per_particle(cdft::dual) const;
  template cdft::dual2nd LennardJonesJZG::excess_free_energy_per_particle(cdft::dual2nd) const;
  template cdft::dual3rd LennardJonesJZG::excess_free_energy_per_particle(cdft::dual3rd) const;

  // ── LennardJonesMecke ────────────────────────────────────────────────────

  LennardJonesMecke::LennardJonesMecke(double kT, double cutoff_radius, bool shifted) : kT(kT) {
    if (kT <= 0.0) throw std::invalid_argument("Temperature must be positive");
    if (cutoff_radius > 0.0) {
      tail_correction =
          -(32.0 / 9.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - 1.5 * std::pow(cutoff_radius, -3.0));
      if (!shifted) {
        tail_correction +=
            (8.0 / 3.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - std::pow(cutoff_radius, -3.0));
      }
    }
  }

  template <typename T>
  T LennardJonesMecke::excess_free_energy_per_particle(T density) const {
    using std::exp;
    using std::log;
    using std::pow;

    T rho_r = density / RHO_C;
    double t_r = kT / KT_C;

    T s = T(0.1617) * rho_r / (0.689 + 0.311 * std::pow(t_r, 0.3674));
    T fhs = s * (T(4.0) - T(3.0) * s) / ((T(1.0) - s) * (T(1.0) - s));

    T fa = T(0.0);
    for (const auto& t : TERMS) {
      T term_val = t.c * std::pow(t_r, t.m) * pow(rho_r, t.n);
      if (t.p != 0) {
        term_val *= exp(static_cast<double>(t.p) * pow(rho_r, t.q));
      }
      fa += term_val;
    }

    fa += tail_correction * density / kT;
    return fhs + fa;
  }

  // Explicit instantiations
  template double LennardJonesMecke::excess_free_energy_per_particle(double) const;
  template cdft::dual LennardJonesMecke::excess_free_energy_per_particle(cdft::dual) const;
  template cdft::dual2nd LennardJonesMecke::excess_free_energy_per_particle(cdft::dual2nd) const;
  template cdft::dual3rd LennardJonesMecke::excess_free_energy_per_particle(cdft::dual3rd) const;

}  // namespace cdft::physics
