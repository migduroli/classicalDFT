#include "classicaldft_bits/thermodynamics/eos.h"

#include <cmath>
#include <numbers>

namespace dft::thermodynamics::eos {

  // ── LennardJonesJZG ──────────────────────────────────────────────────────

  LennardJonesJZG::LennardJonesJZG(double kT, double cutoff_radius, bool shifted) : EquationOfState(kT) {
    if (cutoff_radius > 0.0) {
      tail_correction_ =
          -(32.0 / 9.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - 1.5 * std::pow(cutoff_radius, -3.0));
      if (!shifted) {
        tail_correction_ +=
            (8.0 / 3.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - std::pow(cutoff_radius, -3.0));
      }
    }
  }

  double LennardJonesJZG::a_coeff(int i) const {
    double t = kT_;
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

  double LennardJonesJZG::b_coeff(int i) const {
    double t_inv = 1.0 / kT_;
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

  double LennardJonesJZG::G_integral(double rho, int i) const {
    double f_val = std::exp(-GAMMA * rho * rho);
    double g_prev = (1.0 - f_val) / (2.0 * GAMMA);
    if (i == 1)
      return g_prev;
    for (int k = 2; k <= i; ++k) {
      double rho_pow = std::pow(rho, 2 * (k - 1));
      g_prev = -(f_val * rho_pow - 2.0 * (k - 1) * g_prev) / (2.0 * GAMMA);
    }
    return g_prev;
  }

  double LennardJonesJZG::excess_free_energy_per_particle(double density) const {
    double f = 0.0;
    for (int i = 1; i <= 8; ++i) {
      f += a_coeff(i) * std::pow(density, i) / static_cast<double>(i);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * G_integral(density, i);
    }
    f += tail_correction_ * density;
    return f / kT_;
  }

  double LennardJonesJZG::d_excess_free_energy_per_particle(double density) const {
    double f_val = std::exp(-GAMMA * density * density);
    double f = 0.0;
    for (int i = 1; i <= 8; ++i) {
      f += a_coeff(i) * std::pow(density, i - 1);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * f_val * std::pow(density, 2 * i - 1);
    }
    f += tail_correction_;
    return f / kT_;
  }

  double LennardJonesJZG::d2_excess_free_energy_per_particle(double density) const {
    double f_val = std::exp(-GAMMA * density * density);
    double f = 0.0;
    for (int i = 2; i <= 8; ++i) {
      f += a_coeff(i) * (i - 1) * std::pow(density, i - 2);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * f_val * ((2 * i - 1) * std::pow(density, 2 * i - 2) - 2.0 * GAMMA * std::pow(density, 2 * i));
    }
    return f / kT_;
  }

  double LennardJonesJZG::d3_excess_free_energy_per_particle(double density) const {
    double f_val = std::exp(-GAMMA * density * density);
    double f = 0.0;
    for (int i = 3; i <= 8; ++i) {
      f += a_coeff(i) * (i - 1) * (i - 2) * std::pow(density, i - 3);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * (-2.0 * GAMMA * density) * f_val *
          ((2 * i - 1) * std::pow(density, 2 * i - 2) - 2.0 * GAMMA * std::pow(density, 2 * i));
    }
    for (int i = 2; i <= 6; ++i) {
      f += b_coeff(i) * f_val * (2 * i - 1) * (2 * i - 2) * std::pow(density, 2 * i - 3);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * f_val * (-4.0 * i * GAMMA) * std::pow(density, 2 * i - 1);
    }
    return f / kT_;
  }

  // ── LennardJonesMecke ────────────────────────────────────────────────────

  LennardJonesMecke::LennardJonesMecke(double kT, double cutoff_radius, bool shifted) : EquationOfState(kT) {
    if (cutoff_radius > 0.0) {
      tail_correction_ =
          -(32.0 / 9.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - 1.5 * std::pow(cutoff_radius, -3.0));
      if (!shifted) {
        tail_correction_ +=
            (8.0 / 3.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - std::pow(cutoff_radius, -3.0));
      }
    }
  }

  double LennardJonesMecke::excess_free_energy_per_particle(double density) const {
    double rho_r = density / RHO_C;
    double t_r = kT_ / KT_C;

    // Hard-sphere contribution via Carnahan-Starling with effective packing
    double s = 0.1617 * rho_r / (0.689 + 0.311 * std::pow(t_r, 0.3674));
    double fhs = s * (4.0 - 3.0 * s) / ((1.0 - s) * (1.0 - s));

    // Attractive/correction terms
    double fa = 0.0;
    for (const auto& t : TERMS) {
      double term_val = t.c * std::pow(t_r, t.m) * std::pow(rho_r, t.n);
      if (t.p != 0) {
        term_val *= std::exp(static_cast<double>(t.p) * std::pow(rho_r, t.q));
      }
      fa += term_val;
    }

    fa += tail_correction_ * density / kT_;
    return fhs + fa;
  }

  double LennardJonesMecke::d_excess_free_energy_per_particle(double density) const {
    double h = density * 1e-7;
    if (h < 1e-12)
      h = 1e-12;
    return (excess_free_energy_per_particle(density + h) - excess_free_energy_per_particle(density - h)) / (2.0 * h);
  }

  double LennardJonesMecke::d2_excess_free_energy_per_particle(double density) const {
    double h = density * 1e-5;
    if (h < 1e-10)
      h = 1e-10;
    return (excess_free_energy_per_particle(density + h) - 2.0 * excess_free_energy_per_particle(density) +
            excess_free_energy_per_particle(density - h)) /
        (h * h);
  }

  double LennardJonesMecke::d3_excess_free_energy_per_particle(double density) const {
    double h = density * 1e-4;
    if (h < 1e-8)
      h = 1e-8;
    return (-excess_free_energy_per_particle(density - 2 * h) + 2.0 * excess_free_energy_per_particle(density - h) -
            2.0 * excess_free_energy_per_particle(density + h) + excess_free_energy_per_particle(density + 2 * h)) /
        (2.0 * h * h * h);
  }

}  // namespace dft::thermodynamics::eos
