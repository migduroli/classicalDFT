#include "classicaldft_bits/physics/thermodynamics/eos.h"

#include <cmath>
#include <numbers>

namespace dft_core::physics::thermodynamics::eos {

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
    double T = kT_;
    double T_inv = 1.0 / T;
    double T_inv2 = T_inv * T_inv;
    switch (i) {
      case 1:
        return x_[0] * T + x_[1] * std::sqrt(T) + x_[2] + x_[3] * T_inv + x_[4] * T_inv2;
      case 2:
        return x_[5] * T + x_[6] + x_[7] * T_inv + x_[8] * T_inv2;
      case 3:
        return x_[9] * T + x_[10] + x_[11] * T_inv;
      case 4:
        return x_[12];
      case 5:
        return x_[13] * T_inv + x_[14] * T_inv2;
      case 6:
        return x_[15] * T_inv;
      case 7:
        return x_[16] * T_inv + x_[17] * T_inv2;
      case 8:
        return x_[18] * T_inv2;
      default:
        throw std::runtime_error("LennardJonesJZG: invalid a_coeff index");
    }
  }

  double LennardJonesJZG::b_coeff(int i) const {
    double T_inv = 1.0 / kT_;
    double T_inv2 = T_inv * T_inv;
    double T_inv3 = T_inv * T_inv2;
    double T_inv4 = T_inv2 * T_inv2;
    switch (i) {
      case 1:
        return x_[19] * T_inv2 + x_[20] * T_inv3;
      case 2:
        return x_[21] * T_inv2 + x_[22] * T_inv4;
      case 3:
        return x_[23] * T_inv2 + x_[24] * T_inv3;
      case 4:
        return x_[25] * T_inv2 + x_[26] * T_inv4;
      case 5:
        return x_[27] * T_inv2 + x_[28] * T_inv3;
      case 6:
        return x_[29] * T_inv2 + x_[30] * T_inv3 + x_[31] * T_inv4;
      default:
        throw std::runtime_error("LennardJonesJZG: invalid b_coeff index");
    }
  }

  double LennardJonesJZG::G_integral(double rho, int i) const {
    double F = std::exp(-gamma_ * rho * rho);
    double G_prev = (1.0 - F) / (2.0 * gamma_);
    if (i == 1)
      return G_prev;
    for (int k = 2; k <= i; ++k) {
      double rho_pow = std::pow(rho, 2 * (k - 1));
      G_prev = -(F * rho_pow - 2.0 * (k - 1) * G_prev) / (2.0 * gamma_);
    }
    return G_prev;
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
    double F = std::exp(-gamma_ * density * density);
    double f = 0.0;
    for (int i = 1; i <= 8; ++i) {
      f += a_coeff(i) * std::pow(density, i - 1);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * F * std::pow(density, 2 * i - 1);
    }
    f += tail_correction_;
    return f / kT_;
  }

  double LennardJonesJZG::d2_excess_free_energy_per_particle(double density) const {
    double F = std::exp(-gamma_ * density * density);
    double f = 0.0;
    for (int i = 2; i <= 8; ++i) {
      f += a_coeff(i) * (i - 1) * std::pow(density, i - 2);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * F * ((2 * i - 1) * std::pow(density, 2 * i - 2) - 2.0 * gamma_ * std::pow(density, 2 * i));
    }
    return f / kT_;
  }

  double LennardJonesJZG::d3_excess_free_energy_per_particle(double density) const {
    double F = std::exp(-gamma_ * density * density);
    double f = 0.0;
    for (int i = 3; i <= 8; ++i) {
      f += a_coeff(i) * (i - 1) * (i - 2) * std::pow(density, i - 3);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * (-2.0 * gamma_ * density) * F *
          ((2 * i - 1) * std::pow(density, 2 * i - 2) - 2.0 * gamma_ * std::pow(density, 2 * i));
    }
    for (int i = 2; i <= 6; ++i) {
      f += b_coeff(i) * F * (2 * i - 1) * (2 * i - 2) * std::pow(density, 2 * i - 3);
    }
    for (int i = 1; i <= 6; ++i) {
      f += b_coeff(i) * F * (-4.0 * i * gamma_) * std::pow(density, 2 * i - 1);
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
    double rho_r = density / rho_c_;
    double T_r = kT_ / kT_c_;

    // Hard-sphere contribution via Carnahan-Starling with effective packing
    double s = 0.1617 * rho_r / (0.689 + 0.311 * std::pow(T_r, 0.3674));
    double fhs = s * (4.0 - 3.0 * s) / ((1.0 - s) * (1.0 - s));

    // Attractive/correction terms
    double fa = 0.0;
    for (const auto& t : terms_) {
      double term_val = t.c * std::pow(T_r, t.m) * std::pow(rho_r, t.n);
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

}  // namespace dft_core::physics::thermodynamics::eos
