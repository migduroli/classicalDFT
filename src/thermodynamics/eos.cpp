#include "dft/thermodynamics/eos.h"

#include <cmath>
#include <numbers>

namespace dft::thermodynamics::eos {

  // ── LennardJonesJZG ──────────────────────────────────────────────────────

  LennardJonesJZG::LennardJonesJZG(double kT, double cutoff_radius, bool shifted) : kT(kT) {
    if (kT <= 0.0)
      throw std::invalid_argument("Temperature must be positive");
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

  // ── LennardJonesMecke ────────────────────────────────────────────────────

  LennardJonesMecke::LennardJonesMecke(double kT, double cutoff_radius, bool shifted) : kT(kT) {
    if (kT <= 0.0)
      throw std::invalid_argument("Temperature must be positive");
    if (cutoff_radius > 0.0) {
      tail_correction =
          -(32.0 / 9.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - 1.5 * std::pow(cutoff_radius, -3.0));
      if (!shifted) {
        tail_correction +=
            (8.0 / 3.0) * std::numbers::pi * (std::pow(cutoff_radius, -9.0) - std::pow(cutoff_radius, -3.0));
      }
    }
  }

}  // namespace dft::thermodynamics::eos
