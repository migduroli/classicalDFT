#include "classicaldft_bits/physics/fmt/functional.h"

#include <numbers>

namespace dft_core::physics::fmt {

  double Functional::phi(const Measures& m) const {
    double e = m.eta;
    return -m.n0 * f1(e) + (m.n1 * m.n2 - m.v1_dot_v2) * f2(e) + phi3(m) * f3(e);
  }

  Measures Functional::d_phi(const Measures& m) const {
    Measures dm;
    double e = m.eta;
    double F1 = f1(e);
    double F2 = f2(e);
    double F3 = f3(e);
    double dF1 = d_f1(e);
    double dF2 = d_f2(e);
    double dF3 = d_f3(e);
    double P3 = phi3(m);

    dm.eta = -m.n0 * dF1 + (m.n1 * m.n2 - m.v1_dot_v2) * dF2 + P3 * dF3;
    dm.n0 = -F1;
    dm.n1 = m.n2 * F2;
    dm.n2 = m.n1 * F2 + d_phi3_d_n2(m) * F3;
    dm.v1 = -m.v2 * F2;
    dm.v2 = -m.v1 * F2 + d_phi3_d_v2(m) * F3;

    if (needs_tensor()) {
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          dm.T(i, j) = d_phi3_d_T(i, j, m) * F3;
    } else {
      dm.T.zeros();
    }

    return dm;
  }

  double Functional::bulk_free_energy_density(double density, double diameter) const {
    auto m = Measures::uniform(density, diameter);
    return phi(m);
  }

  double Functional::bulk_excess_chemical_potential(double density, double diameter) const {
    auto m = Measures::uniform(density, diameter);
    auto dm = d_phi(m);
    double d = diameter;
    double R = 0.5 * d;

    double dn3_drho = (std::numbers::pi / 6.0) * d * d * d;
    double dn2_drho = std::numbers::pi * d * d;
    double dn1_drho = R;
    double dn0_drho = 1.0;

    double mu_ex = dm.eta * dn3_drho + dm.n2 * dn2_drho + dm.n1 * dn1_drho + dm.n0 * dn0_drho;

    if (needs_tensor()) {
      double dT_drho = std::numbers::pi * d * d / 3.0;
      for (int j = 0; j < 3; ++j) {
        mu_ex += dm.T(j, j) * dT_drho;
      }
    }

    return mu_ex;
  }

}  // namespace dft_core::physics::fmt
