#include "classicaldft_bits/functional/fmt/functional.h"

#include <numbers>

namespace dft::functional::fmt {

  double FundamentalMeasureTheory::phi(const FundamentalMeasures& m) const {
    double e = m.eta;
    return -m.n0 * f1(e) + (m.n1 * m.n2 - m.contractions.dot_v1_v2) * f2(e) + phi3(m) * f3(e);
  }

  FundamentalMeasures FundamentalMeasureTheory::d_phi(const FundamentalMeasures& m) const {
    FundamentalMeasures dm;
    double e = m.eta;
    double f1_val = f1(e);
    double f2_val = f2(e);
    double f3_val = f3(e);
    double df1_val = d_f1(e);
    double df2_val = d_f2(e);
    double df3_val = d_f3(e);
    double p3 = phi3(m);

    dm.eta = -m.n0 * df1_val + (m.n1 * m.n2 - m.contractions.dot_v1_v2) * df2_val + p3 * df3_val;
    dm.n0 = -f1_val;
    dm.n1 = m.n2 * f2_val;
    dm.n2 = m.n1 * f2_val + d_phi3_d_n2(m) * f3_val;
    dm.v1 = -m.v2 * f2_val;
    dm.v2 = -m.v1 * f2_val + d_phi3_d_v2(m) * f3_val;

    if (needs_tensor()) {
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          dm.T(i, j) = d_phi3_d_T(i, j, m) * f3_val;
    } else {
      dm.T.zeros();
    }

    return dm;
  }

  double FundamentalMeasureTheory::bulk_free_energy_density(double density, double diameter) const {
    auto m = FundamentalMeasures::uniform(density, diameter);
    return phi(m);
  }

  double FundamentalMeasureTheory::bulk_excess_chemical_potential(double density, double diameter) const {
    auto m = FundamentalMeasures::uniform(density, diameter);
    auto dm = d_phi(m);
    double d = diameter;
    double r = 0.5 * d;

    double dn3_drho = (std::numbers::pi / 6.0) * d * d * d;
    double dn2_drho = std::numbers::pi * d * d;
    double dn1_drho = r;
    double dn0_drho = 1.0;

    double mu_ex = dm.eta * dn3_drho + dm.n2 * dn2_drho + dm.n1 * dn1_drho + dm.n0 * dn0_drho;

    if (needs_tensor()) {
      double dt_drho = std::numbers::pi * d * d / 3.0;
      for (int j = 0; j < 3; ++j) {
        mu_ex += dm.T(j, j) * dt_drho;
      }
    }

    return mu_ex;
  }

}  // namespace dft::functional::fmt
