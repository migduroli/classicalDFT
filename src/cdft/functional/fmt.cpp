#include "cdft/functional/fmt.hpp"

#include <numbers>

namespace cdft::functional {

  double fmt_phi(const FMTModel& model, const Measures<>& m) {
    return std::visit([&m](const auto& mod) {
      double e = m.eta;
      double p1 = -m.n0 * mod.ideal_factor(e);
      double p2 = (m.n1 * m.n2 - m.contractions.dot_v1_v2) * mod.pair_factor(e);
      double p3 = mod.mixing_term(m) * mod.triplet_factor(e);
      return p1 + p2 + p3;
    }, model);
  }

  Measures<> fmt_d_phi(const FMTModel& model, const Measures<>& m) {
    return std::visit([&m](const auto& mod) {
      Measures<> dm;
      double e = m.eta;

      // Use autodiff to get f and df/deta for each factor
      auto [f1, df1] = cdft::derivatives_up_to_1(
          [&](cdft::dual x) { return mod.ideal_factor(x); }, e);
      auto [f2, df2] = cdft::derivatives_up_to_1(
          [&](cdft::dual x) { return mod.pair_factor(x); }, e);
      auto [f3, df3] = cdft::derivatives_up_to_1(
          [&](cdft::dual x) { return mod.triplet_factor(x); }, e);

      double p3 = mod.mixing_term(m);

      dm.eta = -m.n0 * df1 + (m.n1 * m.n2 - m.contractions.dot_v1_v2) * df2 + p3 * df3;
      dm.n0 = -f1;
      dm.n1 = m.n2 * f2;
      dm.n2 = m.n1 * f2 + mod.mixing_term_d_n2(m) * f3;
      dm.v1 = -m.v2 * f2;
      dm.v2 = -m.v1 * f2 + mod.mixing_term_d_v2(m) * f3;

      if (mod.needs_tensor()) {
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            dm.T(i, j) = mod.mixing_term_d_T(i, j, m) * f3;
      } else {
        dm.T.zeros();
      }

      return dm;
    }, model);
  }

  double fmt_bulk_free_energy_density(const FMTModel& model, double density, double diameter) {
    auto m = Measures<>::uniform(density, diameter);
    return fmt_phi(model, m);
  }

  double fmt_bulk_excess_chemical_potential(const FMTModel& model, double density, double diameter) {
    auto m = Measures<>::uniform(density, diameter);
    auto dm = fmt_d_phi(model, m);
    double d = diameter;
    double r = 0.5 * d;

    double dn3_drho = (std::numbers::pi / 6.0) * d * d * d;
    double dn2_drho = std::numbers::pi * d * d;
    double dn1_drho = r;
    double dn0_drho = 1.0;

    double mu_ex = dm.eta * dn3_drho + dm.n2 * dn2_drho + dm.n1 * dn1_drho + dm.n0 * dn0_drho;

    if (fmt_needs_tensor(model)) {
      double dt_drho = std::numbers::pi * d * d / 3.0;
      for (int j = 0; j < 3; ++j) {
        mu_ex += dm.T(j, j) * dt_drho;
      }
    }

    return mu_ex;
  }

}  // namespace cdft::functional
