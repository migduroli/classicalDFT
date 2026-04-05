// legacy.hpp — Verbatim translations of Jim's classicalDFT library.
//
// Single consolidated header for all cross-validation tests. Each namespace
// section corresponds to one functional area of Jim's code.
//
// Source files from lutsko/classicalDFT/:
//   include/Potential1.h, src/Potential.cpp         — potentials
//   include/Enskog.h, include/EOS.h                 — thermodynamics
//   include/FMT.h, src/FMT.cpp                      — FMT models
//   src/Interaction.cpp                              — interaction weights
//   src/DFT_Coex.cpp                                 — solver (spinodal/coexistence)
//   include/Crystal_Lattice.h, src/Crystal_Lattice.cpp — crystal lattices

#ifndef DFT_LEGACY_CLASSICALDFT_HPP
#define DFT_LEGACY_CLASSICALDFT_HPP

#include <algorithm>
#include <armadillo>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace legacy {

// ============================================================================
// Potentials (Potential1.h, Potential.cpp)
// ============================================================================

namespace potentials {

  struct PotentialBase {
    double sigma = 1.0;
    double eps = 1.0;
    double rcut = -1.0;
    double rmin = 1.0;
    double shift = 0.0;
    double Vmin = -1.0;
    double r_att_min = 0.0;
    double r0 = 1.0;
    bool bhFlag = false;
    mutable double kT = 1.0;
  };

  struct LJ : PotentialBase {
    static auto vr(double sigma, double eps, double r) -> double {
      double y = sigma / r;
      double y3 = y * y * y;
      double y6 = y3 * y3;
      return 4.0 * eps * (y6 * y6 - y6);
    }

    static auto vr2(double sigma, double eps, double r2) -> double {
      double y2 = sigma * sigma / r2;
      double y6 = y2 * y2 * y2;
      return 4.0 * eps * (y6 * y6 - y6);
    }

    [[nodiscard]] auto getRmin() const -> double { return std::pow(2.0, 1.0 / 6.0) * sigma; }
    [[nodiscard]] auto getHardCore() const -> double { return 0.0; }
  };

  inline auto make_LJ(double sigma, double eps, double rcut) -> LJ {
    LJ lj;
    lj.sigma = sigma;
    lj.eps = eps;
    lj.rcut = rcut;
    lj.shift = (rcut < 0 ? 0.0 : LJ::vr(sigma, eps, rcut));
    lj.rmin = lj.getRmin();
    lj.Vmin = LJ::vr(sigma, eps, lj.rmin) - lj.shift;
    lj.r0 = std::pow(0.5 * std::sqrt(1.0 + lj.shift) + 0.5, -1.0 / 6.0);
    return lj;
  }

  struct tWF : PotentialBase {
    double alpha = 50.0;

    static auto vr(double sigma, double eps, double alpha, double r) -> double {
      if (r < sigma) return 1e50;
      double s = r / sigma;
      double y = 1.0 / (s * s - 1.0);
      double y3 = y * y * y;
      return (4.0 * eps / (alpha * alpha)) * (y3 * y3 - alpha * y3);
    }

    static auto vr2(double sigma, double eps, double alpha, double r2) -> double {
      if (r2 < sigma * sigma) return 1e50;
      double s2 = r2 / (sigma * sigma);
      double y = 1.0 / (s2 - 1.0);
      double y3 = y * y * y;
      return (4.0 * eps / (alpha * alpha)) * (y3 * y3 - alpha * y3);
    }

    [[nodiscard]] auto getRmin() const -> double {
      return sigma * std::sqrt(1.0 + std::pow(2.0 / alpha, 1.0 / 3.0));
    }
    [[nodiscard]] auto getHardCore() const -> double { return sigma; }
  };

  inline auto make_tWF(double sigma, double eps, double rcut, double alpha = 50.0) -> tWF {
    tWF p;
    p.sigma = sigma;
    p.eps = eps;
    p.rcut = rcut;
    p.alpha = alpha;
    p.shift = (rcut <= 0.0 ? 0.0 : tWF::vr(sigma, eps, alpha, rcut));
    p.rmin = p.getRmin();
    p.Vmin = tWF::vr(sigma, eps, alpha, p.rmin) - p.shift;
    p.r0 = std::sqrt(1.0 + std::pow(25.0 * std::sqrt(1.0 + p.shift) + 25.0, -1.0 / 3.0));
    return p;
  }

  struct WHDF : PotentialBase {
    double eps_rescaled = 0.0;

    static auto vr(double eps_rescaled, double sigma, double rcut, double r) -> double {
      double y = sigma / r;
      double z = rcut / r;
      return eps_rescaled * (y * y - 1.0) * (z * z - 1.0) * (z * z - 1.0);
    }

    static auto vr2(double eps_rescaled, double sigma, double rcut, double r2) -> double {
      double y2 = sigma * sigma / r2;
      double z2 = rcut * rcut / r2;
      return eps_rescaled * (y2 - 1.0) * (z2 - 1.0) * (z2 - 1.0);
    }

    [[nodiscard]] auto getRmin() const -> double {
      return rcut * std::pow((1.0 + 2.0 * (rcut / sigma) * (rcut / sigma)) / 3.0, -0.5);
    }
    [[nodiscard]] auto getHardCore() const -> double { return 0.0; }
  };

  inline auto make_WHDF(double sigma, double eps, double rcut) -> WHDF {
    WHDF w;
    w.sigma = sigma;
    w.eps = eps;
    w.rcut = rcut;
    w.eps_rescaled = eps * 2.0 * std::pow(rcut / sigma, 2)
                   * std::pow(2.0 * ((rcut / sigma) * (rcut / sigma) - 1.0) / 3.0, -3.0);
    w.shift = 0.0;
    w.rmin = w.getRmin();
    w.Vmin = WHDF::vr(w.eps_rescaled, sigma, rcut, w.rmin);
    w.r0 = 1.0;
    return w;
  }

  // V(r) = vr(r) - shift
  inline auto V(const LJ& p, double r) -> double { return LJ::vr(p.sigma, p.eps, r) - p.shift; }
  inline auto V(const tWF& p, double r) -> double { return tWF::vr(p.sigma, p.eps, p.alpha, r) - p.shift; }
  inline auto V(const WHDF& p, double r) -> double { return WHDF::vr(p.eps_rescaled, p.sigma, p.rcut, r) - p.shift; }

  // V2(r2) = vr2(r2) - shift
  inline auto V2(const LJ& p, double r2) -> double { return LJ::vr2(p.sigma, p.eps, r2) - p.shift; }
  inline auto V2(const tWF& p, double r2) -> double { return tWF::vr2(p.sigma, p.eps, p.alpha, r2) - p.shift; }
  inline auto V2(const WHDF& p, double r2) -> double { return WHDF::vr2(p.eps_rescaled, p.sigma, p.rcut, r2) - p.shift; }

  // V0(r) = repulsive part
  template <typename P>
  auto V0(const P& p, double r) -> double {
    if (p.bhFlag) return (r < p.r0 ? V(p, r) : 0.0);
    return (r < p.rmin ? V(p, r) - p.Vmin : 0.0);
  }

  // Watt2(r2) = attractive tail
  template <typename P>
  auto Watt2(const P& p, double r2) -> double {
    double ret = 0.0;
    if (r2 < p.rcut * p.rcut) {
      if (p.bhFlag) {
        ret = (r2 < p.r0 * p.r0) ? 0.0 : V2(p, r2);
      } else if (r2 < p.r_att_min * p.r_att_min) {
        ret = 0.0;
      } else if (r2 < p.rmin * p.rmin) {
        ret = p.Vmin;
      } else {
        ret = V2(p, r2);
      }
    }
    return ret;
  }

  template <typename P>
  auto Watt(const P& p, double r) -> double {
    return Watt2(p, r * r);
  }

  // HSD via numerical integration
  template <typename P>
  auto getHSD(const P& p, double kT, int n_points = 1000000) -> double {
    double hc = p.getHardCore();
    double rlimit = p.bhFlag ? p.r0 : p.rmin;
    double h = (rlimit - hc) / n_points;
    double sum = 0.0;
    for (int i = 0; i < n_points; ++i) {
      double r = hc + (i + 0.5) * h;
      double v0 = V0(p, r);
      sum += (1.0 - std::exp(-v0 / kT));
    }
    return hc + sum * h;
  }

  // a_vdw via numerical integration
  template <typename P>
  auto getVDW(const P& p, double kT, int n_points = 1000000) -> double {
    double sum = 0.0;
    if (p.bhFlag) {
      double a = p.r0;
      double b = p.rcut;
      double h = (b - a) / n_points;
      for (int i = 0; i < n_points; ++i) {
        double r = a + (i + 0.5) * h;
        sum += r * r * Watt(p, r);
      }
      sum *= h;
    } else {
      double a1 = p.r_att_min;
      double b1 = p.rmin;
      if (b1 > a1) {
        double h1 = (b1 - a1) / n_points;
        double sum1 = 0.0;
        for (int i = 0; i < n_points; ++i) {
          double r = a1 + (i + 0.5) * h1;
          sum1 += r * r * Watt(p, r);
        }
        sum += sum1 * h1;
      }
      double a2 = p.rmin;
      double b2 = p.rcut;
      if (b2 > a2) {
        double h2 = (b2 - a2) / n_points;
        double sum2 = 0.0;
        for (int i = 0; i < n_points; ++i) {
          double r = a2 + (i + 0.5) * h2;
          sum2 += r * r * Watt(p, r);
        }
        sum += sum2 * h2;
      }
    }
    return (2.0 * std::numbers::pi / kT) * sum;
  }

}  // namespace potentials

// ============================================================================
// Thermodynamics (Enskog.h, EOS.h)
// ============================================================================

namespace thermodynamics {

  struct Enskog {
    double n;
    double e;
    double c;

    explicit Enskog(double density) : n(density), e(std::numbers::pi * density / 6.0) {
      c = (1.0 - e / 2.0) / (1.0 - 3.0 * e + 3.0 * e * e - e * e * e);
    }

    [[nodiscard]] auto exFreeEnergyPYC() const -> double {
      return -std::log(1.0 - e) + 1.5 * e * (2.0 - e) / (1.0 - 2.0 * e + e * e);
    }
    [[nodiscard]] auto exFreeEnergyPYV() const -> double {
      return 2.0 * std::log(1.0 - e) + 6.0 * e / (1.0 - e);
    }
    [[nodiscard]] auto exFreeEnergyCS() const -> double {
      return e * (4.0 - 3.0 * e) / (1.0 - 2.0 * e + e * e);
    }
    [[nodiscard]] auto dexFreeEnergyPYCdRho() const -> double {
      return (std::numbers::pi / 6.0) * (4.0 - 2.0 * e + e * e) /
             (1.0 - 3.0 * e + 3.0 * e * e - e * e * e);
    }
    [[nodiscard]] auto d2exFreeEnergyPYCdRho2() const -> double {
      return (std::numbers::pi / 6.0) * (std::numbers::pi / 6.0) * (10.0 - 2.0 * e + e * e) /
             (1.0 - 4.0 * e + 6.0 * e * e - 4.0 * e * e * e + e * e * e * e);
    }
    [[nodiscard]] auto d3exFreeEnergyPYCdRho3() const -> double {
      return (std::numbers::pi / 6.0) * (std::numbers::pi / 6.0) * (std::numbers::pi / 6.0) *
             (38.0 - 4.0 * e + 2.0 * e * e) * std::pow(1.0 - e, -5);
    }
    [[nodiscard]] auto dexFreeEnergyCSdRho() const -> double {
      return (std::numbers::pi / 6.0) * (4.0 - 2.0 * e) /
             (1.0 - 3.0 * e + 3.0 * e * e - e * e * e);
    }
    [[nodiscard]] auto d2exFreeEnergyCSdRho2() const -> double {
      return (std::numbers::pi / 6.0) * (std::numbers::pi / 6.0) * (10.0 - 4.0 * e) /
             (1.0 - 4.0 * e + 6.0 * e * e - 4.0 * e * e * e + e * e * e * e);
    }
    [[nodiscard]] auto d3exFreeEnergyCSdRho3() const -> double {
      return (std::numbers::pi / 6.0) * (std::numbers::pi / 6.0) * (std::numbers::pi / 6.0) *
             12.0 * (3.0 - e) * std::pow(1.0 - e, -5);
    }
    [[nodiscard]] auto pressurePYC() const -> double {
      return (1.0 + e + e * e) / (1.0 - 3.0 * e + 3.0 * e * e - e * e * e);
    }
    [[nodiscard]] auto pressureCS() const -> double {
      return (1.0 + e + e * e - e * e * e) / (1.0 - 3.0 * e + 3.0 * e * e - e * e * e);
    }
    [[nodiscard]] auto freeEnergyPYC() const -> double {
      return std::log(n) - 1.0 + exFreeEnergyPYC();
    }
    [[nodiscard]] auto chemPotentialPYC() const -> double {
      return freeEnergyPYC() + pressurePYC();
    }
  };

  struct LJ_JZG {
    double kT;
    double da;
    static constexpr double gamma = 3.0;
    static constexpr double x1 = 0.8623085097507421;
    static constexpr double x2 = 2.976218765822098;
    static constexpr double x3 = -8.402230115796038;
    static constexpr double x4 = 0.1054136629203555;
    static constexpr double x5 = -0.8564583828174598;
    static constexpr double x6 = 1.582759470107601;
    static constexpr double x7 = 0.7639421948305453;
    static constexpr double x8 = 1.753173414312048;
    static constexpr double x9 = 2.798291772190376e3;
    static constexpr double x10 = -4.8394220260857657e-2;
    static constexpr double x11 = 0.9963265197721935;
    static constexpr double x12 = -3.698000291272493e1;
    static constexpr double x13 = 2.084012299434647e1;
    static constexpr double x14 = 8.305402124717285e1;
    static constexpr double x15 = -9.574799715203068e2;
    static constexpr double x16 = -1.477746229234994e2;
    static constexpr double x17 = 6.398607852471505e1;
    static constexpr double x18 = 1.603993673294834e1;
    static constexpr double x19 = 6.805916615864377e1;
    static constexpr double x20 = -2.791293578795945e3;
    static constexpr double x21 = -6.245128304568454;
    static constexpr double x22 = -8.116836104958410e3;
    static constexpr double x23 = 1.488735559561229e1;
    static constexpr double x24 = -1.059346754655084e4;
    static constexpr double x25 = -1.131607632802822e2;
    static constexpr double x26 = -8.867771540418822e3;
    static constexpr double x27 = -3.986982844450543e1;
    static constexpr double x28 = -4.689270299917261e3;
    static constexpr double x29 = 2.593535277438717e2;
    static constexpr double x30 = -2.694523589434903e3;
    static constexpr double x31 = -7.218487631550215e2;
    static constexpr double x32 = 1.721802063863269e2;

    [[nodiscard]] auto a(int i) const -> double {
      if (i == 1) return x1 * kT + x2 * std::sqrt(kT) + x3 + (x4 / kT) + x5 / (kT * kT);
      if (i == 2) return x6 * kT + x7 + (x8 / kT) + x9 / (kT * kT);
      if (i == 3) return x10 * kT + x11 + (x12 / kT);
      if (i == 4) return x13;
      if (i == 5) return (x14 / kT) + x15 / (kT * kT);
      if (i == 6) return x16 / kT;
      if (i == 7) return (x17 / kT) + x18 / (kT * kT);
      if (i == 8) return x19 / (kT * kT);
      throw std::runtime_error("LJ_JZG: invalid a index");
    }

    [[nodiscard]] auto b(int i) const -> double {
      double T1 = 1.0 / kT;
      double T2 = T1 * T1;
      double T3 = T1 * T2;
      double T4 = T2 * T2;
      if (i == 1) return x20 * T2 + x21 * T3;
      if (i == 2) return x22 * T2 + x23 * T4;
      if (i == 3) return x24 * T2 + x25 * T3;
      if (i == 4) return x26 * T2 + x27 * T4;
      if (i == 5) return x28 * T2 + x29 * T3;
      if (i == 6) return x30 * T2 + x31 * T3 + x32 * T4;
      throw std::runtime_error("LJ_JZG: invalid b index");
    }

    [[nodiscard]] auto G(double d, int i) const -> double {
      double F = std::exp(-gamma * d * d);
      double G1 = (1.0 - F) / (2.0 * gamma);
      if (i == 1) return G1;
      double G2 = -(F * std::pow(d, 2) - 2.0 * G1) / (2.0 * gamma);
      if (i == 2) return G2;
      double G3 = -(F * std::pow(d, 4) - 4.0 * G2) / (2.0 * gamma);
      if (i == 3) return G3;
      double G4 = -(F * std::pow(d, 6) - 6.0 * G3) / (2.0 * gamma);
      if (i == 4) return G4;
      double G5 = -(F * std::pow(d, 8) - 8.0 * G4) / (2.0 * gamma);
      if (i == 5) return G5;
      double G6 = -(F * std::pow(d, 10) - 10.0 * G5) / (2.0 * gamma);
      if (i == 6) return G6;
      throw std::runtime_error("LJ_JZG: invalid G index");
    }

    [[nodiscard]] auto phix(double density) const -> double {
      double f = 0.0;
      for (int i = 1; i <= 8; ++i) f += a(i) * std::pow(density, i) / i;
      for (int i = 1; i <= 6; ++i) f += b(i) * G(density, i);
      f += da * density;
      return f / kT;
    }

    [[nodiscard]] auto phi1x(double density) const -> double {
      double F = std::exp(-gamma * density * density);
      double f = 0.0;
      for (int i = 1; i <= 8; ++i) f += a(i) * std::pow(density, i - 1);
      for (int i = 1; i <= 6; ++i) f += b(i) * F * std::pow(density, 2 * i - 1);
      f += da;
      return f / kT;
    }

    [[nodiscard]] auto phi2x(double density) const -> double {
      double F = std::exp(-gamma * density * density);
      double f = 0.0;
      for (int i = 2; i <= 8; ++i) f += a(i) * (i - 1) * std::pow(density, i - 2);
      for (int i = 1; i <= 6; ++i) {
        f += b(i) * F * ((2 * i - 1) * std::pow(density, 2 * i - 2) -
                          2.0 * gamma * std::pow(density, 2 * i));
      }
      return f / kT;
    }

    [[nodiscard]] auto phi3x(double density) const -> double {
      double F = std::exp(-gamma * density * density);
      double f = 0.0;
      for (int i = 3; i <= 8; ++i) f += a(i) * (i - 1) * (i - 2) * std::pow(density, i - 3);
      for (int i = 1; i <= 6; ++i) {
        f += b(i) * (-2.0 * gamma * density) * F *
             ((2 * i - 1) * std::pow(density, 2 * i - 2) -
              2.0 * gamma * std::pow(density, 2 * i));
      }
      for (int i = 2; i <= 6; ++i) {
        f += b(i) * F * (2 * i - 1) * (2 * i - 2) * std::pow(density, 2 * i - 3);
      }
      for (int i = 1; i <= 6; ++i) {
        f += b(i) * F * (-2.0 * 2.0 * i * gamma * std::pow(density, 2 * i - 1));
      }
      return f / kT;
    }
  };

  inline auto make_LJ_JZG(double kT, double rc = -1.0) -> LJ_JZG {
    double da = 0.0;
    if (rc > 0.0) {
      da = -(32.0 / 9.0) * std::numbers::pi * (std::pow(rc, -9.0) - 1.5 * std::pow(rc, -3.0));
    }
    return LJ_JZG{.kT = kT, .da = da};
  }

  struct LJ_Mecke {
    double kT;
    double da;

    static constexpr double rhoc = 0.3107;
    static constexpr double kTc = 1.328;
    static constexpr double c[32] = {
        0.33619760720e-05,  -0.14707220591e+01, -0.11972121043e+00, -0.11350363539e-04,
        -0.26778688896e-04,  0.12755936511e-05,  0.40088615477e-02,  0.52305580273e-05,
        -0.10214454556e-07, -0.14526799362e-01,  0.64975356409e-01, -0.60304755494e-01,
        -0.14925537332e+00, -0.31664355868e-03,  0.28312781935e-01,  0.13039603845e-03,
         0.10121435381e-01, -0.15425936014e-04, -0.61568007279e-01,  0.76001994423e-02,
        -0.18906040708e+00,  0.33141311846e+00, -0.25229604842e+00,  0.13145401812e+00,
        -0.48672350917e-01,  0.14756043863e-02, -0.85996667747e-02,  0.33880247915e-01,
         0.69427495094e-02, -0.22271531045e-07, -0.22656880018e-03,  0.24056013779e-02,
    };
    static constexpr double m[32] = {
        -2, -1, -1, -1, -0.5, -0.5, 0.5, 0.5, 1, -5, -4, -2, -2, -2, -1, -1,
        0, 0, -5, -4, -3, -2, -2, -2, -1, -10, -6, -4, 0, -24, -10, -2,
    };
    static constexpr int n[32] = {
        9, 1, 2, 9, 8, 10, 1, 7, 10, 1, 1, 1, 2, 8, 1, 10,
        4, 9, 2, 5, 1, 2, 3, 4, 2, 3, 4, 2, 2, 5, 2, 10,
    };
    static constexpr int p[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    };
    static constexpr int q[32] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
    };

    [[nodiscard]] auto phix(double density) const -> double {
      double rho_r = density / rhoc;
      double t_r = kT / kTc;
      double s = 0.1617 * rho_r / (0.689 + 0.311 * std::pow(t_r, 0.3674));
      double fhs = (4.0 * s - 3.0 * s * s) / ((1.0 - s) * (1.0 - s));
      double fa = 0.0;
      for (int i = 0; i < 32; ++i) {
        fa += c[i] * std::pow(t_r, m[i]) * std::pow(rho_r, n[i]) *
              std::exp(p[i] * std::pow(rho_r, q[i]));
      }
      fa += da * density / kT;
      return fhs + fa;
    }

    [[nodiscard]] auto phi1x(double density) const -> double {
      double rho_r = density / rhoc;
      double t_r = kT / kTc;
      double s1 = 0.1617 * (1.0 / rhoc) / (0.689 + 0.311 * std::pow(t_r, 0.3674));
      double s = s1 * density;
      double fhs = s1 * (4.0 - 2.0 * s) / ((1.0 - s) * (1.0 - s) * (1.0 - s));
      double fa = 0.0;
      for (int i = 0; i < 32; ++i) {
        fa += c[i] * std::pow(t_r, m[i]) * (n[i] / rhoc) *
              std::pow(rho_r, n[i] - 1) *
              std::exp(p[i] * std::pow(rho_r, q[i]));
        fa += c[i] * std::pow(t_r, m[i]) * std::pow(rho_r, n[i]) *
              std::exp(p[i] * std::pow(rho_r, q[i])) *
              p[i] * q[i] * std::pow(rho_r, q[i] - 1) / rhoc;
      }
      fa += da / kT;
      return fhs + fa;
    }
  };

  inline auto make_LJ_Mecke(double kT, double rc = -1.0) -> LJ_Mecke {
    double da = 0.0;
    if (rc > 0.0) {
      da = -(32.0 / 9.0) * std::numbers::pi * (std::pow(rc, -9.0) - 1.5 * std::pow(rc, -3.0));
    }
    return LJ_Mecke{.kT = kT, .da = da};
  }

}  // namespace thermodynamics

// ============================================================================
// FMT models (FMT.h, FMT.cpp)
// ============================================================================

namespace fmt {

  struct FundamentalMeasures {
    double eta = 0.0;
    double s0 = 0.0;
    double s1 = 0.0;
    double s2 = 0.0;
    double v1[3] = {};
    double v2[3] = {};
    double T[3][3] = {};

    double v1_v2 = 0.0;
    double v2_v2 = 0.0;
    double vTv = 0.0;
    double T2 = 0.0;
    double T3 = 0.0;
    double vT[3] = {};
    double Tv[3] = {};
    double TT[3][3] = {};

    void calculate_derived_quantities() {
      v1_v2 = 0;
      v2_v2 = 0;
      vTv = 0;
      for (int i = 0; i < 3; ++i) {
        v1_v2 += v1[i] * v2[i];
        v2_v2 += v2[i] * v2[i];
        vT[i] = 0;
        Tv[i] = 0;
        for (int j = 0; j < 3; ++j) {
          vT[i] += v2[j] * T[j][i];
          Tv[i] += T[i][j] * v2[j];
        }
        vTv += v2[i] * Tv[i];
      }
      T2 = 0;
      T3 = 0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          TT[i][j] = 0;
          for (int k = 0; k < 3; ++k) TT[i][j] += T[i][k] * T[k][j];
          T2 += T[i][j] * T[j][i];
        }
      }
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          T3 += T[i][j] * TT[j][i];
    }
  };

  namespace Rosenfeld_model {
    inline auto f1(double eta) -> double { return std::log(1.0 - eta); }
    inline auto f2(double eta) -> double { return 1.0 / (1.0 - eta); }
    inline auto f3(double eta) -> double { double e = 1.0 - eta; return 1.0 / (e * e); }
    inline auto fex(double eta) -> double {
      return -std::log(1.0 - eta) + 1.5 * eta * (2.0 - eta) / ((1.0 - eta) * (1.0 - eta));
    }
    inline auto Phi3(const FundamentalMeasures& fm) -> double {
      return (1.0 / (24.0 * std::numbers::pi)) * (fm.s2 * fm.s2 * fm.s2 - 3.0 * fm.s2 * fm.v2_v2);
    }
    inline auto dPhi3_dS2(const FundamentalMeasures& fm) -> double {
      return (1.0 / (24.0 * std::numbers::pi)) * (3.0 * fm.s2 * fm.s2 - 3.0 * fm.v2_v2);
    }
    inline auto dPhi3_dV2(int k, const FundamentalMeasures& fm) -> double {
      return (1.0 / (24.0 * std::numbers::pi)) * (-6.0 * fm.s2 * fm.v2[k]);
    }
  }

  namespace RSLT_model {
    inline auto f1(double eta) -> double { return std::log(1.0 - eta); }
    inline auto f2(double eta) -> double { return 1.0 / (1.0 - eta); }
    inline auto f3(double x) -> double {
      if (x < 1e-12) return 1.5 + (8.0 / 3.0) * x + 3.75 * x * x + 4.8 * x * x * x;
      return (1.0 / (x * (1.0 - x) * (1.0 - x))) + (std::log(1.0 - x) / (x * x));
    }
    inline auto fex(double eta) -> double {
      return eta * (4.0 - 3.0 * eta) / ((1.0 - eta) * (1.0 - eta));
    }
    inline auto Phi3(const FundamentalMeasures& fm) -> double {
      if (fm.s2 < 1e-20) return 0.0;
      double psi = fm.v2_v2 / (fm.s2 * fm.s2);
      return (1.0 / (36.0 * std::numbers::pi)) * fm.s2 * fm.s2 * fm.s2 *
             (1.0 - psi) * (1.0 - psi) * (1.0 - psi);
    }
    inline auto dPhi3_dS2(const FundamentalMeasures& fm) -> double {
      if (fm.s2 < 1e-20) return 0.0;
      double psi = fm.v2_v2 / (fm.s2 * fm.s2);
      return (1.0 / (12.0 * std::numbers::pi)) * fm.s2 * fm.s2 *
             (1.0 + psi) * (1.0 - psi) * (1.0 - psi);
    }
    inline auto dPhi3_dV2(int k, const FundamentalMeasures& fm) -> double {
      if (fm.s2 < 1e-20) return 0.0;
      double psi = fm.v2_v2 / (fm.s2 * fm.s2);
      return -(1.0 / (6.0 * std::numbers::pi)) * fm.s2 * fm.v2[k] *
             (1.0 - psi) * (1.0 - psi);
    }
  }

  namespace esFMT_model {
    inline auto f1(double eta) -> double { return std::log(1.0 - eta); }
    inline auto f2(double eta) -> double { return 1.0 / (1.0 - eta); }
    inline auto f3(double eta) -> double { double e = 1.0 - eta; return 1.0 / (e * e); }
    inline auto Phi3(double A, double B, const FundamentalMeasures& fm) -> double {
      return (A / (24.0 * std::numbers::pi)) * (fm.s2 * fm.s2 * fm.s2 - 3.0 * fm.s2 * fm.v2_v2 + 3.0 * fm.vTv - fm.T3) +
             (B / (24.0 * std::numbers::pi)) * (fm.s2 * fm.s2 * fm.s2 - 3.0 * fm.s2 * fm.T2 + 2.0 * fm.T3);
    }
    inline auto dPhi3_dS2(double A, double B, const FundamentalMeasures& fm) -> double {
      return (A / (24.0 * std::numbers::pi)) * (3.0 * fm.s2 * fm.s2 - 3.0 * fm.v2_v2) +
             (B / (24.0 * std::numbers::pi)) * (3.0 * fm.s2 * fm.s2 - 3.0 * fm.T2);
    }
    inline auto dPhi3_dV2(double A, int k, const FundamentalMeasures& fm) -> double {
      return (A / (24.0 * std::numbers::pi)) * (-6.0 * fm.s2 * fm.v2[k] + 3.0 * fm.vT[k] + 3.0 * fm.Tv[k]);
    }
    inline auto dPhi3_dT(double A, double B, int j, int k, const FundamentalMeasures& fm) -> double {
      return (A / (8.0 * std::numbers::pi)) * (fm.v2[j] * fm.v2[k] - fm.TT[k][j]) +
             (B / (4.0 * std::numbers::pi)) * (-fm.s2 * fm.T[k][j] + fm.TT[k][j]);
    }
    inline auto fex(double eta, double A = 1.0, double B = 0.0) -> double {
      return -std::log(1.0 - eta)
           + 1.5 * eta * (2.0 - eta) / ((1.0 - eta) * (1.0 - eta))
           + (1.0 / 6.0) * (8.0 * A + 2.0 * B - 9.0)
                 * (eta * eta / ((1.0 - eta) * (1.0 - eta)));
    }
  }

  // Full free energy density: Phi = -s0*log(1-eta) + (s1*s2 - v1·v2)/(1-eta) + Phi3/(1-eta)²
  inline auto phi(const FundamentalMeasures& fm, double A = 1.0, double B = 0.0) -> double {
    if (fm.eta < 1e-30) return 0.0;
    double om = 1.0 / (1.0 - fm.eta);
    return -fm.s0 * std::log(1.0 - fm.eta)
         + (fm.s1 * fm.s2 - fm.v1_v2) * om
         + esFMT_model::Phi3(A, B, fm) * om * om;
  }

  namespace WhiteBearI_model {
    inline auto f1(double eta) -> double { return std::log(1.0 - eta); }
    inline auto f2(double eta) -> double { return 1.0 / (1.0 - eta); }
    inline auto f3(double x) -> double {
      if (x < 1e-12) return 1.5 + (8.0 / 3.0) * x + 3.75 * x * x + 4.8 * x * x * x;
      return (1.0 / (x * (1.0 - x) * (1.0 - x))) + (std::log(1.0 - x) / (x * x));
    }
    inline auto fex(double eta) -> double {
      return eta * (4.0 - 3.0 * eta) / ((1.0 - eta) * (1.0 - eta));
    }
  }

  namespace WhiteBearII_model {
    inline auto f1(double eta) -> double { return std::log(1.0 - eta); }
    inline auto f2(double x) -> double {
      if (x < 1e-12) return 1.0 + x * (1.0 + x * ((10.0 / 9.0) + (7.0 / 6.0) * x));
      return (1.0 / 3.0) + (4.0 / 3.0) * (1.0 / (1.0 - x)) + (2.0 / (3.0 * x)) * std::log(1.0 - x);
    }
    inline auto f3(double x) -> double {
      if (x < 1e-12) return 1.5 + (7.0 / 3.0) * x + 3.25 * x * x + 4.2 * x * x * x;
      return -((1.0 - 3.0 * x + x * x) / (x * (1.0 - x) * (1.0 - x))) - (1.0 / (x * x)) * std::log(1.0 - x);
    }
    inline auto fex(double eta) -> double {
      return eta * (4.0 - 3.0 * eta) / ((1.0 - eta) * (1.0 - eta));
    }
  }

}  // namespace fmt

// ============================================================================
// Interaction weights (Interaction.cpp)
// ============================================================================

namespace interactions {

  // Watt2 for LJ (WCA)
  inline auto Watt2_LJ(double sigma, double eps, double rcut, double r2) -> double {
    double y2 = sigma * sigma / r2;
    double y6 = y2 * y2 * y2;
    double vr2 = 4.0 * eps * (y6 * y6 - y6);
    double shift = [&] {
      double y = sigma / rcut;
      double y3 = y * y * y;
      double y6s = y3 * y3;
      return 4.0 * eps * (y6s * y6s - y6s);
    }();
    double rmin = std::pow(2.0, 1.0 / 6.0) * sigma;
    double Vmin = [&] {
      double y = sigma / rmin;
      double y3 = y * y * y;
      double y6s = y3 * y3;
      return 4.0 * eps * (y6s * y6s - y6s) - shift;
    }();

    if (r2 >= rcut * rcut) return 0.0;
    if (r2 < rmin * rmin) return Vmin;
    return vr2 - shift;
  }

  // QF weight at a single displacement
  inline auto generate_weight_QF(
      double sigma, double eps, double rcut,
      int Sx, int Sy, int Sz, double dx
  ) -> double {
    static constexpr double vv[] = {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    static constexpr double pt[] = {-0.5, 0.0, 0.5};
    double sum = 0.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) {
          double r2 = (Sx + pt[i]) * (Sx + pt[i]) * dx * dx
                    + (Sy + pt[j]) * (Sy + pt[j]) * dx * dx
                    + (Sz + pt[k]) * (Sz + pt[k]) * dx * dx;
          sum += vv[i] * vv[j] * vv[k] * Watt2_LJ(sigma, eps, rcut, r2);
        }
    return sum;
  }

  // a_vdw on a grid via QF weights
  inline auto compute_a_vdw_QF(
      double sigma, double eps, double rcut, double kT, double dx
  ) -> double {
    int Nx_lim = 1 + static_cast<int>(rcut / dx);
    double sum = 0.0;
    double global_factor = std::pow(dx, 6);

    for (int ix = -(Nx_lim + 1); ix <= Nx_lim + 1; ++ix)
      for (int iy = -(Nx_lim + 1); iy <= Nx_lim + 1; ++iy)
        for (int iz = -(Nx_lim + 1); iz <= Nx_lim + 1; ++iz) {
          int nx = std::abs(ix);
          int ny = std::abs(iy);
          int nz = std::abs(iz);
          if (ny > nx) std::swap(nx, ny);
          if (nz > nx) std::swap(nx, nz);
          if (nz > ny) std::swap(ny, nz);
          sum += global_factor * generate_weight_QF(sigma, eps, rcut, nx, ny, nz, dx);
        }

    return sum / (dx * dx * dx) / kT;
  }

  struct WeightResult {
    double a_vdw;
    double a_vdw_over_kT;
    int n_nonzero;
  };

  inline auto compute_a_vdw_QF_detailed(
      double sigma, double eps, double rcut, double kT, double dx
  ) -> WeightResult {
    int Nx_lim = 1 + static_cast<int>(rcut / dx);
    double a_vdw = 0.0;
    int n_nonzero = 0;
    double global_factor = std::pow(dx, 6);

    for (int ix = -(Nx_lim + 1); ix <= Nx_lim + 1; ++ix)
      for (int iy = -(Nx_lim + 1); iy <= Nx_lim + 1; ++iy)
        for (int iz = -(Nx_lim + 1); iz <= Nx_lim + 1; ++iz) {
          int nx = std::abs(ix);
          int ny = std::abs(iy);
          int nz = std::abs(iz);
          if (ny > nx) std::swap(nx, ny);
          if (nz > nx) std::swap(nx, nz);
          if (nz > ny) std::swap(ny, nz);
          double w_raw = global_factor * generate_weight_QF(sigma, eps, rcut, nx, ny, nz, dx);
          a_vdw += w_raw;
          if (std::abs(w_raw) > 1e-30) n_nonzero++;
        }

    a_vdw /= (dx * dx * dx);
    return {a_vdw, a_vdw / kT, n_nonzero};
  }

}  // namespace interactions

// ============================================================================
// Solver algorithms (DFT_Coex.cpp)
// ============================================================================

namespace solver {

  // EOS callbacks: the test file supplies these from our library.
  // legacy.hpp itself never touches dft::*.
  using PressureFn = std::function<double(double)>;
  using ChemPotFn = std::function<double(double)>;

  struct EOS {
    PressureFn pressure;
    ChemPotFn chemical_potential;
  };

  struct Spinodal {
    double xs1;
    double xs2;
  };

  inline auto findSpinodal(const EOS& eos, double xmax, double dx, double tol = 1e-8) -> Spinodal {
    double x = 2 * dx;
    double p0 = eos.pressure(dx);
    double p = eos.pressure(2 * dx);
    double dp = p - p0;
    if (dp < 0) throw std::runtime_error("Could not get started in findSpinodal");

    while (dp > 0 && x < xmax - dx) {
      dp = -p;
      x += dx;
      p = eos.pressure(x);
      dp += p;
    }
    if (x >= xmax - dx) throw std::runtime_error("Xmax exceeded in findSpinodal (max)");

    double a = x - 2 * dx;
    double b = x;
    double r = (3.0 - std::sqrt(5.0)) / 2.0;
    double u = a + r * (b - a);
    double v = b - r * (b - a);
    double fu = eos.pressure(u);
    double fv = eos.pressure(v);

    do {
      if (fu > fv) { b = v; v = u; fv = fu; u = a + r * (b - a); fu = eos.pressure(u); }
      else { a = u; u = v; fu = fv; v = b - r * (b - a); fv = eos.pressure(v); }
    } while (b - a > tol);
    double xs1 = (a + b) / 2;

    x = xs1;
    p = eos.pressure(x);
    do { dp = -p; x += dx; p = eos.pressure(x); dp += p; } while (dp < 0 && x < xmax - dx);
    if (x >= xmax - dx) throw std::runtime_error("Xmax exceeded in findSpinodal (min)");

    a = x - 2 * dx;
    b = x;
    u = a + r * (b - a);
    v = b - r * (b - a);
    fu = eos.pressure(u);
    fv = eos.pressure(v);

    do {
      if (fu < fv) { b = v; v = u; fv = fu; u = a + r * (b - a); fu = eos.pressure(u); }
      else { a = u; u = v; fu = fv; v = b - r * (b - a); fv = eos.pressure(v); }
    } while (b - a > tol);
    double xs2 = (a + b) / 2;

    return {xs1, xs2};
  }

  inline auto find_density_from_mu(const EOS& eos, double mu, double xmin, double xmax, double tol = 1e-8) -> double {
    double mu1 = eos.chemical_potential(xmin);
    double mu2 = eos.chemical_potential(xmax);
    if (mu1 > mu2) { std::swap(xmin, xmax); std::swap(mu1, mu2); }
    if (mu2 < mu || mu1 > mu) throw std::runtime_error("find_density_from_mu: target mu out of range");
    do {
      double x = (xmin + xmax) / 2;
      double mux = eos.chemical_potential(x);
      if (mux > mu) { xmax = x; mu2 = mux; } else { xmin = x; mu1 = mux; }
    } while (mu2 - mu1 > tol);
    return (xmin + xmax) / 2;
  }

  struct Coexistence {
    double x1;
    double x2;
  };

  inline auto findCoex(const EOS& eos, double xmax, double dx, double tol = 1e-8) -> Coexistence {
    auto [xs1, xs2] = findSpinodal(eos, xmax, dx, tol);
    double xvap = xs1;
    double xliq = find_density_from_mu(eos, eos.chemical_potential(xvap), xs2, xmax - dx, tol);
    double dp1 = eos.pressure(xvap) - eos.pressure(xliq);
    double dp2;
    do {
      dp2 = dp1;
      xvap /= 1.1;
      xliq = find_density_from_mu(eos, eos.chemical_potential(xvap), xs2, xmax - dx, tol);
      dp1 = eos.pressure(xvap) - eos.pressure(xliq);
    } while (xvap > 1e-40 && ((dp1 < 0) == (dp2 < 0)));
    if (xvap < 1e-40) throw std::runtime_error("findCoex bracket failed");

    double y1 = xvap;
    double y2 = xvap * 1.1;
    do {
      double y = (y1 + y2) / 2;
      xliq = find_density_from_mu(eos, eos.chemical_potential(y), xs2, xmax - dx, tol);
      double dp = eos.pressure(y) - eos.pressure(xliq);
      if ((dp < 0) == (dp1 < 0)) { y1 = y; dp1 = dp; } else { y2 = y; dp2 = dp; }
    } while (std::abs(y2 - y1) > xvap * tol);

    double rv = (y1 + y2) / 2;
    double rl = find_density_from_mu(eos, eos.chemical_potential(rv), xs2, xmax - dx, tol);
    return {rv, rl};
  }

}  // namespace solver

// ============================================================================
// Crystal lattices (Crystal_Lattice.h, Crystal_Lattice.cpp)
// ============================================================================

namespace crystal {

  using Atom = std::array<double, 3>;

  struct UnitCell {
    std::vector<Atom> atoms;
    std::array<double, 3> L;
  };

  inline auto regular_bcc() -> std::vector<Atom> {
    return {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
  }

  inline auto regular_fcc() -> std::vector<Atom> {
    return {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}};
  }

  inline auto hcp_001() -> std::vector<Atom> {
    return {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.0, 2.0 / 6.0, 0.5}, {0.5, 5.0 / 6.0, 0.5}};
  }

  inline void scale_atoms(std::vector<Atom>& atoms, const std::array<double, 3>& dims) {
    for (auto& a : atoms) { a[0] *= dims[0]; a[1] *= dims[1]; a[2] *= dims[2]; }
  }

  inline auto make_unit_cell(const std::string& lattice, const std::string& orient) -> UnitCell {
    UnitCell uc;

    if (lattice == "BCC") {
      double a = 2.0 / std::sqrt(3.0);
      if (orient == "001" || orient == "010" || orient == "100") {
        uc.atoms = regular_bcc();
        uc.L = {a, a, a};
      } else if (orient == "110" || orient == "101" || orient == "011") {
        uc.atoms = regular_fcc();
        uc.L = {a, a, a};
        if (orient == "110") { uc.L[0] *= std::sqrt(2.0); uc.L[1] *= std::sqrt(2.0); }
        if (orient == "101") { uc.L[0] *= std::sqrt(2.0); uc.L[2] *= std::sqrt(2.0); }
        if (orient == "011") { uc.L[1] *= std::sqrt(2.0); uc.L[2] *= std::sqrt(2.0); }
      } else if (orient == "111") {
        uc.atoms = {
            {1.0/6, 0.5, 0.0}, {0.0, 0.0, 1.0/3}, {1.0/6, 0.5, 0.5}, {0.0, 0.0, 5.0/6},
            {3.0/6, 0.5, 1.0/3}, {1.0/3, 0.0, 2.0/3}, {3.0/6, 0.5, 5.0/6}, {1.0/3, 0.0, 1.0/6},
            {5.0/6, 0.5, 2.0/3}, {2.0/3, 0.0, 0.0}, {5.0/6, 0.5, 1.0/6}, {2.0/3, 0.0, 0.5},
        };
        uc.L = {2.0 * std::sqrt(2.0), 2.0 * std::sqrt(2.0 / 3.0), 2.0};
      }
    } else if (lattice == "FCC") {
      double a = std::sqrt(2.0);
      if (orient == "001" || orient == "010" || orient == "100") {
        uc.atoms = regular_fcc();
        uc.L = {a, a, a};
      } else if (orient == "110" || orient == "101" || orient == "011") {
        uc.atoms = regular_bcc();
        uc.L = {1.0, 1.0, 1.0};
        if (orient == "110") uc.L[2] *= std::sqrt(2.0);
        if (orient == "101") uc.L[1] *= std::sqrt(2.0);
        if (orient == "011") uc.L[0] *= std::sqrt(2.0);
      } else if (orient == "111") {
        uc.atoms = {
            {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 1.0/6, 1.0/3},
            {0.0, 4.0/6, 1.0/3}, {0.0, 2.0/6, 2.0/3}, {0.5, 5.0/6, 2.0/3},
        };
        uc.L = {1.0, std::sqrt(3.0), std::sqrt(6.0)};
      }
    } else if (lattice == "HCP") {
      uc.atoms = hcp_001();
      uc.L = {1.0, std::sqrt(3.0), std::sqrt(8.0 / 3.0)};
      scale_atoms(uc.atoms, uc.L);
      if (orient == "010") {
        for (auto& atom : uc.atoms) { double t = atom[2]; atom[2] = atom[1]; atom[1] = -t; }
        std::array<double, 3> nL = {uc.L[0], uc.L[2], uc.L[1]};
        for (auto& atom : uc.atoms)
          for (int d = 0; d < 3; ++d) { atom[d] = std::fmod(atom[d], nL[d]); if (atom[d] < 0) atom[d] += nL[d]; }
        uc.L = nL;
      } else if (orient == "100") {
        for (auto& atom : uc.atoms) { double t = atom[2]; atom[2] = atom[0]; atom[0] = -t; }
        std::array<double, 3> nL = {uc.L[2], uc.L[1], uc.L[0]};
        for (auto& atom : uc.atoms)
          for (int d = 0; d < 3; ++d) { atom[d] = std::fmod(atom[d], nL[d]); if (atom[d] < 0) atom[d] += nL[d]; }
        uc.L = nL;
      }
      return uc;  // HCP already scaled
    }

    scale_atoms(uc.atoms, uc.L);
    return uc;
  }

  struct Lattice {
    std::vector<Atom> atoms;
    std::array<double, 3> L;
  };

  inline auto build(const std::string& lattice, const std::string& orient, int nx = 1, int ny = 1, int nz = 1) -> Lattice {
    auto uc = make_unit_cell(lattice, orient);
    std::vector<Atom> all;
    all.reserve(uc.atoms.size() * static_cast<size_t>(nx * ny * nz));
    for (int iz = 0; iz < nz; ++iz)
      for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix)
          for (const auto& atom : uc.atoms)
            all.push_back({atom[0] + ix * uc.L[0], atom[1] + iy * uc.L[1], atom[2] + iz * uc.L[2]});
    return {std::move(all), {uc.L[0] * nx, uc.L[1] * ny, uc.L[2] * nz}};
  }

  inline auto scaled(const Lattice& lat, double dnn) -> Lattice {
    Lattice r;
    r.L = {lat.L[0] * dnn, lat.L[1] * dnn, lat.L[2] * dnn};
    r.atoms.reserve(lat.atoms.size());
    for (const auto& a : lat.atoms) r.atoms.push_back({a[0] * dnn, a[1] * dnn, a[2] * dnn});
    return r;
  }

}  // namespace crystal

}  // namespace legacy

#endif  // DFT_LEGACY_CLASSICALDFT_HPP
