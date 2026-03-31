#ifndef DFT_PHYSICS_POTENTIALS_HPP
#define DFT_PHYSICS_POTENTIALS_HPP

#include "dft/math/integration.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <variant>

namespace dft::physics::potentials {

  static constexpr double MAX_POTENTIAL_VALUE = 1e50;

  // Split scheme controlling how the potential is decomposed into
  // repulsive and attractive parts.
  enum class SplitScheme {
    WeeksChandlerAndersen,  // split at r_min (WCA)
    BarkerHenderson,        // split at r_zero (BH)
  };

  // Lennard-Jones 12-6 potential.
  //
  // V(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6] - epsilon_shift
  // where epsilon_shift = V_raw(r_cutoff) if cutoff > 0.
  struct LennardJones {
    double sigma{1.0};
    double epsilon{1.0};
    double r_cutoff{-1.0};

    // Precomputed derived quantities (set by the factory)
    double epsilon_shift{0.0};
    double r_min{0.0};
    double v_min{0.0};
    double r_zero{0.0};
  };

  // ten Wolde-Frenkel short-range attractive potential for globular proteins.
  //
  // V(r) = (4 eps / alpha^2) [(1/(s^2-1))^6 - alpha (1/(s^2-1))^3]
  // with s = r / sigma. Hard core at r = sigma.
  struct TenWoldeFrenkel {
    double sigma{1.0};
    double epsilon{1.0};
    double r_cutoff{-1.0};
    double alpha{50.0};

    double epsilon_shift{0.0};
    double r_min{0.0};
    double v_min{0.0};
    double r_zero{0.0};
  };

  // Wang-Ramirez-Dobnikar-Frenkel (WRDF/WHDF) potential.
  //
  // V(r) = eps_eff (sigma^2/r^2 - 1)(r_c^2/r^2 - 1)^2
  // Finite-ranged, vanishes quadratically at r_cutoff.
  struct WangRamirezDobnikarFrenkel {
    double sigma{1.0};
    double epsilon{1.0};
    double r_cutoff{3.0};

    // After rescaling: eps_eff stores the normalised epsilon
    double epsilon_effective{0.0};
    double r_min{0.0};
    double v_min{0.0};
  };

  using Potential = std::variant<LennardJones, TenWoldeFrenkel, WangRamirezDobnikarFrenkel>;

  // Factories

  [[nodiscard]] inline auto make_lennard_jones(double sigma, double epsilon, double r_cutoff = -1.0) -> LennardJones {
    LennardJones lj{.sigma = sigma, .epsilon = epsilon, .r_cutoff = r_cutoff};

    auto vr = [&](double r) {
      double y = sigma / r;
      double y6 = y * y * y * y * y * y;
      return 4.0 * epsilon * (y6 * y6 - y6);
    };

    lj.epsilon_shift = (r_cutoff > 0.0) ? vr(r_cutoff) : 0.0;
    lj.r_min = std::pow(2.0, 1.0 / 6.0) * sigma;
    lj.v_min = vr(lj.r_min) - lj.epsilon_shift;
    lj.r_zero = sigma * std::pow(0.5 * std::sqrt(1.0 + lj.epsilon_shift / (4.0 * epsilon)) + 0.5, -1.0 / 6.0);
    return lj;
  }

  [[nodiscard]] inline auto make_ten_wolde_frenkel(
      double sigma, double epsilon, double r_cutoff = -1.0, double alpha = 50.0
  ) -> TenWoldeFrenkel {
    TenWoldeFrenkel twf{.sigma = sigma, .epsilon = epsilon, .r_cutoff = r_cutoff, .alpha = alpha};

    auto vr = [&](double r) -> double {
      if (r < sigma) {
        return MAX_POTENTIAL_VALUE;
      }
      double s = r / sigma;
      double y = 1.0 / (s * s - 1.0);
      double y3 = y * y * y;
      return (4.0 * epsilon / (alpha * alpha)) * (y3 * y3 - alpha * y3);
    };

    twf.epsilon_shift = (r_cutoff > 0.0) ? vr(r_cutoff) : 0.0;
    twf.r_min = sigma * std::sqrt(1.0 + std::pow(2.0 / alpha, 1.0 / 3.0));
    twf.v_min = vr(twf.r_min) - twf.epsilon_shift;
    twf.r_zero = sigma * std::sqrt(1.0 + std::pow(25.0 * std::sqrt(1.0 + twf.epsilon_shift) + 25.0, -1.0 / 3.0));
    return twf;
  }

  [[nodiscard]] inline auto make_wang_ramirez_dobnikar_frenkel(
      double sigma, double epsilon, double r_cutoff = 3.0
  ) -> WangRamirezDobnikarFrenkel {
    double rc_s = r_cutoff / sigma;
    double eps_eff = epsilon * 2.0 * rc_s * rc_s * std::pow(2.0 * (rc_s * rc_s - 1.0) / 3.0, -3.0);

    WangRamirezDobnikarFrenkel w{
        .sigma = sigma, .epsilon = epsilon, .r_cutoff = r_cutoff, .epsilon_effective = eps_eff};

    auto vr = [&](double r) -> double {
      if (r >= r_cutoff) {
        return 0.0;
      }
      double y = sigma / r;
      double z = r_cutoff / r;
      return eps_eff * (y * y - 1.0) * (z * z - 1.0) * (z * z - 1.0);
    };

    w.r_min = r_cutoff * std::pow((1.0 + 2.0 * rc_s * rc_s) / 3.0, -0.5);
    w.v_min = vr(w.r_min);
    return w;
  }

  // Raw potential energy at distance r

  [[nodiscard]] inline auto raw_energy(const LennardJones& lj, double r) -> double {
    double y = lj.sigma / r;
    double y6 = y * y * y * y * y * y;
    return 4.0 * lj.epsilon * (y6 * y6 - y6);
  }

  [[nodiscard]] inline auto raw_energy(const TenWoldeFrenkel& twf, double r) -> double {
    if (r < twf.sigma) {
      return MAX_POTENTIAL_VALUE;
    }
    double s = r / twf.sigma;
    double y = 1.0 / (s * s - 1.0);
    double y3 = y * y * y;
    return (4.0 * twf.epsilon / (twf.alpha * twf.alpha)) * (y3 * y3 - twf.alpha * y3);
  }

  [[nodiscard]] inline auto raw_energy(const WangRamirezDobnikarFrenkel& w, double r) -> double {
    if (r >= w.r_cutoff) {
      return 0.0;
    }
    double y = w.sigma / r;
    double z = w.r_cutoff / r;
    return w.epsilon_effective * (y * y - 1.0) * (z * z - 1.0) * (z * z - 1.0);
  }

  // Raw potential from r^2 (avoids sqrt)

  [[nodiscard]] inline auto raw_energy_r2(const LennardJones& lj, double r2) -> double {
    double y2 = lj.sigma * lj.sigma / r2;
    double y6 = y2 * y2 * y2;
    return 4.0 * lj.epsilon * (y6 * y6 - y6);
  }

  [[nodiscard]] inline auto raw_energy_r2(const TenWoldeFrenkel& twf, double r2) -> double {
    if (r2 < twf.sigma * twf.sigma) {
      return MAX_POTENTIAL_VALUE;
    }
    double s2 = r2 / (twf.sigma * twf.sigma);
    double y = 1.0 / (s2 - 1.0);
    double y3 = y * y * y;
    return (4.0 * twf.epsilon / (twf.alpha * twf.alpha)) * (y3 * y3 - twf.alpha * y3);
  }

  [[nodiscard]] inline auto raw_energy_r2(const WangRamirezDobnikarFrenkel& w, double r2) -> double {
    double rc2 = w.r_cutoff * w.r_cutoff;
    if (r2 >= rc2) {
      return 0.0;
    }
    double y = w.sigma * w.sigma / r2;
    double z = rc2 / r2;
    return w.epsilon_effective * (y - 1.0) * (z - 1.0) * (z - 1.0);
  }

  // Cut-and-shifted potential

  [[nodiscard]] inline auto energy(const Potential& pot, double r) -> double {
    return std::visit(
        [r](const auto& p) {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, WangRamirezDobnikarFrenkel>) {
            return raw_energy(p, r);
          } else {
            return raw_energy(p, r) - p.epsilon_shift;
          }
        },
        pot
    );
  }

  // Hard-core diameter (below which potential is infinite)

  [[nodiscard]] inline auto hard_core_diameter(const Potential& pot) -> double {
    return std::visit(
        [](const auto& p) -> double {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, TenWoldeFrenkel>) {
            return p.sigma;
          } else {
            return 0.0;
          }
        },
        pot
    );
  }

  // Position of the potential minimum

  [[nodiscard]] inline auto r_min(const Potential& pot) -> double {
    return std::visit([](const auto& p) { return p.r_min; }, pot);
  }

  // Minimum value of the potential

  [[nodiscard]] inline auto v_min(const Potential& pot) -> double {
    return std::visit([](const auto& p) { return p.v_min; }, pot);
  }

  // Name of the potential

  [[nodiscard]] inline auto name(const Potential& pot) -> std::string {
    return std::visit(
        [](const auto& p) -> std::string {
          using T = std::decay_t<decltype(p)>;
          if constexpr (std::is_same_v<T, LennardJones>) {
            return "LennardJones";
          } else if constexpr (std::is_same_v<T, TenWoldeFrenkel>) {
            return "TenWoldeFrenkel";
          } else {
            return "WangRamirezDobnikarFrenkel";
          }
        },
        pot
    );
  }

  // Repulsive part (WCA or BH split)

  [[nodiscard]] inline auto repulsive(const Potential& pot, double r, SplitScheme scheme) -> double {
    return std::visit(
        [r, scheme](const auto& p) -> double {
          using T = std::decay_t<decltype(p)>;
          double v = raw_energy(p, r);
          if constexpr (std::is_same_v<T, WangRamirezDobnikarFrenkel>) {
            // WRDF: epsilon_shift is always 0
            if (scheme == SplitScheme::BarkerHenderson) {
              return (r < p.r_min) ? v : 0.0;
            }
            return (r < p.r_min) ? v - p.v_min : 0.0;
          } else {
            double v_shifted = v - p.epsilon_shift;
            if (scheme == SplitScheme::BarkerHenderson) {
              double r_z = p.r_zero;
              return (r < r_z) ? v_shifted : 0.0;
            }
            return (r < p.r_min) ? v_shifted - p.v_min : 0.0;
          }
        },
        pot
    );
  }

  // Attractive tail (WCA or BH split)

  [[nodiscard]] inline auto attractive(const Potential& pot, double r, SplitScheme scheme) -> double {
    return std::visit(
        [r, scheme](const auto& p) -> double {
          using T = std::decay_t<decltype(p)>;
          double rc = [&]() -> double {
            if constexpr (std::is_same_v<T, WangRamirezDobnikarFrenkel>) {
              return p.r_cutoff;
            } else {
              return p.r_cutoff > 0.0 ? p.r_cutoff : 1e30;
            }
          }();
          if (r >= rc) {
            return 0.0;
          }

          double v = raw_energy(p, r);
          if constexpr (std::is_same_v<T, WangRamirezDobnikarFrenkel>) {
            if (scheme == SplitScheme::BarkerHenderson) {
              return (r >= p.r_min) ? v : 0.0;
            }
            return (r >= p.r_min) ? v : p.v_min;
          } else {
            double v_shifted = v - p.epsilon_shift;
            if (scheme == SplitScheme::BarkerHenderson) {
              return (r >= p.r_zero) ? v_shifted : 0.0;
            }
            return (r >= p.r_min) ? v_shifted : p.v_min;
          }
        },
        pot
    );
  }

  // Barker-Henderson hard-sphere diameter via numerical integration.
  // d_HS = d_core + integral_{d_core}^{r_split} [1 - exp(-w_rep(r)/kT)] dr

  [[nodiscard]] inline auto hard_sphere_diameter(const Potential& pot, double kT, SplitScheme scheme) -> double {
    double d_core = hard_core_diameter(pot);
    double r_split = std::visit(
        [scheme](const auto& p) -> double {
          using T = std::decay_t<decltype(p)>;
          if (scheme == SplitScheme::BarkerHenderson) {
            if constexpr (std::is_same_v<T, WangRamirezDobnikarFrenkel>) {
              return p.r_min;
            } else {
              return p.r_zero;
            }
          }
          return p.r_min;
        },
        pot
    );

    auto kernel = [&](double r) -> double {
      double w = repulsive(pot, r, scheme);
      return 1.0 - std::exp(-w / kT);
    };

    auto integrator = math::Integrator(kernel, {.absolute_tolerance = 1e-10, .relative_tolerance = 1e-10});
    auto result = integrator.integrate(d_core, r_split);
    return d_core + result.value;
  }

  // Van der Waals integral: a_vdw = (2 pi / kT) * integral r^2 w_att(r) dr

  [[nodiscard]] inline auto vdw_integral(const Potential& pot, double kT, SplitScheme scheme) -> double {
    double r_cutoff = std::visit(
        [](const auto& p) -> double {
          if (p.r_cutoff > 0.0) {
            return p.r_cutoff;
          }
          return 0.0;
        },
        pot
    );
    if (r_cutoff <= 0.0) {
      return 0.0;
    }

    double r_lower = (scheme == SplitScheme::BarkerHenderson) ? std::visit(
                                                                    [](const auto& p) -> double {
                                                                      using T = std::decay_t<decltype(p)>;
                                                                      if constexpr (std::is_same_v<
                                                                                        T,
                                                                                        WangRamirezDobnikarFrenkel>) {
                                                                        return p.r_min;
                                                                      } else {
                                                                        return p.r_zero;
                                                                      }
                                                                    },
                                                                    pot
                                                                )
                                                              : hard_core_diameter(pot);

    auto kernel = [&](double r) -> double { return r * r * attractive(pot, r, scheme); };

    auto integrator = math::Integrator(kernel, {.absolute_tolerance = 1e-10, .relative_tolerance = 1e-10});
    auto result = integrator.integrate(r_lower, r_cutoff);
    return (2.0 * std::numbers::pi / kT) * result.value;
  }

}  // namespace dft::physics::potentials

#endif  // DFT_PHYSICS_POTENTIALS_HPP
