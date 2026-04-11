#ifndef DFT_PHYSICS_POTENTIALS_HPP
#define DFT_PHYSICS_POTENTIALS_HPP

#include "dft/math/integration.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace dft::physics::potentials {

  static constexpr double MAX_POTENTIAL_VALUE = 1e50;

  // Split scheme controlling how the potential is decomposed into
  // repulsive and attractive parts.
  enum class SplitScheme {
    WeeksChandlerAndersen, // split at r_min (WCA)
    BarkerHenderson,       // split at r_zero (BH)
  };

  // Parse a split scheme name (case-insensitive).

  [[nodiscard]] inline auto parse_split_scheme(const std::string& name) -> SplitScheme {
    if (name == "WeeksChandlerAndersen" || name == "WCA")
      return SplitScheme::WeeksChandlerAndersen;
    return SplitScheme::BarkerHenderson;
  }

  // CRTP base providing shared computed properties for all potential types.
  // Derived must provide: hard_core_diameter(), split_point(), repulsive(),
  // attractive(), and r_cutoff (data member).

  template <typename Derived> struct PotentialBase {
    [[nodiscard]] auto hard_sphere_diameter(double kT, SplitScheme scheme) const -> double {
      const auto& self = static_cast<const Derived&>(*this);
      double d_core = self.hard_core_diameter();
      double r_split = self.split_point(scheme);
      auto kernel = [&](double r) -> double {
        return 1.0 - std::exp(-self.repulsive(r, scheme) / kT);
      };
      auto integrator = math::Integrator(kernel, {.absolute_tolerance = 1e-10, .relative_tolerance = 1e-10});
      return d_core + integrator.integrate(d_core, r_split).value;
    }

    [[nodiscard]] auto vdw_integral(double kT, SplitScheme scheme) const -> double {
      const auto& self = static_cast<const Derived&>(*this);
      if (self.r_cutoff <= 0.0) {
        return 0.0;
      }
      double r_lower = (scheme == SplitScheme::BarkerHenderson) ? self.split_point(scheme) : 0.0;
      auto kernel = [&](double r) -> double {
        return r * r * self.attractive(r, scheme);
      };
      auto integrator = math::Integrator(kernel, {.absolute_tolerance = 1e-10, .relative_tolerance = 1e-10});
      return (2.0 * std::numbers::pi / kT) * integrator.integrate(r_lower, self.r_cutoff).value;
    }
  };

  // Lennard-Jones 12-6 potential.
  //
  // V(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6] - epsilon_shift
  // where epsilon_shift = V_raw(r_cutoff) if cutoff > 0.
  struct LennardJones : PotentialBase<LennardJones> {
    static constexpr std::string_view NAME = "LennardJones";

    double sigma{1.0};
    double epsilon{1.0};
    double r_cutoff{-1.0};

    // Precomputed derived quantities (set by the factory)
    double epsilon_shift{0.0};
    double r_min{0.0};
    double v_min{0.0};
    double r_zero{0.0};

    // Raw pair potential V(r)
    [[nodiscard]] auto operator()(double r) const -> double {
      double y = sigma / r;
      double y6 = y * y * y * y * y * y;
      return 4.0 * epsilon * (y6 * y6 - y6);
    }

    // V(r^2) — avoids sqrt in inner loops
    [[nodiscard]] auto from_r2(double r2) const -> double {
      double y2 = sigma * sigma / r2;
      double y6 = y2 * y2 * y2;
      return 4.0 * epsilon * (y6 * y6 - y6);
    }

    // Cut-and-shifted energy
    [[nodiscard]] auto energy(double r) const -> double { return (*this)(r)-epsilon_shift; }

    // Repulsive part (WCA or BH split)
    [[nodiscard]] auto repulsive(double r, SplitScheme scheme) const -> double {
      double v_shifted = (*this)(r)-epsilon_shift;
      if (scheme == SplitScheme::BarkerHenderson) {
        return (r < r_zero) ? v_shifted : 0.0;
      }
      return (r < r_min) ? v_shifted - v_min : 0.0;
    }

    // Attractive tail (WCA or BH split)
    [[nodiscard]] auto attractive(double r, SplitScheme scheme) const -> double {
      double rc = r_cutoff > 0.0 ? r_cutoff : 1e30;
      if (r >= rc) {
        return 0.0;
      }
      double v_shifted = (*this)(r)-epsilon_shift;
      if (scheme == SplitScheme::BarkerHenderson) {
        return (r >= r_zero) ? v_shifted : 0.0;
      }
      return (r >= r_min) ? v_shifted : v_min;
    }

    // Hard-core diameter (zero for LJ)
    [[nodiscard]] static constexpr auto hard_core_diameter() -> double { return 0.0; }

    // Split point for BH integration
    [[nodiscard]] auto split_point(SplitScheme scheme) const -> double {
      return (scheme == SplitScheme::BarkerHenderson) ? r_zero : r_min;
    }
  };

  // ten Wolde-Frenkel short-range attractive potential for globular proteins.
  //
  // V(r) = (4 eps / alpha^2) [(1/(s^2-1))^6 - alpha (1/(s^2-1))^3]
  // with s = r / sigma. Hard core at r = sigma.
  struct TenWoldeFrenkel : PotentialBase<TenWoldeFrenkel> {
    static constexpr std::string_view NAME = "TenWoldeFrenkel";

    double sigma{1.0};
    double epsilon{1.0};
    double r_cutoff{-1.0};
    double alpha{50.0};

    double epsilon_shift{0.0};
    double r_min{0.0};
    double v_min{0.0};
    double r_zero{0.0};

    [[nodiscard]] auto operator()(double r) const -> double {
      if (r < sigma) {
        return MAX_POTENTIAL_VALUE;
      }
      double s = r / sigma;
      double y = 1.0 / (s * s - 1.0);
      double y3 = y * y * y;
      return (4.0 * epsilon / (alpha * alpha)) * (y3 * y3 - alpha * y3);
    }

    [[nodiscard]] auto from_r2(double r2) const -> double {
      if (r2 < sigma * sigma) {
        return MAX_POTENTIAL_VALUE;
      }
      double s2 = r2 / (sigma * sigma);
      double y = 1.0 / (s2 - 1.0);
      double y3 = y * y * y;
      return (4.0 * epsilon / (alpha * alpha)) * (y3 * y3 - alpha * y3);
    }

    [[nodiscard]] auto energy(double r) const -> double { return (*this)(r)-epsilon_shift; }

    [[nodiscard]] auto repulsive(double r, SplitScheme scheme) const -> double {
      double v_shifted = (*this)(r)-epsilon_shift;
      if (scheme == SplitScheme::BarkerHenderson) {
        return (r < r_zero) ? v_shifted : 0.0;
      }
      return (r < r_min) ? v_shifted - v_min : 0.0;
    }

    [[nodiscard]] auto attractive(double r, SplitScheme scheme) const -> double {
      double rc = r_cutoff > 0.0 ? r_cutoff : 1e30;
      if (r >= rc) {
        return 0.0;
      }
      double v_shifted = (*this)(r)-epsilon_shift;
      if (scheme == SplitScheme::BarkerHenderson) {
        return (r >= r_zero) ? v_shifted : 0.0;
      }
      return (r >= r_min) ? v_shifted : v_min;
    }

    [[nodiscard]] auto hard_core_diameter() const -> double { return sigma; }

    [[nodiscard]] auto split_point(SplitScheme scheme) const -> double {
      return (scheme == SplitScheme::BarkerHenderson) ? r_zero : r_min;
    }
  };

  // Wang-Ramirez-Dobnikar-Frenkel (WRDF/WHDF) potential.
  //
  // V(r) = eps_eff (sigma^2/r^2 - 1)(r_c^2/r^2 - 1)^2
  // Finite-ranged, vanishes quadratically at r_cutoff.
  struct WangRamirezDobnikarFrenkel : PotentialBase<WangRamirezDobnikarFrenkel> {
    static constexpr std::string_view NAME = "WangRamirezDobnikarFrenkel";

    double sigma{1.0};
    double epsilon{1.0};
    double r_cutoff{3.0};

    double epsilon_effective{0.0};
    double r_min{0.0};
    double v_min{0.0};

    [[nodiscard]] auto operator()(double r) const -> double {
      if (r >= r_cutoff) {
        return 0.0;
      }
      double y = sigma / r;
      double z = r_cutoff / r;
      return epsilon_effective * (y * y - 1.0) * (z * z - 1.0) * (z * z - 1.0);
    }

    [[nodiscard]] auto from_r2(double r2) const -> double {
      double rc2 = r_cutoff * r_cutoff;
      if (r2 >= rc2) {
        return 0.0;
      }
      double y = sigma * sigma / r2;
      double z = rc2 / r2;
      return epsilon_effective * (y - 1.0) * (z - 1.0) * (z - 1.0);
    }

    // WRDF is already finite-ranged — no shift needed
    [[nodiscard]] auto energy(double r) const -> double { return (*this)(r); }

    [[nodiscard]] auto repulsive(double r, SplitScheme scheme) const -> double {
      if (scheme == SplitScheme::BarkerHenderson) {
        return (r < r_min) ? (*this)(r) : 0.0;
      }
      return (r < r_min) ? (*this)(r)-v_min : 0.0;
    }

    [[nodiscard]] auto attractive(double r, SplitScheme scheme) const -> double {
      if (r >= r_cutoff) {
        return 0.0;
      }
      if (scheme == SplitScheme::BarkerHenderson) {
        return (r >= r_min) ? (*this)(r) : 0.0;
      }
      return (r >= r_min) ? (*this)(r) : v_min;
    }

    [[nodiscard]] static constexpr auto hard_core_diameter() -> double { return 0.0; }

    [[nodiscard]] auto split_point(SplitScheme scheme) const -> double {
      return (scheme == SplitScheme::BarkerHenderson) ? r_min : r_min;
    }
  };

  // Potential wrapper — hides variant, exposes unified interface.
  // Constructible from any concrete potential type.

  class Potential {
   public:
    Potential() = default;

    template <typename T>
      requires(!std::is_same_v<std::decay_t<T>, Potential>)
    Potential(T concrete) : data_(std::move(concrete)) {}

    [[nodiscard]] auto operator()(double r) const -> double {
      return std::visit([r](const auto& p) { return p(r); }, data_);
    }

    [[nodiscard]] auto from_r2(double r2) const -> double {
      return std::visit([r2](const auto& p) { return p.from_r2(r2); }, data_);
    }

    [[nodiscard]] auto energy(double r) const -> double {
      return std::visit([r](const auto& p) { return p.energy(r); }, data_);
    }

    [[nodiscard]] auto repulsive(double r, SplitScheme scheme) const -> double {
      return std::visit([r, scheme](const auto& p) { return p.repulsive(r, scheme); }, data_);
    }

    [[nodiscard]] auto attractive(double r, SplitScheme scheme) const -> double {
      return std::visit([r, scheme](const auto& p) { return p.attractive(r, scheme); }, data_);
    }

    [[nodiscard]] auto hard_core_diameter() const -> double {
      return std::visit([](const auto& p) { return p.hard_core_diameter(); }, data_);
    }

    [[nodiscard]] auto split_point(SplitScheme scheme) const -> double {
      return std::visit([scheme](const auto& p) { return p.split_point(scheme); }, data_);
    }

    [[nodiscard]] auto hard_sphere_diameter(double kT, SplitScheme scheme) const -> double {
      return std::visit([kT, scheme](const auto& p) { return p.hard_sphere_diameter(kT, scheme); }, data_);
    }

    [[nodiscard]] auto vdw_integral(double kT, SplitScheme scheme) const -> double {
      return std::visit([kT, scheme](const auto& p) { return p.vdw_integral(kT, scheme); }, data_);
    }

    [[nodiscard]] auto r_cutoff() const -> double {
      return std::visit([](const auto& p) { return p.r_cutoff; }, data_);
    }

    [[nodiscard]] auto name() const -> std::string_view {
      return std::visit([](const auto& p) -> std::string_view { return p.NAME; }, data_);
    }

    // Access underlying variant for rare type-specific inspection
    using VariantType = std::variant<LennardJones, TenWoldeFrenkel, WangRamirezDobnikarFrenkel>;

    [[nodiscard]] auto variant() const -> const VariantType& { return data_; }

   private:
    VariantType data_;
  };

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
    lj.r_zero = sigma * std::pow(0.5 * std::sqrt(1.0 + lj.epsilon_shift / epsilon) + 0.5, -1.0 / 6.0);
    return lj;
  }

  [[nodiscard]] inline auto
  make_ten_wolde_frenkel(double sigma, double epsilon, double r_cutoff = -1.0, double alpha = 50.0) -> TenWoldeFrenkel {
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
    double half_alpha = 0.5 * alpha;
    twf.r_zero = sigma
        * std::sqrt(1.0 + std::pow(half_alpha * (std::sqrt(1.0 + twf.epsilon_shift / epsilon) + 1.0), -1.0 / 3.0));
    return twf;
  }

  [[nodiscard]] inline auto make_wang_ramirez_dobnikar_frenkel(double sigma, double epsilon, double r_cutoff = 3.0)
      -> WangRamirezDobnikarFrenkel {
    double rc_s = r_cutoff / sigma;
    double eps_eff = epsilon * 2.0 * rc_s * rc_s * std::pow(2.0 * (rc_s * rc_s - 1.0) / 3.0, -3.0);

    WangRamirezDobnikarFrenkel
        w{.sigma = sigma, .epsilon = epsilon, .r_cutoff = r_cutoff, .epsilon_effective = eps_eff};

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

} // namespace dft::physics::potentials

#endif // DFT_PHYSICS_POTENTIALS_HPP
