#ifndef DFT_FUNCTIONALS_FMT_MODELS_HPP
#define DFT_FUNCTIONALS_FMT_MODELS_HPP

#include "dft/functionals/fmt/measures.hpp"
#include "dft/math/autodiff.hpp"

#include <cmath>
#include <numbers>
#include <string_view>
#include <variant>

namespace dft::functionals::fmt {

  // Tensor-based phi3 and its partial derivatives, shared by White Bear models.

  namespace detail {

    [[nodiscard]] inline auto tensor_phi3(const Measures& m) -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI *
          (0.5 * m.n2 * (m.n2 * m.n2 + m.products.trace_T2) -
           1.5 * (m.n2 * m.products.dot_v1_v1 - m.products.quadratic_form));
    }

    [[nodiscard]] inline auto tensor_phi3_d_n2(const Measures& m) -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (1.5 * m.n2 * m.n2 + 0.5 * m.products.trace_T2 - 1.5 * m.products.dot_v1_v1);
    }

    [[nodiscard]] inline auto tensor_phi3_d_v1(const Measures& m) -> arma::rowvec3 {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * 3.0 * ((m.v1 * m.T) - m.n2 * m.v1);
    }

    [[nodiscard]] inline auto tensor_phi3_d_T(int i, int j, const Measures& m) -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (m.n2 * m.T(i, j) + 1.5 * m.v1(i) * m.v1(j));
    }

  }  // namespace detail

  // Rosenfeld (1989)

  struct Rosenfeld {
    static constexpr bool NEEDS_TENSOR = false;
    static constexpr std::string_view NAME = "Rosenfeld";

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T {
      T e = T(1.0) - eta;
      return T(1.0) / (e * e);
    }

    [[nodiscard]] static auto phi3(const Measures& m) -> double {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (m.n2 * m.n2 * m.n2 - 3.0 * m.n2 * m.products.dot_v1_v1);
    }

    [[nodiscard]] static auto d_phi3_d_n2(const Measures& m) -> double {
      constexpr double INV_8PI = 1.0 / (8.0 * std::numbers::pi);
      return INV_8PI * (m.n2 * m.n2 - m.products.dot_v1_v1);
    }

    [[nodiscard]] static auto d_phi3_d_v1(const Measures& m) -> arma::rowvec3 {
      constexpr double INV_4PI = 1.0 / (4.0 * std::numbers::pi);
      return -INV_4PI * m.n2 * m.v1;
    }
  };

  // RSLT (Rosenfeld-Schmidt-Lowen-Tarazona)

  struct RSLT {
    static constexpr bool NEEDS_TENSOR = false;
    static constexpr std::string_view NAME = "RSLT";

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        return T(1.0) + (T(2.0) / T(3.0)) * eta + T(0.5) * eta * eta;
      }
      T e = T(1.0) - eta;
      return T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
    }

    [[nodiscard]] static auto phi3(const Measures& m) -> double {
      if (m.n2 < 1e-30)
        return 0.0;
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * m.n2 * m.n2 * m.n2 * q * q * q;
    }

    [[nodiscard]] static auto d_phi3_d_n2(const Measures& m) -> double {
      if (m.n2 < 1e-30)
        return 0.0;
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * 3.0 * m.n2 * m.n2 * q * q * (1.0 + xi);
    }

    [[nodiscard]] static auto d_phi3_d_v1(const Measures& m) -> arma::rowvec3 {
      if (m.n2 < 1e-30)
        return arma::zeros<arma::rowvec>(3);
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * (-6.0) * m.n2 * q * q * m.v1;
    }
  };

  // White Bear Mark I (Roth et al. 2002)

  struct WhiteBearI {
    static constexpr bool NEEDS_TENSOR = true;
    static constexpr std::string_view NAME = "WhiteBearI";

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        return T(1.0) + (T(2.0) / T(3.0)) * eta + T(0.5) * eta * eta;
      }
      T e = T(1.0) - eta;
      return T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
    }

    [[nodiscard]] static auto phi3(const Measures& m) -> double { return detail::tensor_phi3(m); }
    [[nodiscard]] static auto d_phi3_d_n2(const Measures& m) -> double { return detail::tensor_phi3_d_n2(m); }
    [[nodiscard]] static auto d_phi3_d_v1(const Measures& m) -> arma::rowvec3 { return detail::tensor_phi3_d_v1(m); }

    [[nodiscard]] static auto d_phi3_d_T(int i, int j, const Measures& m) -> double {
      return detail::tensor_phi3_d_T(i, j, m);
    }
  };

  // White Bear Mark II (Hansen-Goos & Roth 2006)

  struct WhiteBearII {
    static constexpr bool NEEDS_TENSOR = true;
    static constexpr std::string_view NAME = "WhiteBearII";

    template <typename T = double>
    [[nodiscard]] static auto f1(T eta) -> T {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static auto f2(T eta) -> T {
      T e = T(1.0) - eta;
      return T(1.0) / e + eta * eta / (T(3.0) * e * e * e);
    }

    template <typename T = double>
    [[nodiscard]] static auto f3(T eta) -> T {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        return T(1.0) + (T(2.0) / T(3.0)) * eta + T(0.5) * eta * eta;
      }
      T e = T(1.0) - eta;
      T f3_rslt = T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
      T e2 = e * e;
      T e4 = e2 * e2;
      T df2 = T(1.0) / e2 + eta * (T(2.0) + eta) / (T(3.0) * e4);
      return f3_rslt - eta * eta * df2 / T(3.0);
    }

    [[nodiscard]] static auto phi3(const Measures& m) -> double { return detail::tensor_phi3(m); }
    [[nodiscard]] static auto d_phi3_d_n2(const Measures& m) -> double { return detail::tensor_phi3_d_n2(m); }
    [[nodiscard]] static auto d_phi3_d_v1(const Measures& m) -> arma::rowvec3 { return detail::tensor_phi3_d_v1(m); }

    [[nodiscard]] static auto d_phi3_d_T(int i, int j, const Measures& m) -> double {
      return detail::tensor_phi3_d_T(i, j, m);
    }
  };

  using FMTModel = std::variant<Rosenfeld, RSLT, WhiteBearI, WhiteBearII>;

  // Queries

  [[nodiscard]] inline auto name(const FMTModel& model) -> std::string_view {
    return std::visit([](const auto& m) { return std::decay_t<decltype(m)>::NAME; }, model);
  }

  [[nodiscard]] inline auto needs_tensor(const FMTModel& model) -> bool {
    return std::visit([](const auto& m) { return std::decay_t<decltype(m)>::NEEDS_TENSOR; }, model);
  }

  // Scalar f-functions evaluated at a given packing fraction

  [[nodiscard]] inline auto ideal_factor(const FMTModel& model, double eta) -> double {
    return std::visit([eta](const auto& m) { return static_cast<double>(m.f1(eta)); }, model);
  }

  [[nodiscard]] inline auto pair_factor(const FMTModel& model, double eta) -> double {
    return std::visit([eta](const auto& m) { return static_cast<double>(m.f2(eta)); }, model);
  }

  [[nodiscard]] inline auto triplet_factor(const FMTModel& model, double eta) -> double {
    return std::visit([eta](const auto& m) { return static_cast<double>(m.f3(eta)); }, model);
  }

  // Derivatives of the scalar f-functions via autodiff

  [[nodiscard]] inline auto d_ideal_factor(const FMTModel& model, double eta) -> double {
    return std::visit(
        [eta](const auto& m) {
          auto [f, df] = math::derivatives_up_to_1([&](math::dual x) -> math::dual { return m.f1(x); }, eta);
          return df;
        },
        model
    );
  }

  [[nodiscard]] inline auto d_pair_factor(const FMTModel& model, double eta) -> double {
    return std::visit(
        [eta](const auto& m) {
          auto [f, df] = math::derivatives_up_to_1([&](math::dual x) -> math::dual { return m.f2(x); }, eta);
          return df;
        },
        model
    );
  }

  [[nodiscard]] inline auto d_triplet_factor(const FMTModel& model, double eta) -> double {
    return std::visit(
        [eta](const auto& m) {
          auto [f, df] = math::derivatives_up_to_1([&](math::dual x) -> math::dual { return m.f3(x); }, eta);
          return df;
        },
        model
    );
  }

  // Free energy density Phi(measures) for a given FMT model.
  // Phi = -n0 f1(eta) + (n1 n2 - v0.v1) f2(eta) + phi3(m) f3(eta)

  [[nodiscard]] inline auto phi(const FMTModel& model, const Measures& m) -> double {
    return std::visit(
        [&](const auto& func) {
          double e = m.eta;
          return -m.n0 * static_cast<double>(func.f1(e)) +
              (m.n1 * m.n2 - m.products.dot_v0_v1) * static_cast<double>(func.f2(e)) +
              func.phi3(m) * static_cast<double>(func.f3(e));
        },
        model
    );
  }

  // Functional derivatives dPhi/dn_alpha for all 19 weighted densities.
  // Returns a Measures struct with derivatives in the corresponding fields.

  [[nodiscard]] inline auto d_phi(const FMTModel& model, const Measures& m) -> Measures {
    return std::visit(
        [&](const auto& func) -> Measures {
          double e = m.eta;
          auto [f1_val, df1_val] =
              math::derivatives_up_to_1([&](math::dual x) -> math::dual { return func.f1(x); }, e);
          auto [f2_val, df2_val] =
              math::derivatives_up_to_1([&](math::dual x) -> math::dual { return func.f2(x); }, e);
          auto [f3_val, df3_val] =
              math::derivatives_up_to_1([&](math::dual x) -> math::dual { return func.f3(x); }, e);
          double p3 = func.phi3(m);

          Measures dm;
          dm.eta = -m.n0 * df1_val + (m.n1 * m.n2 - m.products.dot_v0_v1) * df2_val + p3 * df3_val;
          dm.n0 = -f1_val;
          dm.n1 = m.n2 * f2_val;
          dm.n2 = m.n1 * f2_val + func.d_phi3_d_n2(m) * f3_val;
          dm.v0 = -m.v1 * f2_val;
          dm.v1 = -m.v0 * f2_val + func.d_phi3_d_v1(m) * f3_val;

          if constexpr (std::decay_t<decltype(func)>::NEEDS_TENSOR) {
            for (int i = 0; i < 3; ++i)
              for (int j = 0; j < 3; ++j)
                dm.T(i, j) = func.d_phi3_d_T(i, j, m) * f3_val;
          } else {
            dm.T.zeros();
          }

          return dm;
        },
        model
    );
  }

  // Free energy density for a uniform fluid.

  [[nodiscard]] inline auto free_energy_density(const FMTModel& model, double density, double diameter) -> double {
    auto m = make_uniform_measures(density, diameter);
    return phi(model, m);
  }

  // Excess chemical potential for a uniform fluid via chain rule:
  // mu_ex = sum_alpha (dPhi/dn_alpha)(dn_alpha/drho)

  [[nodiscard]] inline auto excess_chemical_potential(const FMTModel& model, double density, double diameter)
      -> double {
    auto m = make_uniform_measures(density, diameter);
    auto dm = d_phi(model, m);
    double d = diameter;
    double r = 0.5 * d;

    double dn3_drho = (std::numbers::pi / 6.0) * d * d * d;
    double dn2_drho = std::numbers::pi * d * d;
    double dn1_drho = r;
    double dn0_drho = 1.0;

    double mu_ex = dm.eta * dn3_drho + dm.n2 * dn2_drho + dm.n1 * dn1_drho + dm.n0 * dn0_drho;

    if (needs_tensor(model)) {
      double dt_drho = std::numbers::pi * d * d / 3.0;
      for (int j = 0; j < 3; ++j) {
        mu_ex += dm.T(j, j) * dt_drho;
      }
    }

    return mu_ex;
  }

}  // namespace dft::functionals::fmt

#endif  // DFT_FUNCTIONALS_FMT_MODELS_HPP
