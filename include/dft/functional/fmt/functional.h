#ifndef DFT_FUNCTIONAL_FMT_FUNCTIONAL_H
#define DFT_FUNCTIONAL_FMT_FUNCTIONAL_H

#include "dft/functional/fmt/measures.h"
#include "dft/math/autodiff.h"

#include <cmath>
#include <numbers>
#include <string>
#include <variant>

namespace dft::functional::fmt {

  // ── Tensor-based Phi3 helper (shared by White Bear models) ────────────────

  namespace detail {

    struct TensorPhi3 {
      [[nodiscard]] static double value(const Measures& m) {
        constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
        return INV_24PI *
            (0.5 * m.n2 * (m.n2 * m.n2 + m.products.trace_T2) -
             1.5 * (m.n2 * m.products.dot_v1_v1 - m.products.quadratic_form));
      }

      [[nodiscard]] static double d_n2(const Measures& m) {
        constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
        return INV_24PI * (1.5 * m.n2 * m.n2 + 0.5 * m.products.trace_T2 - 1.5 * m.products.dot_v1_v1);
      }

      [[nodiscard]] static arma::rowvec3 d_v1(const Measures& m) {
        constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
        return INV_24PI * 3.0 * ((m.v1 * m.T) - m.n2 * m.v1);
      }

      [[nodiscard]] static double d_T(int i, int j, const Measures& m) {
        constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
        return INV_24PI * (m.n2 * m.T(i, j) + 1.5 * m.v1(i) * m.v1(j));
      }
    };

  }  // namespace detail

  // ── Rosenfeld (1989) ──────────────────────────────────────────────────────

  struct Rosenfeld {
    static constexpr bool NEEDS_TENSOR = false;

    template <typename T = double>
    [[nodiscard]] static T f1(T eta) {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T f2(T eta) {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T f3(T eta) {
      T e = T(1.0) - eta;
      return T(1.0) / (e * e);
    }

    [[nodiscard]] static double phi3(const Measures& m) {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (m.n2 * m.n2 * m.n2 - 3.0 * m.n2 * m.products.dot_v1_v1);
    }

    [[nodiscard]] static double d_phi3_d_n2(const Measures& m) {
      constexpr double INV_8PI = 1.0 / (8.0 * std::numbers::pi);
      return INV_8PI * (m.n2 * m.n2 - m.products.dot_v1_v1);
    }

    [[nodiscard]] static arma::rowvec3 d_phi3_d_v1(const Measures& m) {
      constexpr double INV_4PI = 1.0 / (4.0 * std::numbers::pi);
      return -INV_4PI * m.n2 * m.v1;
    }

    [[nodiscard]] static std::string name() { return "Rosenfeld"; }
  };

  // ── RSLT ──────────────────────────────────────────────────────────────────

  struct RSLT {
    static constexpr bool NEEDS_TENSOR = false;

    template <typename T = double>
    [[nodiscard]] static T f1(T eta) {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T f2(T eta) {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T f3(T eta) {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        return T(1.0) + (T(2.0) / T(3.0)) * eta + T(0.5) * eta * eta;
      }
      T e = T(1.0) - eta;
      return T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
    }

    [[nodiscard]] static double phi3(const Measures& m) {
      if (m.n2 < 1e-30)
        return 0.0;
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * m.n2 * m.n2 * m.n2 * q * q * q;
    }

    [[nodiscard]] static double d_phi3_d_n2(const Measures& m) {
      if (m.n2 < 1e-30)
        return 0.0;
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * 3.0 * m.n2 * m.n2 * q * q * (1.0 + xi);
    }

    [[nodiscard]] static arma::rowvec3 d_phi3_d_v1(const Measures& m) {
      if (m.n2 < 1e-30)
        return arma::zeros<arma::rowvec>(3);
      double xi = std::min(m.products.dot_v1_v1 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * (-6.0) * m.n2 * q * q * m.v1;
    }

    [[nodiscard]] static std::string name() { return "RSLT"; }
  };

  // ── White Bear Mark I (Roth et al. 2002) ──────────────────────────────────

  struct WhiteBearI {
    static constexpr bool NEEDS_TENSOR = true;

    template <typename T = double>
    [[nodiscard]] static T f1(T eta) {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T f2(T eta) {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T f3(T eta) {
      using std::log;
      if (autodiff::val(eta) < 1e-6) {
        return T(1.0) + (T(2.0) / T(3.0)) * eta + T(0.5) * eta * eta;
      }
      T e = T(1.0) - eta;
      return T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
    }

    [[nodiscard]] static double phi3(const Measures& m) { return detail::TensorPhi3::value(m); }
    [[nodiscard]] static double d_phi3_d_n2(const Measures& m) { return detail::TensorPhi3::d_n2(m); }
    [[nodiscard]] static arma::rowvec3 d_phi3_d_v1(const Measures& m) { return detail::TensorPhi3::d_v1(m); }
    [[nodiscard]] static double d_phi3_d_T(int i, int j, const Measures& m) { return detail::TensorPhi3::d_T(i, j, m); }

    [[nodiscard]] static std::string name() { return "WhiteBearI"; }
  };

  // ── White Bear Mark II (Hansen-Goos & Roth 2006) ──────────────────────────

  struct WhiteBearII {
    static constexpr bool NEEDS_TENSOR = true;

    template <typename T = double>
    [[nodiscard]] static T f1(T eta) {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T f2(T eta) {
      T e = T(1.0) - eta;
      return T(1.0) / e + eta * eta / (T(3.0) * e * e * e);
    }

    template <typename T = double>
    [[nodiscard]] static T f3(T eta) {
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

    [[nodiscard]] static double phi3(const Measures& m) { return detail::TensorPhi3::value(m); }
    [[nodiscard]] static double d_phi3_d_n2(const Measures& m) { return detail::TensorPhi3::d_n2(m); }
    [[nodiscard]] static arma::rowvec3 d_phi3_d_v1(const Measures& m) { return detail::TensorPhi3::d_v1(m); }
    [[nodiscard]] static double d_phi3_d_T(int i, int j, const Measures& m) { return detail::TensorPhi3::d_T(i, j, m); }

    [[nodiscard]] static std::string name() { return "WhiteBearII"; }
  };

  // ── Sum type for FMT models ───────────────────────────────────────────────

  using FMTModel = std::variant<Rosenfeld, RSLT, WhiteBearI, WhiteBearII>;

  // ── FMT: wrapper class providing a clean interface over model variants ────

  class FMT {
   public:
    FMT(FMTModel model) : model_(std::move(model)) {}

    [[nodiscard]] const FMTModel& model() const noexcept { return model_; }

    [[nodiscard]] bool needs_tensor() const {
      return std::visit([](const auto& m) { return std::decay_t<decltype(m)>::NEEDS_TENSOR; }, model_);
    }

    [[nodiscard]] std::string name() const {
      return std::visit([](const auto& m) { return m.name(); }, model_);
    }

    [[nodiscard]] double ideal_factor(double eta, int order = 0) const {
      return std::visit(
          [eta, order](const auto& m) {
            if (order == 0)
              return static_cast<double>(m.f1(eta));
            auto [f, df, d2f] = math::derivatives_up_to_2([&](math::dual2nd x) { return m.f1(x); }, eta);
            return (order == 1) ? df : d2f;
          },
          model_
      );
    }

    [[nodiscard]] double pair_factor(double eta, int order = 0) const {
      return std::visit(
          [eta, order](const auto& m) {
            if (order == 0)
              return static_cast<double>(m.f2(eta));
            auto [f, df, d2f] = math::derivatives_up_to_2([&](math::dual2nd x) { return m.f2(x); }, eta);
            return (order == 1) ? df : d2f;
          },
          model_
      );
    }

    [[nodiscard]] double triplet_factor(double eta, int order = 0) const {
      return std::visit(
          [eta, order](const auto& m) {
            if (order == 0)
              return static_cast<double>(m.f3(eta));
            auto [f, df, d2f] = math::derivatives_up_to_2([&](math::dual2nd x) { return m.f3(x); }, eta);
            return (order == 1) ? df : d2f;
          },
          model_
      );
    }

    [[nodiscard]] double triplet_phi(const Measures& m) const {
      return std::visit([&](const auto& func) { return func.phi3(m); }, model_);
    }

    [[nodiscard]] double phi(const Measures& m) const {
      return std::visit(
          [&](const auto& func) {
            double e = m.eta;
            return -m.n0 * static_cast<double>(func.f1(e)) +
                (m.n1 * m.n2 - m.products.dot_v0_v1) * static_cast<double>(func.f2(e)) +
                func.phi3(m) * static_cast<double>(func.f3(e));
          },
          model_
      );
    }

    [[nodiscard]] Measures d_phi(const Measures& m) const {
      return std::visit(
          [&](const auto& func) -> Measures {
            double e = m.eta;
            auto [f1_val, df1_val] = math::derivatives_up_to_1([&](math::dual x) { return func.f1(x); }, e);
            auto [f2_val, df2_val] = math::derivatives_up_to_1([&](math::dual x) { return func.f2(x); }, e);
            auto [f3_val, df3_val] = math::derivatives_up_to_1([&](math::dual x) { return func.f3(x); }, e);
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
          model_
      );
    }

    [[nodiscard]] double bulk_free_energy_density(double density, double diameter) const {
      auto m = Measures::uniform(density, diameter);
      return phi(m);
    }

    [[nodiscard]] double bulk_excess_chemical_potential(double density, double diameter) const {
      auto m = Measures::uniform(density, diameter);
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

   private:
    FMTModel model_;
  };

}  // namespace dft::functional::fmt

#endif  // DFT_FUNCTIONAL_FMT_FUNCTIONAL_H
