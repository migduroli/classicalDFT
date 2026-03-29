#pragma once

#include "cdft/numerics/autodiff.hpp"

#include <array>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <variant>

#include <armadillo>

namespace cdft::functional {

  // ── Contractions: derived products from measures ──────────────────────────

  template <typename Scalar = double>
  struct Contractions {
    Scalar dot_v1_v2 = 0.0;
    Scalar norm_v2_squared = 0.0;
    Scalar quadratic_v2_T = 0.0;
    Scalar trace_T_squared = 0.0;
    Scalar trace_T_cubed = 0.0;
  };

  // ── Measures: the 19-component weighted density set ───────────────────────

  template <typename Scalar = double>
  struct Measures {
    Scalar eta = 0.0;
    Scalar n0 = 0.0;
    Scalar n1 = 0.0;
    Scalar n2 = 0.0;

    arma::rowvec3 v1 = arma::zeros<arma::rowvec>(3);
    arma::rowvec3 v2 = arma::zeros<arma::rowvec>(3);
    arma::mat33 T = arma::zeros<arma::mat>(3, 3);

    Contractions<Scalar> contractions;

    void compute_contractions() {
      contractions.dot_v1_v2 = arma::dot(v1, v2);
      contractions.norm_v2_squared = arma::dot(v2, v2);
      arma::mat33 t2 = T * T;
      contractions.trace_T_squared = arma::trace(t2);
      contractions.trace_T_cubed = arma::trace(t2 * T);
      contractions.quadratic_v2_T = arma::as_scalar(v2 * T * v2.t());
    }

    [[nodiscard]] static Measures uniform(double density, double diameter) {
      Measures m;
      double r = 0.5 * diameter;

      m.eta = (std::numbers::pi / 6.0) * density * diameter * diameter * diameter;
      m.n2 = std::numbers::pi * density * diameter * diameter;
      m.n1 = r * density;
      m.n0 = density;

      m.v1.zeros();
      m.v2.zeros();

      double t_diag = m.n2 / 3.0;
      m.T.zeros();
      m.T(0, 0) = t_diag;
      m.T(1, 1) = t_diag;
      m.T(2, 2) = t_diag;

      m.compute_contractions();
      return m;
    }
  };

  // ── FMT model structs (flat, no inheritance) ──────────────────────────────
  //
  // Each model implements:
  //   ideal_factor(eta)            — templated on Scalar, autodiff-ready
  //   pair_factor(eta)             — templated on Scalar, autodiff-ready
  //   triplet_factor(eta)          — templated on Scalar, autodiff-ready
  //   mixing_term(m)               — Phi_3 value
  //   mixing_term_d_n2(m)          — dPhi_3/dn2
  //   mixing_term_d_v2(m)          — dPhi_3/dv2
  //   mixing_term_d_T(i, j, m)    — dPhi_3/dT_ij (zero for non-tensor models)
  //   needs_tensor()               — whether the model uses the tensor term
  //   name()                       — model name

  struct Rosenfeld {
    [[nodiscard]] static constexpr bool needs_tensor() { return false; }
    [[nodiscard]] static std::string name() { return "Rosenfeld"; }

    template <typename T = double>
    [[nodiscard]] static T ideal_factor(T eta) {
      using std::log;
      return log(T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T pair_factor(T eta) {
      return T(1.0) / (T(1.0) - eta);
    }

    template <typename T = double>
    [[nodiscard]] static T triplet_factor(T eta) {
      T e = T(1.0) - eta;
      return T(1.0) / (e * e);
    }

    [[nodiscard]] static double mixing_term(const Measures<>& m) {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (m.n2 * m.n2 * m.n2 - 3.0 * m.n2 * m.contractions.norm_v2_squared);
    }

    [[nodiscard]] static double mixing_term_d_n2(const Measures<>& m) {
      constexpr double INV_8PI = 1.0 / (8.0 * std::numbers::pi);
      return INV_8PI * (m.n2 * m.n2 - m.contractions.norm_v2_squared);
    }

    [[nodiscard]] static arma::rowvec3 mixing_term_d_v2(const Measures<>& m) {
      constexpr double INV_4PI = 1.0 / (4.0 * std::numbers::pi);
      return -INV_4PI * m.n2 * m.v2;
    }

    [[nodiscard]] static double mixing_term_d_T(int /*i*/, int /*j*/, const Measures<>& /*m*/) { return 0.0; }
  };

  struct RSLT {
    [[nodiscard]] static constexpr bool needs_tensor() { return false; }
    [[nodiscard]] static std::string name() { return "RSLT"; }

    template <typename T = double>
    [[nodiscard]] static T ideal_factor(T eta) {
      return Rosenfeld::ideal_factor(eta);
    }

    template <typename T = double>
    [[nodiscard]] static T pair_factor(T eta) {
      return Rosenfeld::pair_factor(eta);
    }

    template <typename T = double>
    [[nodiscard]] static T triplet_factor(T eta) {
      using std::abs;
      using std::log;
      if (abs(eta) < 1e-6) return T(1.0) + (T(2.0) / T(3.0)) * eta + T(0.5) * eta * eta;
      T e = T(1.0) - eta;
      return T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
    }

    [[nodiscard]] static double mixing_term(const Measures<>& m) {
      if (m.n2 < 1e-30) return 0.0;
      double xi = std::min(m.contractions.norm_v2_squared / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * m.n2 * m.n2 * m.n2 * q * q * q;
    }

    [[nodiscard]] static double mixing_term_d_n2(const Measures<>& m) {
      if (m.n2 < 1e-30) return 0.0;
      double xi = std::min(m.contractions.norm_v2_squared / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * 3.0 * m.n2 * m.n2 * q * q * (1.0 + xi);
    }

    [[nodiscard]] static arma::rowvec3 mixing_term_d_v2(const Measures<>& m) {
      if (m.n2 < 1e-30) return arma::zeros<arma::rowvec>(3);
      double xi = std::min(m.contractions.norm_v2_squared / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * (-6.0) * m.n2 * q * q * m.v2;
    }

    [[nodiscard]] static double mixing_term_d_T(int /*i*/, int /*j*/, const Measures<>& /*m*/) { return 0.0; }
  };

  // ── Shared tensor mixing term (White Bear models) ─────────────────────────

  namespace detail {

    [[nodiscard]] inline double tensor_mixing_value(const Measures<>& m) {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (0.5 * m.n2 * (m.n2 * m.n2 + m.contractions.trace_T_squared)
                         - 1.5 * (m.n2 * m.contractions.norm_v2_squared - m.contractions.quadratic_v2_T));
    }

    [[nodiscard]] inline double tensor_mixing_d_n2(const Measures<>& m) {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (1.5 * m.n2 * m.n2 + 0.5 * m.contractions.trace_T_squared
                         - 1.5 * m.contractions.norm_v2_squared);
    }

    [[nodiscard]] inline arma::rowvec3 tensor_mixing_d_v2(const Measures<>& m) {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * 3.0 * ((m.v2 * m.T) - m.n2 * m.v2);
    }

    [[nodiscard]] inline double tensor_mixing_d_T(int i, int j, const Measures<>& m) {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (m.n2 * m.T(i, j) + 1.5 * m.v2(i) * m.v2(j));
    }

  }  // namespace detail

  struct WhiteBearI {
    [[nodiscard]] static constexpr bool needs_tensor() { return true; }
    [[nodiscard]] static std::string name() { return "WhiteBearI"; }

    template <typename T = double>
    [[nodiscard]] static T ideal_factor(T eta) {
      return Rosenfeld::ideal_factor(eta);
    }

    template <typename T = double>
    [[nodiscard]] static T pair_factor(T eta) {
      return Rosenfeld::pair_factor(eta);
    }

    template <typename T = double>
    [[nodiscard]] static T triplet_factor(T eta) {
      return RSLT::triplet_factor(eta);
    }

    [[nodiscard]] static double mixing_term(const Measures<>& m) { return detail::tensor_mixing_value(m); }
    [[nodiscard]] static double mixing_term_d_n2(const Measures<>& m) { return detail::tensor_mixing_d_n2(m); }
    [[nodiscard]] static arma::rowvec3 mixing_term_d_v2(const Measures<>& m) { return detail::tensor_mixing_d_v2(m); }
    [[nodiscard]] static double mixing_term_d_T(int i, int j, const Measures<>& m) { return detail::tensor_mixing_d_T(i, j, m); }
  };

  struct WhiteBearII {
    [[nodiscard]] static constexpr bool needs_tensor() { return true; }
    [[nodiscard]] static std::string name() { return "WhiteBearII"; }

    template <typename T = double>
    [[nodiscard]] static T ideal_factor(T eta) {
      return Rosenfeld::ideal_factor(eta);
    }

    template <typename T = double>
    [[nodiscard]] static T pair_factor(T eta) {
      T e = T(1.0) - eta;
      return T(1.0) / e + eta * eta / (T(3.0) * e * e * e);
    }

    template <typename T = double>
    [[nodiscard]] static T triplet_factor(T eta) {
      using std::abs;
      using std::log;
      if (abs(eta) < 1e-6) return T(1.0) + (T(2.0) / T(3.0)) * eta + T(0.5) * eta * eta;
      T e = T(1.0) - eta;
      // f3_rslt
      T f3_rslt = T(1.0) / (eta * e * e) + log(T(1.0) - eta) / (eta * eta);
      // df2/deta for WB-II
      T e2 = e * e;
      T df2 = T(1.0) / e2 + eta * (T(2.0) + eta) / (T(3.0) * e2 * e2);
      return f3_rslt - eta * eta * df2 / T(3.0);
    }

    [[nodiscard]] static double mixing_term(const Measures<>& m) { return detail::tensor_mixing_value(m); }
    [[nodiscard]] static double mixing_term_d_n2(const Measures<>& m) { return detail::tensor_mixing_d_n2(m); }
    [[nodiscard]] static arma::rowvec3 mixing_term_d_v2(const Measures<>& m) { return detail::tensor_mixing_d_v2(m); }
    [[nodiscard]] static double mixing_term_d_T(int i, int j, const Measures<>& m) { return detail::tensor_mixing_d_T(i, j, m); }
  };

  // ── Sum type ──────────────────────────────────────────────────────────────

  using FMTModel = std::variant<Rosenfeld, RSLT, WhiteBearI, WhiteBearII>;

  // ── Free functions: dispatch through variant ──────────────────────────────

  [[nodiscard]] inline bool fmt_needs_tensor(const FMTModel& model) {
    return std::visit([](const auto& m) { return m.needs_tensor(); }, model);
  }

  [[nodiscard]] inline std::string fmt_name(const FMTModel& model) {
    return std::visit([](const auto& m) { return m.name(); }, model);
  }

  [[nodiscard]] inline double fmt_ideal_factor(const FMTModel& model, double eta) {
    return std::visit([eta](const auto& m) { return static_cast<double>(m.ideal_factor(eta)); }, model);
  }

  [[nodiscard]] inline double fmt_pair_factor(const FMTModel& model, double eta) {
    return std::visit([eta](const auto& m) { return static_cast<double>(m.pair_factor(eta)); }, model);
  }

  [[nodiscard]] inline double fmt_triplet_factor(const FMTModel& model, double eta) {
    return std::visit([eta](const auto& m) { return static_cast<double>(m.triplet_factor(eta)); }, model);
  }

  [[nodiscard]] inline double fmt_mixing_term(const FMTModel& model, const Measures<>& m) {
    return std::visit([&m](const auto& mod) { return mod.mixing_term(m); }, model);
  }

  // ── Phi and dPhi: the core FMT evaluation ─────────────────────────────────

  [[nodiscard]] double fmt_phi(const FMTModel& model, const Measures<>& m);
  [[nodiscard]] Measures<> fmt_d_phi(const FMTModel& model, const Measures<>& m);
  [[nodiscard]] double fmt_bulk_free_energy_density(const FMTModel& model, double density, double diameter);
  [[nodiscard]] double fmt_bulk_excess_chemical_potential(const FMTModel& model, double density, double diameter);

}  // namespace cdft::functional
