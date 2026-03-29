#ifndef CLASSICALDFT_FUNCTIONAL_FMT_FUNCTIONAL_H
#define CLASSICALDFT_FUNCTIONAL_FMT_FUNCTIONAL_H

#include "classicaldft_bits/functional/fmt/data_structures.h"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

namespace dft::functional::fmt {

  class FundamentalMeasureTheory {
   public:
    virtual ~FundamentalMeasureTheory() = default;

    // ── Public factor interface (order = derivative order) ──────────────────

    [[nodiscard]] double ideal_factor(double eta, int order = 0) const {
      switch (order) {
        case 0: return ideal_factor_impl(eta);
        case 1: return ideal_factor_d1_impl(eta);
        case 2: return ideal_factor_d2_impl(eta);
        default: throw std::invalid_argument("ideal_factor: order must be 0, 1, or 2");
      }
    }

    [[nodiscard]] double pair_factor(double eta, int order = 0) const {
      switch (order) {
        case 0: return pair_factor_impl(eta);
        case 1: return pair_factor_d1_impl(eta);
        case 2: return pair_factor_d2_impl(eta);
        default: throw std::invalid_argument("pair_factor: order must be 0, 1, or 2");
      }
    }

    [[nodiscard]] double triplet_factor(double eta, int order = 0) const {
      switch (order) {
        case 0: return triplet_factor_impl(eta);
        case 1: return triplet_factor_d1_impl(eta);
        case 2: return triplet_factor_d2_impl(eta);
        default: throw std::invalid_argument("triplet_factor: order must be 0, 1, or 2");
      }
    }

    // ── Public mixing term interface ────────────────────────────────────────

    [[nodiscard]] double mixing_term(const FundamentalMeasures& m) const {
      return mixing_term_impl(m);
    }

    [[nodiscard]] double mixing_term_d_n2(const FundamentalMeasures& m) const {
      return mixing_term_d_n2_impl(m);
    }

    [[nodiscard]] arma::rowvec3 mixing_term_d_v2(const FundamentalMeasures& m) const {
      return mixing_term_d_v2_impl(m);
    }

    [[nodiscard]] double mixing_term_d_T(int i, int j, const FundamentalMeasures& m) const {
      return mixing_term_d_T_impl(i, j, m);
    }

    // ── Model properties ────────────────────────────────────────────────────

    [[nodiscard]] virtual bool needs_tensor() const = 0;
    [[nodiscard]] virtual std::string name() const = 0;

    // ── Non-virtual interface ───────────────────────────────────────────────

    [[nodiscard]] double phi(const FundamentalMeasures& m) const;
    [[nodiscard]] FundamentalMeasures d_phi(const FundamentalMeasures& m) const;
    [[nodiscard]] double bulk_free_energy_density(double density, double diameter) const;
    [[nodiscard]] double bulk_excess_chemical_potential(double density, double diameter) const;

   private:
    [[nodiscard]] virtual double ideal_factor_impl(double eta) const = 0;
    [[nodiscard]] virtual double ideal_factor_d1_impl(double eta) const = 0;
    [[nodiscard]] virtual double ideal_factor_d2_impl(double eta) const = 0;

    [[nodiscard]] virtual double pair_factor_impl(double eta) const = 0;
    [[nodiscard]] virtual double pair_factor_d1_impl(double eta) const = 0;
    [[nodiscard]] virtual double pair_factor_d2_impl(double eta) const = 0;

    [[nodiscard]] virtual double triplet_factor_impl(double eta) const = 0;
    [[nodiscard]] virtual double triplet_factor_d1_impl(double eta) const = 0;
    [[nodiscard]] virtual double triplet_factor_d2_impl(double eta) const = 0;

    [[nodiscard]] virtual double mixing_term_impl(const FundamentalMeasures& m) const = 0;
    [[nodiscard]] virtual double mixing_term_d_n2_impl(const FundamentalMeasures& m) const = 0;
    [[nodiscard]] virtual arma::rowvec3 mixing_term_d_v2_impl(const FundamentalMeasures& m) const = 0;

    [[nodiscard]] virtual double mixing_term_d_T_impl(int /*i*/, int /*j*/, const FundamentalMeasures& /*m*/) const {
      return 0.0;
    }
  };

  // ── Rosenfeld (1989) ──────────────────────────────────────────────────────

  class Rosenfeld final : public FundamentalMeasureTheory {
   public:
    [[nodiscard]] bool needs_tensor() const override { return false; }
    [[nodiscard]] std::string name() const override { return "Rosenfeld"; }

   private:
    [[nodiscard]] double ideal_factor_impl(double eta) const override { return std::log(1.0 - eta); }
    [[nodiscard]] double ideal_factor_d1_impl(double eta) const override { return -1.0 / (1.0 - eta); }
    [[nodiscard]] double ideal_factor_d2_impl(double eta) const override {
      double e = 1.0 - eta;
      return -1.0 / (e * e);
    }

    [[nodiscard]] double pair_factor_impl(double eta) const override { return 1.0 / (1.0 - eta); }
    [[nodiscard]] double pair_factor_d1_impl(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / (e * e);
    }
    [[nodiscard]] double pair_factor_d2_impl(double eta) const override {
      double e = 1.0 - eta;
      return 2.0 / (e * e * e);
    }

    [[nodiscard]] double triplet_factor_impl(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / (e * e);
    }
    [[nodiscard]] double triplet_factor_d1_impl(double eta) const override {
      double e = 1.0 - eta;
      return 2.0 / (e * e * e);
    }
    [[nodiscard]] double triplet_factor_d2_impl(double eta) const override {
      double e = 1.0 - eta;
      return 6.0 / (e * e * e * e);
    }

    [[nodiscard]] double mixing_term_impl(const FundamentalMeasures& m) const override {
      constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
      return INV_24PI * (m.n2 * m.n2 * m.n2 - 3.0 * m.n2 * m.contractions.norm_v2_squared);
    }
    [[nodiscard]] double mixing_term_d_n2_impl(const FundamentalMeasures& m) const override {
      constexpr double INV_8PI = 1.0 / (8.0 * std::numbers::pi);
      return INV_8PI * (m.n2 * m.n2 - m.contractions.norm_v2_squared);
    }
    [[nodiscard]] arma::rowvec3 mixing_term_d_v2_impl(const FundamentalMeasures& m) const override {
      constexpr double INV_4PI = 1.0 / (4.0 * std::numbers::pi);
      return -INV_4PI * m.n2 * m.v2;
    }
  };

  // ── RSLT ──────────────────────────────────────────────────────────────────

  class RSLT final : public FundamentalMeasureTheory {
   public:
    [[nodiscard]] bool needs_tensor() const override { return false; }
    [[nodiscard]] std::string name() const override { return "RSLT"; }

   private:
    [[nodiscard]] double ideal_factor_impl(double eta) const override { return std::log(1.0 - eta); }
    [[nodiscard]] double ideal_factor_d1_impl(double eta) const override { return -1.0 / (1.0 - eta); }
    [[nodiscard]] double ideal_factor_d2_impl(double eta) const override {
      double e = 1.0 - eta;
      return -1.0 / (e * e);
    }

    [[nodiscard]] double pair_factor_impl(double eta) const override { return 1.0 / (1.0 - eta); }
    [[nodiscard]] double pair_factor_d1_impl(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / (e * e);
    }
    [[nodiscard]] double pair_factor_d2_impl(double eta) const override {
      double e = 1.0 - eta;
      return 2.0 / (e * e * e);
    }

    [[nodiscard]] double triplet_factor_impl(double eta) const override {
      if (std::abs(eta) < 1e-6) return 1.0 + (2.0 / 3.0) * eta + 0.5 * eta * eta;
      double e = 1.0 - eta;
      return 1.0 / (eta * e * e) + std::log(1.0 - eta) / (eta * eta);
    }
    [[nodiscard]] double triplet_factor_d1_impl(double eta) const override {
      if (std::abs(eta) < 1e-6) return 2.0 / 3.0 + eta;
      double e = 1.0 - eta;
      return -(1.0 - 3.0 * eta) / (eta * eta * e * e * e) - 1.0 / (eta * eta * e) -
          2.0 * std::log(1.0 - eta) / (eta * eta * eta);
    }
    [[nodiscard]] double triplet_factor_d2_impl(double eta) const override {
      double h = 1e-5;
      return (triplet_factor_d1_impl(eta + h) - triplet_factor_d1_impl(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double mixing_term_impl(const FundamentalMeasures& m) const override {
      if (m.n2 < 1e-30) return 0.0;
      double xi = std::min(m.contractions.norm_v2_squared / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * m.n2 * m.n2 * m.n2 * q * q * q;
    }
    [[nodiscard]] double mixing_term_d_n2_impl(const FundamentalMeasures& m) const override {
      if (m.n2 < 1e-30) return 0.0;
      double xi = std::min(m.contractions.norm_v2_squared / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * 3.0 * m.n2 * m.n2 * q * q * (1.0 + xi);
    }
    [[nodiscard]] arma::rowvec3 mixing_term_d_v2_impl(const FundamentalMeasures& m) const override {
      if (m.n2 < 1e-30) return arma::zeros<arma::rowvec>(3);
      double xi = std::min(m.contractions.norm_v2_squared / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double INV_36PI = 1.0 / (36.0 * std::numbers::pi);
      return INV_36PI * (-6.0) * m.n2 * q * q * m.v2;
    }
  };

  // ── Tensor mixing term (shared by White Bear models) ──────────────────────

  namespace detail {

    struct TensorMixingTerm {
      [[nodiscard]] double value(const FundamentalMeasures& m) const {
        constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
        return INV_24PI * (0.5 * m.n2 * (m.n2 * m.n2 + m.contractions.trace_T_squared)
                           - 1.5 * (m.n2 * m.contractions.norm_v2_squared - m.contractions.quadratic_v2_T));
      }
      [[nodiscard]] double d_n2(const FundamentalMeasures& m) const {
        constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
        return INV_24PI * (1.5 * m.n2 * m.n2 + 0.5 * m.contractions.trace_T_squared - 1.5 * m.contractions.norm_v2_squared);
      }
      [[nodiscard]] arma::rowvec3 d_v2(const FundamentalMeasures& m) const {
        constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
        return INV_24PI * 3.0 * ((m.v2 * m.T) - m.n2 * m.v2);
      }
      [[nodiscard]] double d_T(int i, int j, const FundamentalMeasures& m) const {
        constexpr double INV_24PI = 1.0 / (24.0 * std::numbers::pi);
        return INV_24PI * (m.n2 * m.T(i, j) + 1.5 * m.v2(i) * m.v2(j));
      }
    };

  }  // namespace detail

  // ── White Bear Mark I (Roth et al. 2002) ──────────────────────────────────

  class WhiteBearI final : public FundamentalMeasureTheory {
   public:
    [[nodiscard]] bool needs_tensor() const override { return true; }
    [[nodiscard]] std::string name() const override { return "WhiteBearI"; }

   private:
    [[nodiscard]] double ideal_factor_impl(double eta) const override { return std::log(1.0 - eta); }
    [[nodiscard]] double ideal_factor_d1_impl(double eta) const override { return -1.0 / (1.0 - eta); }
    [[nodiscard]] double ideal_factor_d2_impl(double eta) const override {
      double e = 1.0 - eta;
      return -1.0 / (e * e);
    }

    [[nodiscard]] double pair_factor_impl(double eta) const override { return 1.0 / (1.0 - eta); }
    [[nodiscard]] double pair_factor_d1_impl(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / (e * e);
    }
    [[nodiscard]] double pair_factor_d2_impl(double eta) const override {
      double e = 1.0 - eta;
      return 2.0 / (e * e * e);
    }

    [[nodiscard]] double triplet_factor_impl(double eta) const override {
      if (std::abs(eta) < 1e-6) return 1.0 + (2.0 / 3.0) * eta + 0.5 * eta * eta;
      double e = 1.0 - eta;
      return 1.0 / (eta * e * e) + std::log(1.0 - eta) / (eta * eta);
    }
    [[nodiscard]] double triplet_factor_d1_impl(double eta) const override {
      if (std::abs(eta) < 1e-6) return 2.0 / 3.0 + eta;
      double e = 1.0 - eta;
      return -(1.0 - 3.0 * eta) / (eta * eta * e * e * e) - 1.0 / (eta * eta * e) -
          2.0 * std::log(1.0 - eta) / (eta * eta * eta);
    }
    [[nodiscard]] double triplet_factor_d2_impl(double eta) const override {
      double h = 1e-5;
      return (triplet_factor_d1_impl(eta + h) - triplet_factor_d1_impl(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double mixing_term_impl(const FundamentalMeasures& m) const override { return tensor_phi3_.value(m); }
    [[nodiscard]] double mixing_term_d_n2_impl(const FundamentalMeasures& m) const override { return tensor_phi3_.d_n2(m); }
    [[nodiscard]] arma::rowvec3 mixing_term_d_v2_impl(const FundamentalMeasures& m) const override { return tensor_phi3_.d_v2(m); }
    [[nodiscard]] double mixing_term_d_T_impl(int i, int j, const FundamentalMeasures& m) const override { return tensor_phi3_.d_T(i, j, m); }

    static constexpr detail::TensorMixingTerm tensor_phi3_{};
  };

  // ── White Bear Mark II (Hansen-Goos & Roth 2006) ──────────────────────────

  class WhiteBearII final : public FundamentalMeasureTheory {
   public:
    [[nodiscard]] bool needs_tensor() const override { return true; }
    [[nodiscard]] std::string name() const override { return "WhiteBearII"; }

   private:
    [[nodiscard]] double ideal_factor_impl(double eta) const override { return std::log(1.0 - eta); }
    [[nodiscard]] double ideal_factor_d1_impl(double eta) const override { return -1.0 / (1.0 - eta); }
    [[nodiscard]] double ideal_factor_d2_impl(double eta) const override {
      double e = 1.0 - eta;
      return -1.0 / (e * e);
    }

    [[nodiscard]] double pair_factor_impl(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / e + eta * eta / (3.0 * e * e * e);
    }
    [[nodiscard]] double pair_factor_d1_impl(double eta) const override {
      double e = 1.0 - eta;
      double e2 = e * e;
      double e4 = e2 * e2;
      return 1.0 / e2 + eta * (2.0 + eta) / (3.0 * e4);
    }
    [[nodiscard]] double pair_factor_d2_impl(double eta) const override {
      double h = 1e-5;
      return (pair_factor_d1_impl(eta + h) - pair_factor_d1_impl(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double triplet_factor_impl(double eta) const override {
      if (std::abs(eta) < 1e-6) return 1.0 + (2.0 / 3.0) * eta + 0.5 * eta * eta;
      double e = 1.0 - eta;
      double f3_rslt = 1.0 / (eta * e * e) + std::log(1.0 - eta) / (eta * eta);
      double e2 = e * e;
      double e4 = e2 * e2;
      double df2 = 1.0 / e2 + eta * (2.0 + eta) / (3.0 * e4);
      return f3_rslt - eta * eta * df2 / 3.0;
    }
    [[nodiscard]] double triplet_factor_d1_impl(double eta) const override {
      double h = 1e-5;
      return (triplet_factor_impl(eta + h) - triplet_factor_impl(eta - h)) / (2.0 * h);
    }
    [[nodiscard]] double triplet_factor_d2_impl(double eta) const override {
      double h = 1e-5;
      return (triplet_factor_d1_impl(eta + h) - triplet_factor_d1_impl(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double mixing_term_impl(const FundamentalMeasures& m) const override { return tensor_phi3_.value(m); }
    [[nodiscard]] double mixing_term_d_n2_impl(const FundamentalMeasures& m) const override { return tensor_phi3_.d_n2(m); }
    [[nodiscard]] arma::rowvec3 mixing_term_d_v2_impl(const FundamentalMeasures& m) const override { return tensor_phi3_.d_v2(m); }
    [[nodiscard]] double mixing_term_d_T_impl(int i, int j, const FundamentalMeasures& m) const override { return tensor_phi3_.d_T(i, j, m); }

    static constexpr detail::TensorMixingTerm tensor_phi3_{};
  };

}  // namespace dft::functional::fmt

#endif  // CLASSICALDFT_FUNCTIONAL_FMT_FUNCTIONAL_H
