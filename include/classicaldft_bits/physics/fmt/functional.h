#ifndef CLASSICALDFT_PHYSICS_FMT_FUNCTIONAL_H
#define CLASSICALDFT_PHYSICS_FMT_FUNCTIONAL_H

#include "classicaldft_bits/physics/fmt/measures.h"

#include <cmath>
#include <numbers>
#include <string>

namespace dft_core::physics::fmt {

  /**
   * @brief Abstract base for Fundamental Measure Theory functionals.
   *
   * The excess free energy density is:
   * $\Phi = -n_0\,f_1(\eta) + (n_1 n_2 - \mathbf{v}_1 \cdot \mathbf{v}_2)\,f_2(\eta)
   *        + \Phi_3\,f_3(\eta)$
   *
   * Subclasses implement $f_1, f_2, f_3$ with derivatives, $\Phi_3$ with derivatives,
   * and whether tensor measures are required.
   */
  class Functional {
   public:
    virtual ~Functional() = default;

    // ── Model-specific functions (pure virtual) ─────────────────────────────

    [[nodiscard]] virtual double f1(double eta) const = 0;
    [[nodiscard]] virtual double d_f1(double eta) const = 0;
    [[nodiscard]] virtual double d2_f1(double eta) const = 0;

    [[nodiscard]] virtual double f2(double eta) const = 0;
    [[nodiscard]] virtual double d_f2(double eta) const = 0;
    [[nodiscard]] virtual double d2_f2(double eta) const = 0;

    [[nodiscard]] virtual double f3(double eta) const = 0;
    [[nodiscard]] virtual double d_f3(double eta) const = 0;
    [[nodiscard]] virtual double d2_f3(double eta) const = 0;

    [[nodiscard]] virtual double phi3(const Measures& m) const = 0;
    [[nodiscard]] virtual double d_phi3_d_n2(const Measures& m) const = 0;
    [[nodiscard]] virtual arma::rowvec3 d_phi3_d_v2(const Measures& m) const = 0;

    [[nodiscard]] virtual bool needs_tensor() const = 0;
    [[nodiscard]] virtual std::string name() const = 0;

    // ── Tensor derivative (default: zero for non-tensor models) ─────────────

    [[nodiscard]] virtual double d_phi3_d_T(int i, int j, const Measures& m) const {
      (void)i;
      (void)j;
      (void)m;
      return 0.0;
    }

    // ── Non-virtual interface ───────────────────────────────────────────────

    /**
     * @brief Free energy density $\Phi$ at a single lattice point.
     */
    [[nodiscard]] double phi(const Measures& m) const;

    /**
     * @brief All first derivatives $\partial\Phi / \partial n_\alpha$ at a single lattice point.
     *
     * Returns a Measures where each field holds the corresponding derivative.
     */
    [[nodiscard]] Measures d_phi(const Measures& m) const;

    /**
     * @brief Bulk excess free energy density for a uniform fluid.
     * @param density Number density $\rho$.
     * @param diameter Hard-sphere diameter $d$.
     */
    [[nodiscard]] double bulk_free_energy_density(double density, double diameter) const;

    /**
     * @brief Bulk excess chemical potential $\mu_{\text{ex}} / k_B T$ for a uniform fluid.
     */
    [[nodiscard]] double bulk_excess_chemical_potential(double density, double diameter) const;
  };

  // ── Rosenfeld (1989) ──────────────────────────────────────────────────────

  /**
   * @brief Original Rosenfeld FMT: reproduces the Percus-Yevick compressibility EOS.
   *
   * $f_1 = \ln(1-\eta)$, $f_2 = 1/(1-\eta)$, $f_3 = 1/(1-\eta)^2$.
   * $\Phi_3 = (n_2^3 - 3\,n_2\,|\mathbf{v}_2|^2) / (24\pi)$.
   */
  class Rosenfeld final : public Functional {
   public:
    [[nodiscard]] double f1(double eta) const override { return std::log(1.0 - eta); }
    [[nodiscard]] double d_f1(double eta) const override { return -1.0 / (1.0 - eta); }
    [[nodiscard]] double d2_f1(double eta) const override {
      double e = 1.0 - eta;
      return -1.0 / (e * e);
    }

    [[nodiscard]] double f2(double eta) const override { return 1.0 / (1.0 - eta); }
    [[nodiscard]] double d_f2(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / (e * e);
    }
    [[nodiscard]] double d2_f2(double eta) const override {
      double e = 1.0 - eta;
      return 2.0 / (e * e * e);
    }

    [[nodiscard]] double f3(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / (e * e);
    }
    [[nodiscard]] double d_f3(double eta) const override {
      double e = 1.0 - eta;
      return 2.0 / (e * e * e);
    }
    [[nodiscard]] double d2_f3(double eta) const override {
      double e = 1.0 - eta;
      return 6.0 / (e * e * e * e);
    }

    [[nodiscard]] double phi3(const Measures& m) const override {
      constexpr double inv_24pi = 1.0 / (24.0 * std::numbers::pi);
      return inv_24pi * (m.n2 * m.n2 * m.n2 - 3.0 * m.n2 * m.v2_dot_v2);
    }

    [[nodiscard]] double d_phi3_d_n2(const Measures& m) const override {
      constexpr double inv_8pi = 1.0 / (8.0 * std::numbers::pi);
      return inv_8pi * (m.n2 * m.n2 - m.v2_dot_v2);
    }

    [[nodiscard]] arma::rowvec3 d_phi3_d_v2(const Measures& m) const override {
      constexpr double inv_4pi = 1.0 / (4.0 * std::numbers::pi);
      return -inv_4pi * m.n2 * m.v2;
    }

    [[nodiscard]] bool needs_tensor() const override { return false; }
    [[nodiscard]] std::string name() const override { return "Rosenfeld"; }
  };

  // ── RSLT ──────────────────────────────────────────────────────────────────

  /**
   * @brief Rosenfeld-Tarazona RSLT functional.
   *
   * Modified $f_3$ for dimensional crossover.
   * $\Phi_3 = n_2^3 (1 - \xi)^3 / (36\pi)$ where $\xi = |\mathbf{v}_2|^2 / n_2^2$.
   */
  class RSLT final : public Functional {
   public:
    [[nodiscard]] double f1(double eta) const override { return std::log(1.0 - eta); }
    [[nodiscard]] double d_f1(double eta) const override { return -1.0 / (1.0 - eta); }
    [[nodiscard]] double d2_f1(double eta) const override {
      double e = 1.0 - eta;
      return -1.0 / (e * e);
    }

    [[nodiscard]] double f2(double eta) const override { return 1.0 / (1.0 - eta); }
    [[nodiscard]] double d_f2(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / (e * e);
    }
    [[nodiscard]] double d2_f2(double eta) const override {
      double e = 1.0 - eta;
      return 2.0 / (e * e * e);
    }

    [[nodiscard]] double f3(double eta) const override {
      if (std::abs(eta) < 1e-6) {
        return 1.0 + (2.0 / 3.0) * eta + 0.5 * eta * eta;
      }
      double e = 1.0 - eta;
      return 1.0 / (eta * e * e) + std::log(1.0 - eta) / (eta * eta);
    }

    [[nodiscard]] double d_f3(double eta) const override {
      if (std::abs(eta) < 1e-6) {
        return 2.0 / 3.0 + eta;
      }
      double e = 1.0 - eta;
      return -(1.0 - 3.0 * eta) / (eta * eta * e * e * e) - 1.0 / (eta * eta * e) -
          2.0 * std::log(1.0 - eta) / (eta * eta * eta);
    }

    [[nodiscard]] double d2_f3(double eta) const override {
      double h = 1e-5;
      return (d_f3(eta + h) - d_f3(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double phi3(const Measures& m) const override {
      if (m.n2 < 1e-30)
        return 0.0;
      double xi = std::min(m.v2_dot_v2 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double inv_36pi = 1.0 / (36.0 * std::numbers::pi);
      return inv_36pi * m.n2 * m.n2 * m.n2 * q * q * q;
    }

    [[nodiscard]] double d_phi3_d_n2(const Measures& m) const override {
      if (m.n2 < 1e-30)
        return 0.0;
      double xi = std::min(m.v2_dot_v2 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double inv_36pi = 1.0 / (36.0 * std::numbers::pi);
      return inv_36pi * 3.0 * m.n2 * m.n2 * q * q * (1.0 + xi);
    }

    [[nodiscard]] arma::rowvec3 d_phi3_d_v2(const Measures& m) const override {
      if (m.n2 < 1e-30)
        return arma::zeros<arma::rowvec>(3);
      double xi = std::min(m.v2_dot_v2 / (m.n2 * m.n2), 1.0);
      double q = 1.0 - xi;
      constexpr double inv_36pi = 1.0 / (36.0 * std::numbers::pi);
      return inv_36pi * (-6.0) * m.n2 * q * q * m.v2;
    }

    [[nodiscard]] bool needs_tensor() const override { return false; }
    [[nodiscard]] std::string name() const override { return "RSLT"; }
  };

  // ── Explicitly stable Phi3 (Tarazona 2000) ────────────────────────────────

  namespace detail {

    /**
     * @brief Tensor-based $\Phi_3$ for White Bear models.
     *
     * $\Phi_3 = \frac{1}{24\pi}\bigl[\tfrac{1}{2}n_2(n_2^2 + \mathrm{Tr}(\mathbf{T}^2))
     *           - \tfrac{3}{2}(n_2|\mathbf{v}_2|^2 - \mathbf{v}_2^T \mathbf{T} \mathbf{v}_2)\bigr]$
     */
    struct TensorPhi3 {
      [[nodiscard]] double value(const Measures& m) const {
        constexpr double inv_24pi = 1.0 / (24.0 * std::numbers::pi);
        return inv_24pi * (0.5 * m.n2 * (m.n2 * m.n2 + m.trace_T2) - 1.5 * (m.n2 * m.v2_dot_v2 - m.v_T_v));
      }

      [[nodiscard]] double d_n2(const Measures& m) const {
        constexpr double inv_24pi = 1.0 / (24.0 * std::numbers::pi);
        return inv_24pi * (1.5 * m.n2 * m.n2 + 0.5 * m.trace_T2 - 1.5 * m.v2_dot_v2);
      }

      [[nodiscard]] arma::rowvec3 d_v2(const Measures& m) const {
        constexpr double inv_24pi = 1.0 / (24.0 * std::numbers::pi);
        return inv_24pi * 3.0 * ((m.v2 * m.T) - m.n2 * m.v2);
      }

      [[nodiscard]] double d_T(int i, int j, const Measures& m) const {
        constexpr double inv_24pi = 1.0 / (24.0 * std::numbers::pi);
        return inv_24pi * (m.n2 * m.T(i, j) + 1.5 * m.v2(i) * m.v2(j));
      }
    };

  }  // namespace detail

  // ── White Bear Mark I (Roth et al. 2002) ──────────────────────────────────

  /**
   * @brief White Bear I: reproduces the Carnahan-Starling equation of state.
   *
   * Uses the RSLT $f_1, f_2, f_3$ with tensor-based $\Phi_3$.
   */
  class WhiteBearI final : public Functional {
   public:
    [[nodiscard]] double f1(double eta) const override { return std::log(1.0 - eta); }
    [[nodiscard]] double d_f1(double eta) const override { return -1.0 / (1.0 - eta); }
    [[nodiscard]] double d2_f1(double eta) const override {
      double e = 1.0 - eta;
      return -1.0 / (e * e);
    }

    [[nodiscard]] double f2(double eta) const override { return 1.0 / (1.0 - eta); }
    [[nodiscard]] double d_f2(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / (e * e);
    }
    [[nodiscard]] double d2_f2(double eta) const override {
      double e = 1.0 - eta;
      return 2.0 / (e * e * e);
    }

    [[nodiscard]] double f3(double eta) const override {
      if (std::abs(eta) < 1e-6) {
        return 1.0 + (2.0 / 3.0) * eta + 0.5 * eta * eta;
      }
      double e = 1.0 - eta;
      return 1.0 / (eta * e * e) + std::log(1.0 - eta) / (eta * eta);
    }

    [[nodiscard]] double d_f3(double eta) const override {
      if (std::abs(eta) < 1e-6) {
        return 2.0 / 3.0 + eta;
      }
      double e = 1.0 - eta;
      return -(1.0 - 3.0 * eta) / (eta * eta * e * e * e) - 1.0 / (eta * eta * e) -
          2.0 * std::log(1.0 - eta) / (eta * eta * eta);
    }

    [[nodiscard]] double d2_f3(double eta) const override {
      double h = 1e-5;
      return (d_f3(eta + h) - d_f3(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double phi3(const Measures& m) const override { return es_.value(m); }
    [[nodiscard]] double d_phi3_d_n2(const Measures& m) const override { return es_.d_n2(m); }
    [[nodiscard]] arma::rowvec3 d_phi3_d_v2(const Measures& m) const override { return es_.d_v2(m); }
    [[nodiscard]] double d_phi3_d_T(int i, int j, const Measures& m) const override { return es_.d_T(i, j, m); }

    [[nodiscard]] bool needs_tensor() const override { return true; }
    [[nodiscard]] std::string name() const override { return "WhiteBearI"; }

   private:
    static constexpr detail::TensorPhi3 es_{};
  };

  // ── White Bear Mark II (Hansen-Goos & Roth 2006) ──────────────────────────

  /**
   * @brief White Bear II: improved thermodynamic consistency.
   *
   * Modified $f_2$, $f_3$ with tensor-based $\Phi_3$.
   */
  class WhiteBearII final : public Functional {
   public:
    [[nodiscard]] double f1(double eta) const override { return std::log(1.0 - eta); }
    [[nodiscard]] double d_f1(double eta) const override { return -1.0 / (1.0 - eta); }
    [[nodiscard]] double d2_f1(double eta) const override {
      double e = 1.0 - eta;
      return -1.0 / (e * e);
    }

    [[nodiscard]] double f2(double eta) const override {
      double e = 1.0 - eta;
      return 1.0 / e + eta * eta / (3.0 * e * e * e);
    }

    [[nodiscard]] double d_f2(double eta) const override {
      double e = 1.0 - eta;
      double e2 = e * e;
      double e4 = e2 * e2;
      return 1.0 / e2 + eta * (2.0 + eta) / (3.0 * e4);
    }

    [[nodiscard]] double d2_f2(double eta) const override {
      double h = 1e-5;
      return (d_f2(eta + h) - d_f2(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double f3(double eta) const override {
      if (std::abs(eta) < 1e-6) {
        return 1.0 + (2.0 / 3.0) * eta + 0.5 * eta * eta;
      }
      double e = 1.0 - eta;
      double f3_rslt = 1.0 / (eta * e * e) + std::log(1.0 - eta) / (eta * eta);
      double e2 = e * e;
      double e4 = e2 * e2;
      double df2 = 1.0 / e2 + eta * (2.0 + eta) / (3.0 * e4);
      return f3_rslt - eta * eta * df2 / 3.0;
    }

    [[nodiscard]] double d_f3(double eta) const override {
      double h = 1e-5;
      return (f3(eta + h) - f3(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double d2_f3(double eta) const override {
      double h = 1e-5;
      return (d_f3(eta + h) - d_f3(eta - h)) / (2.0 * h);
    }

    [[nodiscard]] double phi3(const Measures& m) const override { return es_.value(m); }
    [[nodiscard]] double d_phi3_d_n2(const Measures& m) const override { return es_.d_n2(m); }
    [[nodiscard]] arma::rowvec3 d_phi3_d_v2(const Measures& m) const override { return es_.d_v2(m); }
    [[nodiscard]] double d_phi3_d_T(int i, int j, const Measures& m) const override { return es_.d_T(i, j, m); }

    [[nodiscard]] bool needs_tensor() const override { return true; }
    [[nodiscard]] std::string name() const override { return "WhiteBearII"; }

   private:
    static constexpr detail::TensorPhi3 es_{};
  };

}  // namespace dft_core::physics::fmt

#endif  // CLASSICALDFT_PHYSICS_FMT_FUNCTIONAL_H
