#pragma once

#include "cdft/numerics/autodiff.hpp"

#include <cstddef>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <span>
#include <stdexcept>
#include <vector>

namespace cdft::numerics {

  // ── CubicSpline (pure C++, autodiff-ready, no GSL) ────────────────────────

  class CubicSpline {
   public:
    CubicSpline(std::span<const double> x, std::span<const double> y);

    CubicSpline(const CubicSpline&) = default;
    CubicSpline& operator=(const CubicSpline&) = default;
    CubicSpline(CubicSpline&&) noexcept = default;
    CubicSpline& operator=(CubicSpline&&) noexcept = default;

    // ── Evaluation (templated for autodiff) ─────────────────────────────

    template <typename T = double>
    [[nodiscard]] T eval(T x) const {
      auto [i, dx_double] = locate(static_cast<double>(x));
      T dx = x - T(x_[i]);  // preserve dual-number tracking for autodiff
      return T(a_[i]) + dx * (T(b_[i]) + dx * (T(c_[i]) + dx * T(d_[i])));
    }

    [[nodiscard]] double operator()(double x) const { return eval(x); }

    // ── Derivatives via autodiff ────────────────────────────────────────

    [[nodiscard]] double derivative(double x) const {
      auto [f, df] = cdft::derivatives_up_to_1([this](cdft::dual v) { return eval(v); }, x);
      return df;
    }

    [[nodiscard]] double derivative2(double x) const {
      auto [f, df, d2f, d3f] = cdft::derivatives_up_to_3([this](cdft::dual3rd v) { return eval(v); }, x);
      return d2f;
    }

    // ── Integration via our own numerics ────────────────────────────────

    [[nodiscard]] double integrate(double a, double b) const;

    // ── Metadata ────────────────────────────────────────────────────────

    [[nodiscard]] double x_min() const { return x_.front(); }
    [[nodiscard]] double x_max() const { return x_.back(); }
    [[nodiscard]] std::size_t size() const { return x_.size(); }

   private:
    [[nodiscard]] std::pair<std::size_t, double> locate(double x) const;
    void compute_coefficients();

    std::vector<double> x_;
    std::vector<double> a_;  // function values (y)
    std::vector<double> b_;  // first-order coefficients
    std::vector<double> c_;  // second-order coefficients
    std::vector<double> d_;  // third-order coefficients
  };

  // ── BivariateSpline (GSL-backed, bicubic) ─────────────────────────────────

  class BivariateSpline {
   public:
    BivariateSpline(std::span<const double> x, std::span<const double> y, std::span<const double> z);
    ~BivariateSpline();

    BivariateSpline(BivariateSpline&& other) noexcept;
    BivariateSpline& operator=(BivariateSpline&& other) noexcept;

    BivariateSpline(const BivariateSpline&) = delete;
    BivariateSpline& operator=(const BivariateSpline&) = delete;

    [[nodiscard]] double operator()(double x, double y) const;
    [[nodiscard]] double deriv_x(double x, double y) const;
    [[nodiscard]] double deriv_y(double x, double y) const;
    [[nodiscard]] double deriv_xx(double x, double y) const;
    [[nodiscard]] double deriv_yy(double x, double y) const;
    [[nodiscard]] double deriv_xy(double x, double y) const;

   private:
    void release() noexcept;

    gsl_spline2d* spline_ = nullptr;
    gsl_interp_accel* xacc_ = nullptr;
    gsl_interp_accel* yacc_ = nullptr;
  };

}  // namespace cdft::numerics
