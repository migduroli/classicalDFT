#include "classicaldft_bits/numerics/spline.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace dft::numerics::spline {

  // ── CubicSpline ─────────────────────────────────────────────────────────────

  CubicSpline::CubicSpline(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size()) {
      throw std::invalid_argument("CubicSpline: x and y must have the same size");
    }
    if (x.size() < 3) {
      throw std::invalid_argument("CubicSpline: need at least 3 points (got " + std::to_string(x.size()) + ")");
    }

    size_ = x.size();
    accel_ = gsl_interp_accel_alloc();
    spline_ = gsl_spline_alloc(gsl_interp_cspline, size_);

    if (accel_ == nullptr || spline_ == nullptr) {
      release();
      throw std::runtime_error("CubicSpline: GSL allocation failed");
    }

    gsl_spline_init(spline_, x.data(), y.data(), size_);
  }

  CubicSpline::~CubicSpline() {
    release();
  }

  CubicSpline::CubicSpline(CubicSpline&& other) noexcept
      : spline_(other.spline_), accel_(other.accel_), size_(other.size_) {
    other.spline_ = nullptr;
    other.accel_ = nullptr;
    other.size_ = 0;
  }

  CubicSpline& CubicSpline::operator=(CubicSpline&& other) noexcept {
    if (this != &other) {
      release();
      spline_ = other.spline_;
      accel_ = other.accel_;
      size_ = other.size_;
      other.spline_ = nullptr;
      other.accel_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  double CubicSpline::operator()(double x) const {
    return gsl_spline_eval(spline_, x, accel_);
  }

  double CubicSpline::derivative(double x) const {
    return gsl_spline_eval_deriv(spline_, x, accel_);
  }

  double CubicSpline::derivative2(double x) const {
    return gsl_spline_eval_deriv2(spline_, x, accel_);
  }

  double CubicSpline::integrate(double a, double b) const {
    return gsl_spline_eval_integ(spline_, a, b, accel_);
  }

  double CubicSpline::x_min() const {
    return spline_->x[0];
  }

  double CubicSpline::x_max() const {
    return spline_->x[size_ - 1];
  }

  std::size_t CubicSpline::size() const {
    return size_;
  }

  void CubicSpline::release() noexcept {
    if (spline_ != nullptr) {
      gsl_spline_free(spline_);
      spline_ = nullptr;
    }
    if (accel_ != nullptr) {
      gsl_interp_accel_free(accel_);
      accel_ = nullptr;
    }
    size_ = 0;
  }

  // ── BivariateSpline ─────────────────────────────────────────────────────────

  BivariateSpline::BivariateSpline(std::span<const double> x, std::span<const double> y, std::span<const double> z) {
    if (x.empty() || y.empty()) {
      throw std::invalid_argument("BivariateSpline: x and y must be non-empty");
    }
    if (z.size() != x.size() * y.size()) {
      throw std::invalid_argument(
          "BivariateSpline: z size (" + std::to_string(z.size()) + ") must equal nx*ny (" +
          std::to_string(x.size() * y.size()) + ")"
      );
    }

    xacc_ = gsl_interp_accel_alloc();
    yacc_ = gsl_interp_accel_alloc();
    spline_ = gsl_spline2d_alloc(gsl_interp2d_bicubic, x.size(), y.size());

    if (xacc_ == nullptr || yacc_ == nullptr || spline_ == nullptr) {
      release();
      throw std::runtime_error("BivariateSpline: GSL allocation failed");
    }

    gsl_spline2d_init(spline_, x.data(), y.data(), z.data(), x.size(), y.size());
  }

  BivariateSpline::~BivariateSpline() {
    release();
  }

  BivariateSpline::BivariateSpline(BivariateSpline&& other) noexcept
      : spline_(other.spline_), xacc_(other.xacc_), yacc_(other.yacc_) {
    other.spline_ = nullptr;
    other.xacc_ = nullptr;
    other.yacc_ = nullptr;
  }

  BivariateSpline& BivariateSpline::operator=(BivariateSpline&& other) noexcept {
    if (this != &other) {
      release();
      spline_ = other.spline_;
      xacc_ = other.xacc_;
      yacc_ = other.yacc_;
      other.spline_ = nullptr;
      other.xacc_ = nullptr;
      other.yacc_ = nullptr;
    }
    return *this;
  }

  double BivariateSpline::operator()(double x, double y) const {
    return gsl_spline2d_eval(spline_, x, y, xacc_, yacc_);
  }

  double BivariateSpline::deriv_x(double x, double y) const {
    return gsl_spline2d_eval_deriv_x(spline_, x, y, xacc_, yacc_);
  }

  double BivariateSpline::deriv_y(double x, double y) const {
    return gsl_spline2d_eval_deriv_y(spline_, x, y, xacc_, yacc_);
  }

  double BivariateSpline::deriv_xx(double x, double y) const {
    return gsl_spline2d_eval_deriv_xx(spline_, x, y, xacc_, yacc_);
  }

  double BivariateSpline::deriv_yy(double x, double y) const {
    return gsl_spline2d_eval_deriv_yy(spline_, x, y, xacc_, yacc_);
  }

  double BivariateSpline::deriv_xy(double x, double y) const {
    return gsl_spline2d_eval_deriv_xy(spline_, x, y, xacc_, yacc_);
  }

  void BivariateSpline::release() noexcept {
    if (spline_ != nullptr) {
      gsl_spline2d_free(spline_);
      spline_ = nullptr;
    }
    if (xacc_ != nullptr) {
      gsl_interp_accel_free(xacc_);
      xacc_ = nullptr;
    }
    if (yacc_ != nullptr) {
      gsl_interp_accel_free(yacc_);
      yacc_ = nullptr;
    }
  }

}  // namespace dft::numerics::spline
