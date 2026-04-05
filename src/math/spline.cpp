#include "dft/math/spline.hpp"

#include <stdexcept>
#include <string>

namespace dft::math {

  CubicSpline::CubicSpline(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size()) {
      throw std::invalid_argument("CubicSpline: x and y must have the same size");
    }
    if (x.size() < 3) {
      throw std::invalid_argument("CubicSpline: need at least 3 points (got " + std::to_string(x.size()) + ")");
    }

    size_ = x.size();
    accel_.reset(gsl_interp_accel_alloc());
    spline_.reset(gsl_spline_alloc(gsl_interp_cspline, size_));

    if (!accel_ || !spline_) {
      throw std::runtime_error("CubicSpline: GSL allocation failed");
    }

    gsl_spline_init(spline_.get(), x.data(), y.data(), size_);
  }

  auto CubicSpline::operator()(double x) const -> double {
    return gsl_spline_eval(spline_.get(), x, accel_.get());
  }

  auto CubicSpline::derivative(double x) const -> double {
    return gsl_spline_eval_deriv(spline_.get(), x, accel_.get());
  }

  auto CubicSpline::derivative2(double x) const -> double {
    return gsl_spline_eval_deriv2(spline_.get(), x, accel_.get());
  }

  auto CubicSpline::integrate(double a, double b) const -> double {
    return gsl_spline_eval_integ(spline_.get(), a, b, accel_.get());
  }

  auto CubicSpline::x_min() const -> double {
    return spline_->x[0];
  }

  auto CubicSpline::x_max() const -> double {
    return spline_->x[size_ - 1];
  }

  auto CubicSpline::size() const -> std::size_t {
    return size_;
  }

  // BivariateSpline

  BivariateSpline::BivariateSpline(std::span<const double> x, std::span<const double> y, std::span<const double> z) {
    if (x.empty() || y.empty()) {
      throw std::invalid_argument("BivariateSpline: x and y must be non-empty");
    }
    if (z.size() != x.size() * y.size()) {
      throw std::invalid_argument(
          "BivariateSpline: z size (" + std::to_string(z.size()) + ") must equal nx*ny ("
          + std::to_string(x.size() * y.size()) + ")"
      );
    }

    xacc_.reset(gsl_interp_accel_alloc());
    yacc_.reset(gsl_interp_accel_alloc());
    spline_.reset(gsl_spline2d_alloc(gsl_interp2d_bicubic, x.size(), y.size()));

    if (!xacc_ || !yacc_ || !spline_) {
      throw std::runtime_error("BivariateSpline: GSL allocation failed");
    }

    gsl_spline2d_init(spline_.get(), x.data(), y.data(), z.data(), x.size(), y.size());
  }

  auto BivariateSpline::operator()(double x, double y) const -> double {
    return gsl_spline2d_eval(spline_.get(), x, y, xacc_.get(), yacc_.get());
  }

  auto BivariateSpline::deriv_x(double x, double y) const -> double {
    return gsl_spline2d_eval_deriv_x(spline_.get(), x, y, xacc_.get(), yacc_.get());
  }

  auto BivariateSpline::deriv_y(double x, double y) const -> double {
    return gsl_spline2d_eval_deriv_y(spline_.get(), x, y, xacc_.get(), yacc_.get());
  }

  auto BivariateSpline::deriv_xx(double x, double y) const -> double {
    return gsl_spline2d_eval_deriv_xx(spline_.get(), x, y, xacc_.get(), yacc_.get());
  }

  auto BivariateSpline::deriv_yy(double x, double y) const -> double {
    return gsl_spline2d_eval_deriv_yy(spline_.get(), x, y, xacc_.get(), yacc_.get());
  }

  auto BivariateSpline::deriv_xy(double x, double y) const -> double {
    return gsl_spline2d_eval_deriv_xy(spline_.get(), x, y, xacc_.get(), yacc_.get());
  }

} // namespace dft::math
