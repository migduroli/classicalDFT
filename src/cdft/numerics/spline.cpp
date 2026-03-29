#include "cdft/numerics/spline.hpp"

#include "cdft/numerics/math.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace cdft::numerics {

  // ── CubicSpline (natural boundary conditions) ─────────────────────────────

  CubicSpline::CubicSpline(std::span<const double> x, std::span<const double> y) {
    if (x.size() != y.size()) {
      throw std::invalid_argument("CubicSpline: x and y must have the same size");
    }
    if (x.size() < 3) {
      throw std::invalid_argument("CubicSpline: need at least 3 points (got " + std::to_string(x.size()) + ")");
    }

    x_.assign(x.begin(), x.end());
    a_.assign(y.begin(), y.end());

    compute_coefficients();
  }

  void CubicSpline::compute_coefficients() {
    // Natural cubic spline: tridiagonal system for second derivatives
    auto n = x_.size() - 1;

    std::vector<double> h(n);
    for (std::size_t i = 0; i < n; ++i) {
      h[i] = x_[i + 1] - x_[i];
    }

    // Right-hand side
    std::vector<double> alpha(n + 1, 0.0);
    for (std::size_t i = 1; i < n; ++i) {
      alpha[i] = 3.0 * ((a_[i + 1] - a_[i]) / h[i] - (a_[i] - a_[i - 1]) / h[i - 1]);
    }

    // Solve tridiagonal system for c coefficients
    c_.resize(n + 1, 0.0);
    std::vector<double> l(n + 1, 1.0);
    std::vector<double> mu(n + 1, 0.0);
    std::vector<double> z(n + 1, 0.0);

    for (std::size_t i = 1; i < n; ++i) {
      l[i] = 2.0 * (x_[i + 1] - x_[i - 1]) - h[i - 1] * mu[i - 1];
      mu[i] = h[i] / l[i];
      z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    // Back substitution
    c_[n] = 0.0;
    b_.resize(n);
    d_.resize(n);
    for (auto j = static_cast<long>(n) - 1; j >= 0; --j) {
      auto i = static_cast<std::size_t>(j);
      c_[i] = z[i] - mu[i] * c_[i + 1];
      b_[i] = (a_[i + 1] - a_[i]) / h[i] - h[i] * (c_[i + 1] + 2.0 * c_[i]) / 3.0;
      d_[i] = (c_[i + 1] - c_[i]) / (3.0 * h[i]);
    }

    // Resize c_ to n (drop the n+1 entry, only needed during construction)
    c_.resize(n);
  }

  std::pair<std::size_t, double> CubicSpline::locate(double x) const {
    // Clamp to domain
    if (x <= x_.front()) { return {0, x - x_.front()}; }
    if (x >= x_.back()) { return {x_.size() - 2, x - x_[x_.size() - 2]}; }

    // Binary search for interval
    auto it = std::upper_bound(x_.begin(), x_.end(), x);
    auto i = static_cast<std::size_t>(std::distance(x_.begin(), it)) - 1;
    return {i, x - x_[i]};
  }

  double CubicSpline::integrate(double a, double b) const {
    return integrate_qags([this](double x) { return eval(x); }, a, b).value;
  }

  // ── BivariateSpline ───────────────────────────────────────────────────────

  BivariateSpline::BivariateSpline(std::span<const double> x, std::span<const double> y, std::span<const double> z) {
    if (x.empty() || y.empty()) {
      throw std::invalid_argument("BivariateSpline: x and y must be non-empty");
    }
    if (z.size() != x.size() * y.size()) {
      throw std::invalid_argument(
          "BivariateSpline: z size (" + std::to_string(z.size()) + ") must equal nx*ny (" +
          std::to_string(x.size() * y.size()) + ")");
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

  BivariateSpline::~BivariateSpline() { release(); }

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

  double BivariateSpline::operator()(double x, double y) const { return gsl_spline2d_eval(spline_, x, y, xacc_, yacc_); }
  double BivariateSpline::deriv_x(double x, double y) const { return gsl_spline2d_eval_deriv_x(spline_, x, y, xacc_, yacc_); }
  double BivariateSpline::deriv_y(double x, double y) const { return gsl_spline2d_eval_deriv_y(spline_, x, y, xacc_, yacc_); }
  double BivariateSpline::deriv_xx(double x, double y) const { return gsl_spline2d_eval_deriv_xx(spline_, x, y, xacc_, yacc_); }
  double BivariateSpline::deriv_yy(double x, double y) const { return gsl_spline2d_eval_deriv_yy(spline_, x, y, xacc_, yacc_); }
  double BivariateSpline::deriv_xy(double x, double y) const { return gsl_spline2d_eval_deriv_xy(spline_, x, y, xacc_, yacc_); }

  void BivariateSpline::release() noexcept {
    if (spline_) { gsl_spline2d_free(spline_); spline_ = nullptr; }
    if (xacc_) { gsl_interp_accel_free(xacc_); xacc_ = nullptr; }
    if (yacc_) { gsl_interp_accel_free(yacc_); yacc_ = nullptr; }
  }

}  // namespace cdft::numerics
