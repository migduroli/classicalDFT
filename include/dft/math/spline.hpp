#ifndef DFT_MATH_SPLINE_HPP
#define DFT_MATH_SPLINE_HPP

#include <cstddef>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <memory>
#include <span>

namespace dft::math {

  // GSL RAII helpers

  struct GslSplineDeleter {
    void operator()(gsl_spline* p) const noexcept {
      if (p) gsl_spline_free(p);
    }
  };

  struct GslSpline2dDeleter {
    void operator()(gsl_spline2d* p) const noexcept {
      if (p) gsl_spline2d_free(p);
    }
  };

  struct GslAccelDeleter {
    void operator()(gsl_interp_accel* p) const noexcept {
      if (p) gsl_interp_accel_free(p);
    }
  };

  using GslSplinePtr = std::unique_ptr<gsl_spline, GslSplineDeleter>;
  using GslSpline2dPtr = std::unique_ptr<gsl_spline2d, GslSpline2dDeleter>;
  using GslAccelPtr = std::unique_ptr<gsl_interp_accel, GslAccelDeleter>;

  // 1D natural cubic spline interpolation wrapping GSL's gsl_spline.
  // Provides evaluation, first/second derivatives, and definite integration.
  // Move-only (owns GSL resources).
  class CubicSpline {
   public:
    CubicSpline(std::span<const double> x, std::span<const double> y);
    ~CubicSpline() = default;

    CubicSpline(CubicSpline&&) noexcept = default;
    CubicSpline& operator=(CubicSpline&&) noexcept = default;
    CubicSpline(const CubicSpline&) = delete;
    CubicSpline& operator=(const CubicSpline&) = delete;

    [[nodiscard]] auto operator()(double x) const -> double;
    [[nodiscard]] auto derivative(double x) const -> double;
    [[nodiscard]] auto derivative2(double x) const -> double;
    [[nodiscard]] auto integrate(double a, double b) const -> double;

    [[nodiscard]] auto x_min() const -> double;
    [[nodiscard]] auto x_max() const -> double;
    [[nodiscard]] auto size() const -> std::size_t;

   private:
    GslSplinePtr spline_;
    GslAccelPtr accel_;
    std::size_t size_ = 0;
  };

  // 2D bicubic spline interpolation on a regular grid wrapping gsl_spline2d.
  // z-values are in column-major order: z[j * nx + i] = f(x[i], y[j]).
  // Move-only (owns GSL resources).
  class BivariateSpline {
   public:
    BivariateSpline(std::span<const double> x, std::span<const double> y, std::span<const double> z);
    ~BivariateSpline() = default;

    BivariateSpline(BivariateSpline&&) noexcept = default;
    BivariateSpline& operator=(BivariateSpline&&) noexcept = default;
    BivariateSpline(const BivariateSpline&) = delete;
    BivariateSpline& operator=(const BivariateSpline&) = delete;

    [[nodiscard]] auto operator()(double x, double y) const -> double;
    [[nodiscard]] auto deriv_x(double x, double y) const -> double;
    [[nodiscard]] auto deriv_y(double x, double y) const -> double;
    [[nodiscard]] auto deriv_xx(double x, double y) const -> double;
    [[nodiscard]] auto deriv_yy(double x, double y) const -> double;
    [[nodiscard]] auto deriv_xy(double x, double y) const -> double;

   private:
    GslSpline2dPtr spline_;
    GslAccelPtr xacc_;
    GslAccelPtr yacc_;
  };

}  // namespace dft::math

#endif  // DFT_MATH_SPLINE_HPP
