#ifndef DFT_MATH_SPLINE_H
#define DFT_MATH_SPLINE_H

#include <cstddef>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <memory>
#include <span>

namespace dft::math::spline {

  // ── GSL RAII helpers ──────────────────────────────────────────────────────

  struct GslSplineDeleter {
    void operator()(gsl_spline* p) const noexcept {
      if (p)
        gsl_spline_free(p);
    }
  };

  struct GslSpline2dDeleter {
    void operator()(gsl_spline2d* p) const noexcept {
      if (p)
        gsl_spline2d_free(p);
    }
  };

  struct GslAccelDeleter {
    void operator()(gsl_interp_accel* p) const noexcept {
      if (p)
        gsl_interp_accel_free(p);
    }
  };

  using GslSplinePtr = std::unique_ptr<gsl_spline, GslSplineDeleter>;
  using GslSpline2dPtr = std::unique_ptr<gsl_spline2d, GslSpline2dDeleter>;
  using GslAccelPtr = std::unique_ptr<gsl_interp_accel, GslAccelDeleter>;

  /**
   * @brief 1D natural cubic spline interpolation, wrapping GSL's gsl_spline.
   *
   * Provides function evaluation, first and second derivatives, and definite
   * integration. Move-only (owns GSL resources).
   */
  class CubicSpline {
   public:
    CubicSpline(std::span<const double> x, std::span<const double> y);

    ~CubicSpline() = default;

    CubicSpline(CubicSpline&& other) noexcept = default;
    CubicSpline& operator=(CubicSpline&& other) noexcept = default;

    CubicSpline(const CubicSpline&) = delete;
    CubicSpline& operator=(const CubicSpline&) = delete;

    [[nodiscard]] double operator()(double x) const;
    [[nodiscard]] double derivative(double x) const;
    [[nodiscard]] double derivative2(double x) const;
    [[nodiscard]] double integrate(double a, double b) const;

    [[nodiscard]] double x_min() const;
    [[nodiscard]] double x_max() const;
    [[nodiscard]] std::size_t size() const;

   private:
    GslSplinePtr spline_;
    GslAccelPtr accel_;
    std::size_t size_ = 0;
  };

  /**
   * @brief 2D bicubic spline interpolation on a regular grid, wrapping GSL's gsl_spline2d.
   *
   * The grid is defined by sorted x-coordinates (size nx) and y-coordinates
   * (size ny). The z-values are provided in row-major order: z[i * ny + j]
   * corresponds to (x[i], y[j]).
   *
   * Provides function evaluation and all first/second partial derivatives.
   * Move-only (owns GSL resources).
   */
  class BivariateSpline {
   public:
    BivariateSpline(std::span<const double> x, std::span<const double> y, std::span<const double> z);

    ~BivariateSpline() = default;

    BivariateSpline(BivariateSpline&& other) noexcept = default;
    BivariateSpline& operator=(BivariateSpline&& other) noexcept = default;

    BivariateSpline(const BivariateSpline&) = delete;
    BivariateSpline& operator=(const BivariateSpline&) = delete;

    [[nodiscard]] double operator()(double x, double y) const;
    [[nodiscard]] double deriv_x(double x, double y) const;
    [[nodiscard]] double deriv_y(double x, double y) const;
    [[nodiscard]] double deriv_xx(double x, double y) const;
    [[nodiscard]] double deriv_yy(double x, double y) const;
    [[nodiscard]] double deriv_xy(double x, double y) const;

   private:
    GslSpline2dPtr spline_;
    GslAccelPtr xacc_;
    GslAccelPtr yacc_;
  };

}  // namespace dft::math::spline

#endif  // DFT_MATH_SPLINE_H
