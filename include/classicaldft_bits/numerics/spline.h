#ifndef CLASSICALDFT_NUMERICS_SPLINE_H
#define CLASSICALDFT_NUMERICS_SPLINE_H

#include <cstddef>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_spline2d.h>
#include <span>

namespace dft_core::numerics::spline {

  /**
   * @brief 1D natural cubic spline interpolation, wrapping GSL's gsl_spline.
   *
   * Provides function evaluation, first and second derivatives, and definite
   * integration. Move-only (owns GSL resources).
   */
  class CubicSpline {
   public:
    CubicSpline(std::span<const double> x, std::span<const double> y);

    ~CubicSpline();

    CubicSpline(CubicSpline&& other) noexcept;
    CubicSpline& operator=(CubicSpline&& other) noexcept;

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
    void release() noexcept;

    gsl_spline* spline_ = nullptr;
    gsl_interp_accel* accel_ = nullptr;
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

}  // namespace dft_core::numerics::spline

#endif  // CLASSICALDFT_NUMERICS_SPLINE_H
