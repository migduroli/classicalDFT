#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <gsl/gsl_integration.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cdft::numerics {

  // ── Integration result (replaces mutable class state) ─────────────────────

  struct IntegrationResult {
    double value = 0.0;
    double error = 0.0;
  };

  // ── GSL integration wrappers (stateless free functions) ───────────────────

  [[nodiscard]] IntegrationResult integrate_qags(
      std::function<double(double)> func,
      double lower,
      double upper,
      double relative_tolerance = 1e-5,
      double absolute_tolerance = 1e-5,
      int workspace_size = 1000
  );

  [[nodiscard]] IntegrationResult integrate_qng(
      std::function<double(double)> func,
      double lower,
      double upper,
      double relative_tolerance = 1e-5,
      double absolute_tolerance = 1e-5
  );

  [[nodiscard]] IntegrationResult integrate_semi_infinite(
      std::function<double(double)> func,
      double lower,
      double relative_tolerance = 1e-5,
      double absolute_tolerance = 1e-5,
      int workspace_size = 1000
  );

  // ── Legacy Integrator class (thin wrapper, preserves existing API) ────────

  template <class T, class x_type = double>
  class Integrator {
   public:
    using class_method = std::function<x_type(const T&, x_type)>;

    explicit Integrator(
        const T& problem,
        class_method method,
        double relative_error = 1e-5,
        double absolute_error = 1e-5,
        int workspace_size = 1000
    )
        : relative_error_(relative_error),
          absolute_error_(absolute_error),
          workspace_size_(workspace_size),
          problem_(problem),
          method_(std::move(method)) {
      workspace_.reset(gsl_integration_workspace_alloc(workspace_size_));
      gsl_func_.function = &integrand_trampoline;
      gsl_func_.params = this;
    }

    [[nodiscard]] double definite_integral(double lower, double upper) const {
      double result = 0.0, error = 0.0;
      gsl_integration_qags(&gsl_func_, lower, upper, absolute_error_, relative_error_,
                           workspace_size_, workspace_.get(), &result, &error);
      error_ = error;
      result_ = result;
      return result;
    }

    [[nodiscard]] double semi_infinite_integral(double lower) const {
      double result = 0.0, error = 0.0;
      gsl_integration_qagiu(&gsl_func_, lower, absolute_error_, relative_error_,
                            workspace_size_, workspace_.get(), &result, &error);
      error_ = error;
      result_ = result;
      return result;
    }

    [[nodiscard]] double numerical_error() const { return error_; }
    [[nodiscard]] double numerical_result() const { return result_; }

   private:
    static x_type integrand_trampoline(x_type x, void* params) {
      auto* self = static_cast<Integrator*>(params);
      return self->method_(self->problem_, x);
    }

    double relative_error_;
    double absolute_error_;
    int workspace_size_;
    const T& problem_;
    class_method method_;
    mutable double error_ = 0.0;
    mutable double result_ = 0.0;
    mutable gsl_function gsl_func_{};
    std::unique_ptr<gsl_integration_workspace, decltype(&gsl_integration_workspace_free)>
        workspace_{nullptr, gsl_integration_workspace_free};
  };

  // ── Compensated summation (Kahan-Babuska-Neumaier) ────────────────────────

  class CompensatedSum {
   public:
    CompensatedSum& operator+=(double value) noexcept {
      double t = sum_ + value;
      if (std::abs(sum_) >= std::abs(value)) {
        compensation_ += (sum_ - t) + value;
      } else {
        compensation_ += (value - t) + sum_;
      }
      sum_ = t;
      return *this;
    }

    [[nodiscard]] double value() const noexcept { return sum_ + compensation_; }
    [[nodiscard]] double error() const noexcept { return std::abs(compensation_); }

    void reset() noexcept {
      sum_ = 0.0;
      compensation_ = 0.0;
    }

   private:
    double sum_ = 0.0;
    double compensation_ = 0.0;
  };

  // ── Standalone summation functions ────────────────────────────────────────

  struct SumResult {
    double sum = 0.0;
    double error = 0.0;
  };

  [[nodiscard]] inline SumResult compensated_sum(const std::vector<double>& values) {
    CompensatedSum cs;
    for (double v : values) cs += v;
    return {cs.value(), cs.error()};
  }

  [[nodiscard]] inline double standard_sum(const std::vector<double>& values) {
    double s = 0.0;
    for (double v : values) s += v;
    return s;
  }

}  // namespace cdft::numerics
