#ifndef DFT_MATH_INTEGRATION_HPP
#define DFT_MATH_INTEGRATION_HPP

#include <functional>
#include <gsl/gsl_integration.h>
#include <memory>
#include <stdexcept>

namespace dft::math {

  // Configuration for GSL numerical integration.
  struct IntegrationConfig {
    double absolute_tolerance{1e-8};
    double relative_tolerance{1e-8};
    int workspace_size{1000};
  };

  // Result of a numerical integration.
  struct IntegrationResult {
    double value;
    double error;
  };

  // RAII wrapper for GSL numerical integration. Accepts any callable
  // f(double) -> double. Owns the GSL workspace and function struct.
  //
  // This is a class (not a struct) because it manages a non-copyable
  // C resource (gsl_integration_workspace).
  class Integrator {
   public:
    explicit Integrator(std::function<double(double)> f, IntegrationConfig config = {})
        : f_(std::move(f)), config_(config),
          workspace_(gsl_integration_workspace_alloc(config.workspace_size), gsl_integration_workspace_free) {
      if (!workspace_) {
        throw std::runtime_error("Integrator: failed to allocate GSL workspace");
      }
      gsl_f_.function = &trampoline;
      gsl_f_.params = this;
    }

    [[nodiscard]] auto integrate(double a, double b) const -> IntegrationResult {
      IntegrationResult r{};
      gsl_integration_qags(
          &gsl_f_,
          a,
          b,
          config_.absolute_tolerance,
          config_.relative_tolerance,
          config_.workspace_size,
          workspace_.get(),
          &r.value,
          &r.error
      );
      return r;
    }

    [[nodiscard]] auto integrate_fast(double a, double b) const -> IntegrationResult {
      IntegrationResult r{};
      std::size_t neval = 0;
      gsl_integration_qng(
          &gsl_f_,
          a,
          b,
          config_.absolute_tolerance,
          config_.relative_tolerance,
          &r.value,
          &r.error,
          &neval
      );
      return r;
    }

    [[nodiscard]] auto integrate_upper_infinite(double a) const -> IntegrationResult {
      IntegrationResult r{};
      gsl_integration_qagiu(
          &gsl_f_,
          a,
          config_.absolute_tolerance,
          config_.relative_tolerance,
          config_.workspace_size,
          workspace_.get(),
          &r.value,
          &r.error
      );
      return r;
    }

    [[nodiscard]] auto integrate_lower_infinite(double b) const -> IntegrationResult {
      IntegrationResult r{};
      gsl_integration_qagil(
          &gsl_f_,
          b,
          config_.absolute_tolerance,
          config_.relative_tolerance,
          config_.workspace_size,
          workspace_.get(),
          &r.value,
          &r.error
      );
      return r;
    }

    [[nodiscard]] auto integrate_infinite() const -> IntegrationResult {
      IntegrationResult r{};
      gsl_integration_qagi(
          &gsl_f_,
          config_.absolute_tolerance,
          config_.relative_tolerance,
          config_.workspace_size,
          workspace_.get(),
          &r.value,
          &r.error
      );
      return r;
    }

   private:
    static double trampoline(double x, void* params) { return static_cast<Integrator*>(params)->f_(x); }

    std::function<double(double)> f_;
    IntegrationConfig config_;
    std::unique_ptr<gsl_integration_workspace, decltype(&gsl_integration_workspace_free)> workspace_;
    mutable gsl_function gsl_f_{};
  };

} // namespace dft::math

#endif // DFT_MATH_INTEGRATION_HPP
