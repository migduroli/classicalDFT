#include "cdft/numerics/math.hpp"

namespace cdft::numerics {

  namespace {
    struct FuncWrapper {
      std::function<double(double)>* func;
    };

    double gsl_trampoline(double x, void* params) {
      auto* wrapper = static_cast<FuncWrapper*>(params);
      return (*wrapper->func)(x);
    }
  }  // namespace

  IntegrationResult integrate_qags(
      std::function<double(double)> func, double lower, double upper,
      double relative_tolerance, double absolute_tolerance, int workspace_size) {
    auto* ws = gsl_integration_workspace_alloc(static_cast<size_t>(workspace_size));
    FuncWrapper wrapper{&func};
    gsl_function gsl_func{gsl_trampoline, &wrapper};

    IntegrationResult result;
    gsl_integration_qags(&gsl_func, lower, upper, absolute_tolerance, relative_tolerance,
                         static_cast<size_t>(workspace_size), ws, &result.value, &result.error);
    gsl_integration_workspace_free(ws);
    return result;
  }

  IntegrationResult integrate_qng(
      std::function<double(double)> func, double lower, double upper,
      double relative_tolerance, double absolute_tolerance) {
    FuncWrapper wrapper{&func};
    gsl_function gsl_func{gsl_trampoline, &wrapper};
    std::size_t neval = 0;

    IntegrationResult result;
    gsl_integration_qng(&gsl_func, lower, upper, absolute_tolerance, relative_tolerance,
                        &result.value, &result.error, &neval);
    return result;
  }

  IntegrationResult integrate_semi_infinite(
      std::function<double(double)> func, double lower,
      double relative_tolerance, double absolute_tolerance, int workspace_size) {
    auto* ws = gsl_integration_workspace_alloc(static_cast<size_t>(workspace_size));
    FuncWrapper wrapper{&func};
    gsl_function gsl_func{gsl_trampoline, &wrapper};

    IntegrationResult result;
    gsl_integration_qagiu(&gsl_func, lower, absolute_tolerance, relative_tolerance,
                          static_cast<size_t>(workspace_size), ws, &result.value, &result.error);
    gsl_integration_workspace_free(ws);
    return result;
  }

}  // namespace cdft::numerics
