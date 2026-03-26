#ifndef DFT_MATH_AUTODIFF_H
#define DFT_MATH_AUTODIFF_H

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/dual.hpp>
#include <tuple>

namespace dft::math {

  using dual = autodiff::dual;
  using dual2nd = autodiff::dual2nd;
  using dual3rd = autodiff::dual3rd;

  /**
   * @brief Compute f(x) and its first derivative f'(x) in a single forward pass.
   */
  template <typename F>
  [[nodiscard]] auto derivatives_up_to_1(F&& f, double x) -> std::pair<double, double> {
    dual v = x;
    auto [u, du] = autodiff::derivatives(std::forward<F>(f), autodiff::wrt(v), autodiff::at(v));
    return {static_cast<double>(u), static_cast<double>(du)};
  }

  /**
   * @brief Compute f(x) and its first two derivatives in a single forward pass.
   */
  template <typename F>
  [[nodiscard]] auto derivatives_up_to_2(F&& f, double x) -> std::tuple<double, double, double> {
    dual2nd v = x;
    auto [u, du, d2u] = autodiff::derivatives(std::forward<F>(f), autodiff::wrt(v, v), autodiff::at(v));
    return {static_cast<double>(u), static_cast<double>(du), static_cast<double>(d2u)};
  }

  /**
   * @brief Compute f(x) and its first three derivatives in a single forward pass.
   */
  template <typename F>
  [[nodiscard]] auto derivatives_up_to_3(F&& f, double x) -> std::tuple<double, double, double, double> {
    dual3rd v = x;
    auto [u, du, d2u, d3u] = autodiff::derivatives(std::forward<F>(f), autodiff::wrt(v, v, v), autodiff::at(v));
    return {static_cast<double>(u), static_cast<double>(du), static_cast<double>(d2u), static_cast<double>(d3u)};
  }

}  // namespace dft::math

#endif  // DFT_MATH_AUTODIFF_H
