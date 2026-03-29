#pragma once

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/dual.hpp>
#include <tuple>

namespace cdft {

  using dual = autodiff::dual;
  using dual2nd = autodiff::dual2nd;
  using dual3rd = autodiff::dual3rd;

  // ── Convenience: compute f and its first 3 derivatives in one call ────────

  template <typename F>
  [[nodiscard]] auto derivatives_up_to_3(F&& f, double x) {
    dual3rd v = x;
    auto [u, du, d2u, d3u] = autodiff::derivatives(std::forward<F>(f), autodiff::wrt(v, v, v), autodiff::at(v));
    return std::tuple{static_cast<double>(u), static_cast<double>(du),
                      static_cast<double>(d2u), static_cast<double>(d3u)};
  }

  template <typename F>
  [[nodiscard]] auto derivatives_up_to_1(F&& f, double x) {
    dual v = x;
    auto [u, du] = autodiff::derivatives(std::forward<F>(f), autodiff::wrt(v), autodiff::at(v));
    return std::pair{static_cast<double>(u), static_cast<double>(du)};
  }

}  // namespace cdft
