#ifndef DFT_MATH_AUTODIFF_HPP
#define DFT_MATH_AUTODIFF_HPP

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/dual/dual.hpp>
#include <tuple>

namespace dft::math {

  using dual = autodiff::dual;
  using dual2nd = autodiff::dual2nd;
  using dual3rd = autodiff::dual3rd;

  template <typename F> [[nodiscard]] auto derivatives_up_to_1(const F& f, double x) -> std::pair<double, double> {
    dual v = x;
    auto wrapper = [&](dual& w) -> dual {
      return f(w);
    };
    auto [u, du] = autodiff::derivatives(wrapper, autodiff::wrt(v), autodiff::at(v));
    return {static_cast<double>(u), static_cast<double>(du)};
  }

  template <typename F>
  [[nodiscard]] auto derivatives_up_to_2(const F& f, double x) -> std::tuple<double, double, double> {
    dual2nd v = x;
    auto wrapper = [&](dual2nd& w) -> dual2nd {
      return f(w);
    };
    auto [u, du, d2u] = autodiff::derivatives(wrapper, autodiff::wrt(v, v), autodiff::at(v));
    return {static_cast<double>(u), static_cast<double>(du), static_cast<double>(d2u)};
  }

  template <typename F>
  [[nodiscard]] auto derivatives_up_to_3(const F& f, double x) -> std::tuple<double, double, double, double> {
    dual3rd v = x;
    auto wrapper = [&](dual3rd& w) -> dual3rd {
      return f(w);
    };
    auto [u, du, d2u, d3u] = autodiff::derivatives(wrapper, autodiff::wrt(v, v, v), autodiff::at(v));
    return {static_cast<double>(u), static_cast<double>(du), static_cast<double>(d2u), static_cast<double>(d3u)};
  }

} // namespace dft::math

#endif // DFT_MATH_AUTODIFF_HPP
