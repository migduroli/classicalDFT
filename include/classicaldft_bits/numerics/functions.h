#ifndef CLASSICALDFT_VECTORIZE_H
#define CLASSICALDFT_VECTORIZE_H

#include <concepts>
#include <functional>
#include <vector>

namespace dft_core::utils::functions {

  template <std::floating_point T, typename F>
  requires std::invocable<F, T> std::vector<T> apply_vector_wise(F func, const std::vector<T>& x) {
    std::vector<T> y;
    y.reserve(x.size());
    for (const auto& k : x) {
      y.push_back(std::invoke(func, k));
    }
    return y;
  }

  template <typename Obj, std::floating_point T, typename F>
  requires std::invocable<F, const Obj&, T> std::vector<T> apply_vector_wise(
      const Obj& obj, F method, const std::vector<T>& x
  ) {
    std::vector<T> y;
    y.reserve(x.size());
    for (const auto& k : x) {
      y.push_back(std::invoke(method, obj, k));
    }
    return y;
  }

}  // namespace dft_core::utils::functions
#endif  // CLASSICALDFT_VECTORIZE_H
