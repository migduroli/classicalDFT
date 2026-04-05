#ifndef DFT_MATH_ARITHMETIC_HPP
#define DFT_MATH_ARITHMETIC_HPP

#include <cmath>
#include <numeric>
#include <span>

namespace dft::math {

  // Kahan-Babuska compensated summation over a span of doubles.
  // Returns the compensated sum with O(1) error independent of N.
  [[nodiscard]] inline auto kahan_sum(std::span<const double> values) -> double {
    double sum = 0.0;
    double c = 0.0;
    for (const double x : values) {
      double y = x - c;
      double t = sum + y;
      c = (t - sum) - y;
      sum = t;
    }
    return sum;
  }

  // Kahan-Babuska-Neumaier compensated summation. More robust than
  // plain Kahan when individual terms are larger than the running sum.
  [[nodiscard]] inline auto neumaier_sum(std::span<const double> values) -> double {
    double sum = 0.0;
    double c = 0.0;
    for (const double x : values) {
      double t = sum + x;
      if (std::abs(sum) >= std::abs(x)) {
        c += (sum - t) + x;
      } else {
        c += (x - t) + sum;
      }
      sum = t;
    }
    return sum + c;
  }

  // Kahan-Babuska-Klein second-order compensated summation. Tracks two
  // levels of error compensation for maximum accuracy.
  [[nodiscard]] inline auto klein_sum(std::span<const double> values) -> double {
    double sum = 0.0;
    double cs = 0.0;
    double ccs = 0.0;
    for (const double x : values) {
      double t = sum + x;
      double c = 0.0;
      if (std::abs(sum) >= std::abs(x)) {
        c = (sum - t) + x;
      } else {
        c = (x - t) + sum;
      }
      sum = t;
      t = cs + c;
      double cc = 0.0;
      if (std::abs(cs) >= std::abs(c)) {
        cc = (cs - t) + c;
      } else {
        cc = (c - t) + cs;
      }
      cs = t;
      ccs += cc;
    }
    return sum + cs + ccs;
  }

} // namespace dft::math

#endif // DFT_MATH_ARITHMETIC_HPP
