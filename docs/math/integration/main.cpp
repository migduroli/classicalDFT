#include "dft.hpp"

#include <iostream>
#include <numbers>
#include <print>

using namespace dft;

int main() {
  // QAGS: definite integral of exp(-x).

  console::info("QAGS: definite integral");
  auto neg_exp = math::Integrator([](double x) { return std::exp(-x); });
  auto r1 = neg_exp.integrate(0.0, -std::log(0.5));
  std::println(std::cout, "  int[0, -ln(0.5)] exp(-x) dx = {:.12}", r1.value);
  std::println(std::cout, "  error = {:.12}", r1.error);

  // QNG: fast non-adaptive.

  console::info("QNG: fast non-adaptive");
  auto r2 = neg_exp.integrate_fast(0.0, -std::log(0.5));
  std::println(std::cout, "  int[0, -ln(0.5)] exp(-x) dx = {:.12}", r2.value);
  std::println(std::cout, "  error = {:.12}", r2.error);

  // QAGIU: upper semi-infinite.

  console::info("QAGIU: upper semi-infinite");
  auto r3 = neg_exp.integrate_upper_infinite(0.0);
  std::println(std::cout, "  int[0, +inf] exp(-x) dx = {:.12}", r3.value);
  std::println(std::cout, "  error = {:.12}", r3.error);

  // QAGIL: lower semi-infinite.

  console::info("QAGIL: lower semi-infinite");
  auto pos_exp = math::Integrator([](double x) { return std::exp(x); });
  auto r4 = pos_exp.integrate_lower_infinite(0.0);
  std::println(std::cout, "  int[-inf, 0] exp(x) dx = {:.12}", r4.value);
  std::println(std::cout, "  error = {:.12}", r4.error);

  // QAGI: full infinite (Gaussian).

  console::info("QAGI: full infinite (normal distribution)");
  auto gaussian = math::Integrator([](double x) { return std::exp(-x * x * 0.5) / std::sqrt(2.0 * std::numbers::pi); });
  auto r5 = gaussian.integrate_infinite();
  std::println(std::cout, "  int[-inf, +inf] normal(x) dx = {:.12}", r5.value);
  std::println(std::cout, "  error = {:.12}", r5.error);
}
